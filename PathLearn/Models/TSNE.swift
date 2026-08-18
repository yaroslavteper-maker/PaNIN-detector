import Foundation
import Accelerate
import SwiftData

/// Symmetric t-SNE (van der Maaten 2008) over the bank's feature vectors.
/// Pure-Swift inner loop with BLAS only for the initial pairwise dot product
/// — fast enough up to a few thousand patches on Apple Silicon.
enum TSNE {

    struct Config: Sendable, Codable {
        var perplexity: Double = 30
        var iterations: Int = 1000
        var earlyExaggeration: Double = 12
        var earlyExaggerationIters: Int = 250
        var learningRate: Double = 200
        var initialMomentum: Double = 0.5
        var finalMomentum: Double = 0.8
        var momentumSwitchIter: Int = 250
        var seed: UInt64 = 42
    }

    struct LoadedBank: Sendable {
        let features: [Float]       // N rows × D cols, row-major
        let dim: Int
        let labels: [String]        // length N
        let slideNames: [String]    // length N
        let extractorIdentity: ExtractorIdentity
        let totalAvailable: Int     // count before subsampling
    }

    enum TSNEError: Error, CustomStringConvertible {
        case notEnoughPoints(Int)
        case mixedExtractorIdentities
        case mixedFeatureDims
        case unsupportedElementType(Int)
        case perplexityTooLarge(Double, Int)
        var description: String {
            switch self {
            case .notEnoughPoints(let n):
                return "Need at least 10 patches for t-SNE (have \(n))."
            case .mixedExtractorIdentities:
                return "Bank contains patches from multiple extractors. Pick one in the t-SNE sheet."
            case .mixedFeatureDims:
                return "Patches have different feature dimensions. Clear the bank and re-extract."
            case .unsupportedElementType(let e):
                return "Unsupported feature element type (\(e)). Clear and re-extract."
            case .perplexityTooLarge(let p, let n):
                return "Perplexity \(Int(p)) is too high for \(n) points (must be < N/3)."
            }
        }
    }

    // MARK: - Bank loader

    /// Pull feature vectors out of the bank. Optionally filter to a single
    /// extractor and stratified-subsample down to `maxCount` (per-class
    /// proportional, so rare classes don't drown).
    @MainActor
    static func loadBank(context: ModelContext,
                        extractorIdentity: String? = nil,
                        maxCount: Int = 5_000,
                        maxWhiteFraction: Float? = nil,
                        minNuclei: Int? = nil,
                        seed: UInt64 = 42) throws -> LoadedBank {
        let descriptor = FetchDescriptor<MLPatch>()
        var patches = try context.fetch(descriptor)
        if let extractorIdentity {
            patches = patches.filter { $0.resolvedExtractorIdentity.stringForm == extractorIdentity }
        }
        if let cutoff = maxWhiteFraction {
            let before = patches.count
            patches = patches.filter { $0.whiteFraction <= cutoff }
            print("[tsne] whiteness filter ≤\(cutoff): \(before) → \(patches.count) patches")
        }
        if let minN = minNuclei {
            let before = patches.count
            // Legacy patches with no measured count (-1) pass, mirroring the
            // whiteness filter's treatment of unscored patches.
            patches = patches.filter { $0.nucleusCount < 0 || $0.nucleusCount >= minN }
            print("[tsne] nuclei filter ≥\(minN): \(before) → \(patches.count) patches")
        }
        guard patches.count >= 10 else { throw TSNEError.notEnoughPoints(patches.count) }

        // Validate single extractor + dim.
        let firstID = patches[0].resolvedExtractorIdentity
        guard patches.allSatisfy({ $0.resolvedExtractorIdentity == firstID }) else {
            throw TSNEError.mixedExtractorIdentities
        }
        let dim = patches[0].featureDim
        guard patches.allSatisfy({ $0.featureDim == dim }) else {
            throw TSNEError.mixedFeatureDims
        }

        let totalAvailable = patches.count

        // Stratified subsample.
        if patches.count > maxCount {
            patches = stratifiedSubsample(patches, target: maxCount, seed: seed)
        }
        print("[tsne] loaded \(patches.count) of \(totalAvailable) patches, dim=\(dim), extractor=\(firstID.stringForm)")

        // Decode features into one flat row-major buffer.
        var features = [Float](repeating: 0, count: patches.count * dim)
        var labels: [String] = []
        var slideNames: [String] = []
        labels.reserveCapacity(patches.count)
        slideNames.reserveCapacity(patches.count)
        for (i, p) in patches.enumerated() {
            let row = try decodeFeatures(patch: p, dim: dim)
            for d in 0..<dim {
                features[i * dim + d] = row[d]
            }
            labels.append(p.classification)
            slideNames.append(p.slideName)
        }
        return LoadedBank(
            features: features,
            dim: dim,
            labels: labels,
            slideNames: slideNames,
            extractorIdentity: firstID,
            totalAvailable: totalAvailable
        )
    }

    private static func stratifiedSubsample(_ patches: [MLPatch],
                                            target: Int,
                                            seed: UInt64) -> [MLPatch] {
        let byClass = Dictionary(grouping: patches.indices, by: { patches[$0].classification })
        let totalCount = Double(patches.count)
        var rng = SplitMix64(seed: seed)
        var selected: [Int] = []
        for (_, indices) in byClass {
            let share = Double(indices.count) / totalCount
            let want = max(1, Int((Double(target) * share).rounded()))
            if indices.count <= want {
                selected.append(contentsOf: indices)
            } else {
                var pool = indices
                for _ in 0..<want {
                    let pick = Int(rng.next() % UInt64(pool.count))
                    selected.append(pool[pick])
                    pool.swapAt(pick, pool.count - 1)
                    pool.removeLast()
                }
            }
        }
        return selected.map { patches[$0] }
    }

    private static func decodeFeatures(patch: MLPatch, dim: Int) throws -> [Float] {
        switch patch.featureElementType {
        case 1:
            let count = patch.featureData.count / MemoryLayout<Float>.size
            var out = [Float](repeating: 0, count: count)
            _ = out.withUnsafeMutableBytes { dst in
                patch.featureData.copyBytes(to: dst)
            }
            return out
        case 2:
            let count = patch.featureData.count / MemoryLayout<Float16>.size
            var f16 = [Float16](repeating: 0, count: count)
            _ = f16.withUnsafeMutableBytes { dst in
                patch.featureData.copyBytes(to: dst)
            }
            return f16.map { Float($0) }
        default:
            throw TSNEError.unsupportedElementType(patch.featureElementType)
        }
    }

    // MARK: - Algorithm

    /// Run t-SNE. Returns N (x, y) points in the same order as `features`.
    /// Progress callback fires up to ~50 times across the run.
    nonisolated static func embed(features: [Float],
                                  dim: Int,
                                  count N: Int,
                                  config: Config,
                                  progress: @Sendable (Int, Int, String) -> Void) throws -> [(Float, Float)] {
        guard N >= 10 else { throw TSNEError.notEnoughPoints(N) }
        guard config.perplexity * 3 < Double(N) else {
            throw TSNEError.perplexityTooLarge(config.perplexity, N)
        }
        try Task.checkCancellation()

        progress(0, config.iterations, "Computing pairwise distances…")
        let D = pairwiseSquaredDistances(features, dim: dim, count: N)

        try Task.checkCancellation()
        progress(0, config.iterations, "Solving \(N) perplexities…")
        let P = symmetricAffinities(D: D, N: N, perplexity: config.perplexity)

        // Init Y ~ N(0, 1e-4).
        var rng = SplitMix64(seed: config.seed)
        var Y = [Float](repeating: 0, count: N * 2)
        for i in 0..<(N * 2) {
            Y[i] = Float(rng.nextGaussian() * 1e-4)
        }
        var dY = [Float](repeating: 0, count: N * 2)
        var iY = [Float](repeating: 0, count: N * 2)

        // Work buffers allocated once, reused every iter — avoids ~100 MB of
        // churn per step at N=5000 and lets BLAS / vDSP do the hot loop.
        var dotYY = [Float](repeating: 0, count: N * N)
        var num   = [Float](repeating: 0, count: N * N)
        var A     = [Float](repeating: 0, count: N * N)
        var yNorm2 = [Float](repeating: 0, count: N)
        var rowSumA = [Float](repeating: 0, count: N)
        var AY = [Float](repeating: 0, count: N * 2)

        // Main loop.
        for iter in 0..<config.iterations {
            try Task.checkCancellation()

            // 1. yNorm2[i] = ||y_i||²
            for i in 0..<N {
                let a = Y[i * 2], b = Y[i * 2 + 1]
                yNorm2[i] = a * a + b * b
            }

            // 2. dotYY = Y Y^T (BLAS, N×2 × 2×N → N×N)
            Y.withUnsafeBufferPointer { yp in
                dotYY.withUnsafeMutableBufferPointer { dp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(N), Int32(N), Int32(2),
                                1.0, yp.baseAddress, Int32(2),
                                yp.baseAddress, Int32(2),
                                0.0, dp.baseAddress, Int32(N))
                }
            }

            // 3. num = 1 + yNorm2[i] + yNorm2[j] - 2*dotYY[i,j]
            //    Build into `num`, then reciprocate.
            var negTwo: Float = -2
            num.withUnsafeMutableBufferPointer { np in
                dotYY.withUnsafeBufferPointer { dp in
                    vDSP_vsmul(dp.baseAddress!, 1, &negTwo,
                               np.baseAddress!, 1, vDSP_Length(N * N))
                }
            }
            // Add yNorm2 to each row (broadcasts ||y_j||² across i).
            num.withUnsafeMutableBufferPointer { np in
                yNorm2.withUnsafeBufferPointer { yp in
                    for i in 0..<N {
                        vDSP_vadd(np.baseAddress! + i * N, 1,
                                  yp.baseAddress!, 1,
                                  np.baseAddress! + i * N, 1,
                                  vDSP_Length(N))
                    }
                }
            }
            // Add yNorm2[i] (a scalar) to each row (broadcasts ||y_i||² across j).
            num.withUnsafeMutableBufferPointer { np in
                for i in 0..<N {
                    var v: Float = yNorm2[i] + 1   // fuse the +1 here for free
                    vDSP_vsadd(np.baseAddress! + i * N, 1, &v,
                               np.baseAddress! + i * N, 1,
                               vDSP_Length(N))
                }
            }
            // Element-wise reciprocal: num = 1 / num.
            var nn = Int32(N * N)
            num.withUnsafeMutableBufferPointer { np in
                vvrecf(np.baseAddress!, np.baseAddress!, &nn)
            }
            // Zero diagonal.
            for i in 0..<N { num[i * N + i] = 0 }

            // 4. Z = sum(num)
            var Z: Float = 0
            num.withUnsafeBufferPointer { np in
                vDSP_sve(np.baseAddress!, 1, &Z, vDSP_Length(N * N))
            }
            let invZ: Float = 1.0 / max(Z, 1e-12)
            let exag: Float = iter < config.earlyExaggerationIters
                ? Float(config.earlyExaggeration) : 1.0

            // 5. A = num ⊙ (P*exag - num*invZ)
            //    Compute as: A = P*exag - invZ*num, then A *= num.
            var exagF = exag
            A.withUnsafeMutableBufferPointer { ap in
                P.withUnsafeBufferPointer { pp in
                    vDSP_vsmul(pp.baseAddress!, 1, &exagF,
                               ap.baseAddress!, 1, vDSP_Length(N * N))
                }
            }
            // A += (-invZ) * num (fused via vsma).
            var negInvZ: Float = -invZ
            A.withUnsafeMutableBufferPointer { ap in
                num.withUnsafeBufferPointer { np in
                    vDSP_vsma(np.baseAddress!, 1, &negInvZ,
                              ap.baseAddress!, 1, ap.baseAddress!, 1,
                              vDSP_Length(N * N))
                }
            }
            // A *= num element-wise.
            A.withUnsafeMutableBufferPointer { ap in
                num.withUnsafeBufferPointer { np in
                    vDSP_vmul(ap.baseAddress!, 1, np.baseAddress!, 1,
                              ap.baseAddress!, 1, vDSP_Length(N * N))
                }
            }

            // 6. rowSumA[i] = Σ_j A[i,j]
            A.withUnsafeBufferPointer { ap in
                for i in 0..<N {
                    var s: Float = 0
                    vDSP_sve(ap.baseAddress! + i * N, 1, &s, vDSP_Length(N))
                    rowSumA[i] = s
                }
            }

            // 7. AY = A @ Y (BLAS, N×N × N×2 → N×2)
            A.withUnsafeBufferPointer { ap in
                Y.withUnsafeBufferPointer { yp in
                    AY.withUnsafeMutableBufferPointer { ayp in
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    Int32(N), Int32(2), Int32(N),
                                    1.0, ap.baseAddress, Int32(N),
                                    yp.baseAddress, Int32(2),
                                    0.0, ayp.baseAddress, Int32(2))
                    }
                }
            }

            // 8. dY[i,d] = 4 * (Y[i,d] * rowSumA[i] - AY[i,d])
            for i in 0..<N {
                let s = rowSumA[i]
                dY[i * 2]     = 4 * (Y[i * 2]     * s - AY[i * 2])
                dY[i * 2 + 1] = 4 * (Y[i * 2 + 1] * s - AY[i * 2 + 1])
            }

            // 9. Update Y with momentum.
            let momentum: Float = iter < config.momentumSwitchIter
                ? Float(config.initialMomentum) : Float(config.finalMomentum)
            let lr = Float(config.learningRate)
            for k in 0..<(N * 2) {
                iY[k] = momentum * iY[k] - lr * dY[k]
                Y[k] += iY[k]
            }

            // 10. Recentre to keep things numerically stable.
            var mx: Float = 0, my: Float = 0
            for i in 0..<N {
                mx += Y[i * 2]
                my += Y[i * 2 + 1]
            }
            mx /= Float(N); my /= Float(N)
            for i in 0..<N {
                Y[i * 2] -= mx
                Y[i * 2 + 1] -= my
            }

            if iter % 20 == 0 || iter == config.iterations - 1 {
                progress(iter + 1, config.iterations,
                         "Iteration \(iter + 1) / \(config.iterations)")
            }
        }

        var result: [(Float, Float)] = []
        result.reserveCapacity(N)
        for i in 0..<N {
            result.append((Y[i * 2], Y[i * 2 + 1]))
        }
        return result
    }

    // MARK: - Math helpers

    /// D[i,j] = ||x_i - x_j||². Uses BLAS for the X X^T product.
    private static func pairwiseSquaredDistances(_ X: [Float],
                                                 dim D: Int,
                                                 count N: Int) -> [Float] {
        var dot = [Float](repeating: 0, count: N * N)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(N), Int32(N), Int32(D),
                    1.0, X, Int32(D),
                    X, Int32(D),
                    0.0, &dot, Int32(N))
        var norm2 = [Float](repeating: 0, count: N)
        for i in 0..<N {
            var s: Float = 0
            for d in 0..<D {
                let v = X[i * D + d]
                s += v * v
            }
            norm2[i] = s
        }
        var result = [Float](repeating: 0, count: N * N)
        for i in 0..<N {
            for j in 0..<N {
                result[i * N + j] = max(0, norm2[i] + norm2[j] - 2 * dot[i * N + j])
            }
        }
        return result
    }

    /// Per-row binary search for σ_i that yields the target perplexity, then
    /// symmetrise: P_ij = (P_{i|j} + P_{j|i}) / (2N), floored to 1e-12.
    private static func symmetricAffinities(D: [Float], N: Int, perplexity: Double) -> [Float] {
        let logPerp = log(perplexity)
        var P = [Float](repeating: 0, count: N * N)
        let tol = 1e-5
        let maxIters = 50

        var Prow = [Double](repeating: 0, count: N)
        for i in 0..<N {
            var betaLow = -Double.infinity
            var betaHigh = Double.infinity
            var beta = 1.0

            for _ in 0..<maxIters {
                var sumP = 0.0
                for j in 0..<N {
                    if i == j {
                        Prow[j] = 0
                    } else {
                        Prow[j] = exp(-Double(D[i * N + j]) * beta)
                        sumP += Prow[j]
                    }
                }
                if sumP < 1e-12 { sumP = 1e-12 }
                var H = 0.0
                let invSum = 1.0 / sumP
                for j in 0..<N {
                    let p = Prow[j] * invSum
                    if p > 1e-12 { H -= p * log(p) }
                }
                let diff = H - logPerp
                if abs(diff) < tol { break }
                if diff > 0 {
                    betaLow = beta
                    beta = betaHigh.isFinite ? (beta + betaHigh) / 2 : beta * 2
                } else {
                    betaHigh = beta
                    beta = betaLow.isFinite ? (beta + betaLow) / 2 : beta / 2
                }
            }

            // Final normalised row using the converged beta.
            var sumP = 0.0
            for j in 0..<N {
                if i != j {
                    Prow[j] = exp(-Double(D[i * N + j]) * beta)
                    sumP += Prow[j]
                } else {
                    Prow[j] = 0
                }
            }
            let invSum = 1.0 / max(sumP, 1e-12)
            for j in 0..<N {
                P[i * N + j] = Float(Prow[j] * invSum)
            }
        }
        // Symmetrise + floor.
        var sym = [Float](repeating: 0, count: N * N)
        let scale: Float = 1.0 / (2 * Float(N))
        for i in 0..<N {
            for j in 0..<N {
                sym[i * N + j] = max((P[i * N + j] + P[j * N + i]) * scale, 1e-12)
            }
        }
        return sym
    }
}

// MARK: - PRNG

/// Seeded splitmix64 — fine for t-SNE init + subsample.
private struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    /// Standard normal via Box-Muller.
    mutating func nextGaussian() -> Double {
        let u1 = Double(next() >> 11) * (1.0 / Double(1 << 53))
        let u2 = Double(next() >> 11) * (1.0 / Double(1 << 53))
        let u1c = max(u1, 1e-300)
        return sqrt(-2 * log(u1c)) * cos(2 * .pi * u2)
    }
}
