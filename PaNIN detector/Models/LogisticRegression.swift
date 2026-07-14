import Foundation
import Accelerate

/// Multinomial logistic regression (softmax classifier) trained with batch
/// gradient descent. All matrix multiplies go through Accelerate / cBLAS, so
/// training thousands of patches takes well under a second.
enum LogisticRegression {

    struct Trained: Sendable {
        let weights: [Float]   // D * K, row-major
        let biases: [Float]    // K
        let featureDim: Int
        let classCount: Int
        let finalLoss: Float
    }

    nonisolated static func train(
        flatX: [Float],
        labels: [Int],
        N: Int,
        D: Int,
        K: Int,
        iterations: Int = 200,
        learningRate: Float = 0.1,
        l2: Float = 0.001,
        progress: (@Sendable (Int, Int) -> Void)? = nil
    ) -> Trained {
        precondition(flatX.count == N * D, "flatX has wrong size")
        precondition(labels.count == N, "labels has wrong count")

        var W = [Float](repeating: 0, count: D * K)
        var b = [Float](repeating: 0, count: K)

        // Reused buffers
        var z  = [Float](repeating: 0, count: N * K)
        var dW = [Float](repeating: 0, count: D * K)
        var db = [Float](repeating: 0, count: K)
        var finalLoss: Float = 0

        for iter in 0..<iterations {
            // z = X @ W   (N x K)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Int32(N), Int32(K), Int32(D),
                1.0, flatX, Int32(D),
                W, Int32(K),
                0.0, &z, Int32(K)
            )
            // z += b   (broadcast)
            for i in 0..<N {
                let row = i * K
                for k in 0..<K { z[row + k] += b[k] }
            }

            // Softmax per row (stable)
            for i in 0..<N {
                let row = i * K
                var rowMax = z[row]
                for k in 1..<K where z[row + k] > rowMax { rowMax = z[row + k] }
                var rowSum: Float = 0
                for k in 0..<K {
                    let v = expf(z[row + k] - rowMax)
                    z[row + k] = v
                    rowSum += v
                }
                let invSum: Float = 1.0 / rowSum
                for k in 0..<K { z[row + k] *= invSum }
            }

            // Cross-entropy loss every 25 iters (cheap)
            if iter % 25 == 0 || iter == iterations - 1 {
                var loss: Float = 0
                for i in 0..<N {
                    let p = max(z[i * K + labels[i]], 1e-9)
                    loss -= logf(p)
                }
                finalLoss = loss / Float(N)
            }

            // gradient = p - one_hot_y   (overwrite z)
            for i in 0..<N { z[i * K + labels[i]] -= 1.0 }

            // dW = X^T @ grad / N   (D x K)
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                Int32(D), Int32(K), Int32(N),
                1.0 / Float(N), flatX, Int32(D),
                z, Int32(K),
                0.0, &dW, Int32(K)
            )
            // L2 regularization
            if l2 > 0 {
                let twoL2 = 2 * l2
                for i in 0..<(D * K) { dW[i] += twoL2 * W[i] }
            }

            // db = mean(grad, axis=0)
            for k in 0..<K { db[k] = 0 }
            for i in 0..<N {
                let row = i * K
                for k in 0..<K { db[k] += z[row + k] }
            }
            let invN: Float = 1.0 / Float(N)
            for k in 0..<K { db[k] *= invN }

            // Update
            for i in 0..<(D * K) { W[i] -= learningRate * dW[i] }
            for k in 0..<K { b[k] -= learningRate * db[k] }

            progress?(iter + 1, iterations)
        }

        return Trained(weights: W, biases: b, featureDim: D, classCount: K, finalLoss: finalLoss)
    }

    nonisolated static func predict(
        features: [Float],
        weights: [Float],
        biases: [Float],
        classCount: Int
    ) -> (index: Int, probs: [Float]) {
        let D = features.count
        let K = classCount
        var z = [Float](repeating: 0, count: K)
        for k in 0..<K {
            var sum: Float = biases[k]
            for d in 0..<D { sum += features[d] * weights[d * K + k] }
            z[k] = sum
        }
        var zMax: Float = z[0]
        for k in 1..<K where z[k] > zMax { zMax = z[k] }
        var sum: Float = 0
        for k in 0..<K {
            let v = expf(z[k] - zMax)
            z[k] = v
            sum += v
        }
        let inv: Float = 1.0 / sum
        for k in 0..<K { z[k] *= inv }
        var maxIdx = 0
        for k in 1..<K where z[k] > z[maxIdx] { maxIdx = k }
        return (maxIdx, z)
    }
}
