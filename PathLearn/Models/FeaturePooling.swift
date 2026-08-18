import Foundation

/// Aggregates a *bag* of per-patch feature vectors (each of dimension `D`) into
/// a single fixed-length descriptor for a whole annotation. Concatenates the
/// element-wise mean, max, and population standard deviation, so the output
/// dimension is `3 * D`.
///
/// Rationale: PanIN *grade* is a per-lesion label, not a per-cell one, and a
/// pathologist grades the whole duct — taking the worst focus. Classifying one
/// small tile at a time throws that structure away. Pooling lets the classifier
/// see the *distribution* of morphologies across the lesion: the mean captures
/// the dominant appearance, the max captures the worst/most-atypical focus, and
/// the std captures heterogeneity (grade 3 lesions tend to be more variable).
nonisolated enum FeaturePooling {
    /// Stored on the classifier / model file so prediction knows the model
    /// expects pooled features. Bump the string if the scheme changes.
    static let meanMaxStdID = "meanmaxstd"

    /// Output-dimension multiplier for the `meanMaxStd` scheme.
    static let meanMaxStdMultiplier = 3

    /// Pool one bag of equal-length vectors into `mean ‖ max ‖ std`.
    /// Returns `nil` if the bag is empty or vectors have inconsistent length.
    static func meanMaxStd(_ vectors: [[Float]]) -> [Float]? {
        guard let first = vectors.first else { return nil }
        let d = first.count
        guard d > 0, vectors.allSatisfy({ $0.count == d }) else { return nil }
        let n = Float(vectors.count)

        var mean = [Float](repeating: 0, count: d)
        var maxV = first
        for v in vectors {
            for i in 0..<d {
                mean[i] += v[i]
                if v[i] > maxV[i] { maxV[i] = v[i] }
            }
        }
        for i in 0..<d { mean[i] /= n }

        var std = [Float](repeating: 0, count: d)
        for v in vectors {
            for i in 0..<d {
                let diff = v[i] - mean[i]
                std[i] += diff * diff
            }
        }
        for i in 0..<d { std[i] = (std[i] / n).squareRoot() }

        return mean + maxV + std
    }
}
