import Foundation
import CoreGraphics

/// Stable identity of a feature extractor. Two extractors with the same
/// identity produce comparable vectors and can share a bank or a model.
/// Persisted as a compact `"kind:name:r<revision>"` string on every patch
/// + classifier so we can refuse to mix incompatible features.
struct ExtractorIdentity: Hashable, Sendable, Codable {
    /// `"vision"` for the built-in Vision feature print, `"coreml"` for any
    /// user-supplied `.mlpackage`.
    let kind: String
    /// Human-readable name, e.g. `"vision"`, `"uni-v1"`, `"virchow-v2"`.
    let name: String
    /// Numeric revision for the same `(kind, name)` family.
    let revision: Int

    var stringForm: String { "\(kind):\(name):r\(revision)" }

    static func legacyVision(revision: Int) -> ExtractorIdentity {
        ExtractorIdentity(kind: "vision", name: "vision", revision: revision)
    }

    /// Parse `"vision:vision:r2"` back into an `ExtractorIdentity`. Returns
    /// `nil` if the string isn't in that shape — callers should fall back to
    /// a legacy Vision identity.
    static func parse(_ string: String) -> ExtractorIdentity? {
        let parts = string.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 3, parts[2].hasPrefix("r"),
              let r = Int(parts[2].dropFirst()) else { return nil }
        return ExtractorIdentity(
            kind: String(parts[0]),
            name: String(parts[1]),
            revision: r
        )
    }
}

/// One extracted feature vector + provenance. `elementType` is the raw value
/// of `VNElementType` (1 = float32, 2 = float16) so existing decode paths
/// don't need to change.
struct FeatureExtractionResult: Sendable {
    let data: Data
    let dim: Int
    let elementType: Int
    let identity: ExtractorIdentity
}

/// Anything that can turn a CGImage into a feature vector. Phase-2 plumbing
/// for swapping in domain-specific extractors (UNI, Virchow) alongside the
/// built-in Vision pipeline. Named `AnyFeatureExtractor` to leave the existing
/// `FeatureExtractor` class as the Vision concrete type.
protocol AnyFeatureExtractor: Sendable {
    var identity: ExtractorIdentity { get }
    var inputSize: CGSize { get }
    var featureDim: Int { get }
    func extract(_ image: CGImage) async throws -> FeatureExtractionResult
}
