import Foundation
import CoreGraphics

/// Pluggable image-feature extractor.
///
/// Stock implementations:
/// - `VisionFeatureExtractor` — Apple's built-in `VNGenerateImageFeaturePrintRequest`.
/// - `CoreMLFeatureExtractor`  — any `.mlpackage` whose output is a feature vector.
public protocol FeatureExtractor: Sendable {
    /// Stable identity. Embed this in any persisted row so a bank can refuse
    /// to mix features from different extractors (they aren't comparable).
    var identity: ExtractorIdentity { get }

    /// Native input dimensions; implementations resize internally to this.
    var inputSize: CGSize { get }

    /// Output feature vector length.
    var featureDim: Int { get }

    /// Extract a feature vector from a single patch.
    func extract(_ image: CGImage) async throws -> FeatureResult
}

public struct FeatureResult: Sendable {
    /// Raw bytes of the feature vector. Interpret as `Float` (4 bytes per element)
    /// or `Float16` (2 bytes) per `elementType`.
    public let data: Data
    public let dim: Int
    public let elementType: ElementType

    public init(data: Data, dim: Int, elementType: ElementType) {
        self.data = data
        self.dim = dim
        self.elementType = elementType
    }
}

public enum ElementType: Int, Sendable, Codable {
    case unknown = 0
    case float32 = 1
    case float16 = 2
}

/// Identifies an extractor for cross-run compatibility checks.
/// Two extractors with the same `identity` produce comparable vectors.
public struct ExtractorIdentity: Hashable, Sendable, Codable {
    /// "vision" | "coreml"
    public let kind: String
    /// Human-readable name, e.g. "vision", "uni-v1", "virchow-v2".
    public let name: String
    /// Numeric revision for the same `(kind, name)` family.
    public let revision: Int

    public init(kind: String, name: String, revision: Int) {
        self.kind = kind
        self.name = name
        self.revision = revision
    }

    /// Compact "vision:vision:r2" or "coreml:uni-v1:r1" string, suitable for
    /// persistence in a feature bank.
    public var stringForm: String {
        "\(kind):\(name):r\(revision)"
    }
}

public enum ExtractError: Error, CustomStringConvertible {
    case noObservation
    case modelLoadFailed(String)
    case invalidInput(String)
    case unexpectedOutput(String)

    public var description: String {
        switch self {
        case .noObservation: return "Vision returned no feature observation."
        case .modelLoadFailed(let s): return "Could not load Core ML model: \(s)"
        case .invalidInput(let s): return "Invalid input image: \(s)"
        case .unexpectedOutput(let s): return "Unexpected model output: \(s)"
        }
    }
}
