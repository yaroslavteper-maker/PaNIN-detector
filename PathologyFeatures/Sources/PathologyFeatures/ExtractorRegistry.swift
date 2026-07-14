import Foundation

/// Convenience factory functions for getting a configured `FeatureExtractor`.
public enum ExtractorRegistry {

    /// Built-in `VNGenerateImageFeaturePrintRequest` — no model file required.
    public static func builtInVision(revision: Int? = nil) -> any FeatureExtractor {
        VisionFeatureExtractor(revision: revision)
    }

    /// Load an extractor from its sidecar JSON. For `coreml`, the model file
    /// is resolved relative to the descriptor's directory.
    public static func load(descriptor url: URL) throws -> any FeatureExtractor {
        let descriptor = try ExtractorDescriptor.load(from: url)
        switch descriptor.kind {
        case .vision:
            return VisionFeatureExtractor(revision: descriptor.identity.revision)
        case .coreml:
            guard let modelFilename = descriptor.modelFilename else {
                throw ExtractError.modelLoadFailed(
                    "Descriptor.modelFilename is missing for kind=coreml"
                )
            }
            let modelURL = url
                .deletingLastPathComponent()
                .appendingPathComponent(modelFilename)
            return try CoreMLFeatureExtractor(modelURL: modelURL, descriptor: descriptor)
        }
    }
}
