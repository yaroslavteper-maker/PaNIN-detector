import Foundation

/// Sidecar JSON describing how to drive a Core ML feature extractor. Saved
/// alongside the `.mlpackage` so the app knows input dimensions, pixel
/// normalisation, output size, and a stable identity for bank-compatibility
/// checks. Mirrors the type in the standalone `PathologyFeatures` package.
struct ExtractorDescriptor: Codable, Sendable {
    let identity: ExtractorIdentity
    let kind: Kind
    let inputWidth: Int
    let inputHeight: Int
    let featureDim: Int
    let pixelNormalization: PixelNormalization
    /// Required for `coreml`; resolved relative to the descriptor's directory.
    let modelFilename: String?

    enum Kind: String, Codable, Sendable {
        case vision
        case coreml
    }

    struct PixelNormalization: Codable, Sendable {
        /// Per-channel mean (typically in [0, 1] after scaling).
        let meanRGB: [Float]
        /// Per-channel std (typically in [0, 1] after scaling).
        let stdRGB: [Float]
        /// Factor applied to raw 0…255 pixel values before mean subtraction
        /// (1/255 by default → produces [0, 1]).
        let scale: Float

        init(meanRGB: [Float], stdRGB: [Float], scale: Float = 1.0 / 255.0) {
            self.meanRGB = meanRGB
            self.stdRGB = stdRGB
            self.scale = scale
        }

        /// Standard ImageNet normalisation. Used by UNI, Virchow, most ViTs.
        static let imageNet = PixelNormalization(
            meanRGB: [0.485, 0.456, 0.406],
            stdRGB:  [0.229, 0.224, 0.225]
        )
    }

    static func load(from url: URL) throws -> ExtractorDescriptor {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ExtractorDescriptor.self, from: data)
    }

    func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url, options: [.atomic])
    }
}
