import Foundation

/// Sidecar JSON describing how to drive a Core ML feature extractor. Saved
/// alongside the `.mlpackage` so the app knows input dimensions, the pixel
/// normalisation to apply, the expected output size, and a stable identity
/// for bank-compatibility checks.
public struct ExtractorDescriptor: Codable, Sendable {
    public let identity: ExtractorIdentity
    public let kind: Kind
    public let inputWidth: Int
    public let inputHeight: Int
    public let featureDim: Int
    public let pixelNormalization: PixelNormalization
    /// Required for `coreml`; resolved relative to the descriptor file's directory.
    public let modelFilename: String?

    public enum Kind: String, Codable, Sendable {
        case vision
        case coreml
    }

    public struct PixelNormalization: Codable, Sendable {
        /// Per-channel mean (typically in [0, 1] after scaling).
        public let meanRGB: [Float]
        /// Per-channel std (typically in [0, 1] after scaling).
        public let stdRGB: [Float]
        /// Factor applied to raw 0..255 pixel values before mean subtraction
        /// (1/255 by default — produces [0, 1]).
        public let scale: Float

        public init(meanRGB: [Float], stdRGB: [Float], scale: Float = 1.0 / 255.0) {
            self.meanRGB = meanRGB
            self.stdRGB = stdRGB
            self.scale = scale
        }

        /// Standard ImageNet normalisation. Used by UNI, Virchow, most ViTs.
        public static let imageNet = PixelNormalization(
            meanRGB: [0.485, 0.456, 0.406],
            stdRGB:  [0.229, 0.224, 0.225]
        )
    }

    public init(
        identity: ExtractorIdentity,
        kind: Kind,
        inputWidth: Int,
        inputHeight: Int,
        featureDim: Int,
        pixelNormalization: PixelNormalization,
        modelFilename: String?
    ) {
        self.identity = identity
        self.kind = kind
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.featureDim = featureDim
        self.pixelNormalization = pixelNormalization
        self.modelFilename = modelFilename
    }

    public static func load(from url: URL) throws -> ExtractorDescriptor {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ExtractorDescriptor.self, from: data)
    }

    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url, options: [.atomic])
    }
}
