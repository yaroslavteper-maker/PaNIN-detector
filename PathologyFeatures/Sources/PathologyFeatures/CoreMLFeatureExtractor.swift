import Foundation
import CoreML
import CoreGraphics

/// Runs an arbitrary `.mlpackage` whose output is a feature vector.
/// The accompanying `ExtractorDescriptor` JSON tells us the input dimensions,
/// normalisation, expected output dim, and identity — everything needed to
/// plug in a UNI / Virchow / CTransPath conversion without code changes.
public final class CoreMLFeatureExtractor: FeatureExtractor, @unchecked Sendable {
    public let identity: ExtractorIdentity
    public let inputSize: CGSize
    public let featureDim: Int

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let normalization: ExtractorDescriptor.PixelNormalization

    public init(modelURL: URL, descriptor: ExtractorDescriptor) throws {
        guard descriptor.kind == .coreml else {
            throw ExtractError.modelLoadFailed("Descriptor kind is not coreml")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        do {
            self.model = try MLModel(contentsOf: modelURL, configuration: cfg)
        } catch {
            throw ExtractError.modelLoadFailed(String(describing: error))
        }
        self.identity = descriptor.identity
        self.inputSize = CGSize(width: descriptor.inputWidth, height: descriptor.inputHeight)
        self.featureDim = descriptor.featureDim
        self.normalization = descriptor.pixelNormalization

        // Use the first input / output description by name.
        let inputs = model.modelDescription.inputDescriptionsByName
        guard let first = inputs.first else {
            throw ExtractError.modelLoadFailed("Model has no inputs")
        }
        self.inputName = first.key

        let outputs = model.modelDescription.outputDescriptionsByName
        guard let firstOutput = outputs.first else {
            throw ExtractError.modelLoadFailed("Model has no outputs")
        }
        self.outputName = firstOutput.key
    }

    public func extract(_ image: CGImage) async throws -> FeatureResult {
        let array = try makeInputArray(image)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: array)]
        )
        let prediction = try await model.prediction(from: provider)

        guard let featureValue = prediction.featureValue(for: outputName),
              let multiArray = featureValue.multiArrayValue else {
            throw ExtractError.unexpectedOutput("Output is not a multi-array")
        }
        // Flatten the multi-array into a Float32 Data blob.
        let count = multiArray.count
        var floats = [Float](repeating: 0, count: count)
        for i in 0..<count {
            floats[i] = Float(truncating: multiArray[i])
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return FeatureResult(data: data, dim: count, elementType: .float32)
    }

    // MARK: - Helpers

    /// Build a `[1, 3, H, W]` MLMultiArray with the descriptor's normalisation.
    private func makeInputArray(_ cgImage: CGImage) throws -> MLMultiArray {
        let H = Int(inputSize.height)
        let W = Int(inputSize.width)
        let shape: [NSNumber] = [1, 3, NSNumber(value: H), NSNumber(value: W)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        // Render the CGImage into a fixed-size RGBA8 buffer.
        let bytesPerPixel = 4
        let bytesPerRow = W * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: H * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ExtractError.invalidInput("Could not create RGBA context")
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: W, height: H))

        let mean = normalization.meanRGB
        let std = normalization.stdRGB
        let scale = normalization.scale
        guard mean.count >= 3, std.count >= 3 else {
            throw ExtractError.invalidInput("Normalisation vectors must have 3 components")
        }

        // Pack CHW directly into the MLMultiArray storage.
        let ptr = UnsafeMutableRawPointer(array.dataPointer)
            .assumingMemoryBound(to: Float.self)
        let planeSize = H * W
        for y in 0..<H {
            for x in 0..<W {
                let i = (y * W + x) * bytesPerPixel
                let r = (Float(pixels[i + 0]) * scale - mean[0]) / std[0]
                let g = (Float(pixels[i + 1]) * scale - mean[1]) / std[1]
                let b = (Float(pixels[i + 2]) * scale - mean[2]) / std[2]
                let idx = y * W + x
                ptr[0 * planeSize + idx] = r
                ptr[1 * planeSize + idx] = g
                ptr[2 * planeSize + idx] = b
            }
        }
        return array
    }
}
