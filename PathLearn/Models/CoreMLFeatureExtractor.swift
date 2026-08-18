import Foundation
import CoreML
import CoreGraphics

/// Runs an arbitrary `.mlpackage` whose output is a feature vector. The
/// accompanying `ExtractorDescriptor` JSON tells us the input dimensions,
/// normalisation, expected output dim, and identity — everything needed to
/// plug in a UNI / Virchow / CTransPath conversion without code changes.
final class CoreMLFeatureExtractor: AnyFeatureExtractor, @unchecked Sendable {
    let identity: ExtractorIdentity
    let inputSize: CGSize
    let featureDim: Int

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let normalization: ExtractorDescriptor.PixelNormalization

    enum LoadError: Error, CustomStringConvertible {
        case wrongKind
        case loadFailed(String)
        case noInputs
        case noOutputs
        var description: String {
            switch self {
            case .wrongKind: return "Descriptor kind is not coreml."
            case .loadFailed(let s): return "Could not load Core ML model: \(s)"
            case .noInputs: return "Core ML model has no declared inputs."
            case .noOutputs: return "Core ML model has no declared outputs."
            }
        }
    }

    enum ExtractError: Error, CustomStringConvertible {
        case invalidInput(String)
        case unexpectedOutput(String)
        var description: String {
            switch self {
            case .invalidInput(let s): return "Invalid input image for Core ML extractor: \(s)"
            case .unexpectedOutput(let s): return "Unexpected Core ML output: \(s)"
            }
        }
    }

    init(modelURL: URL, descriptor: ExtractorDescriptor) throws {
        guard descriptor.kind == .coreml else { throw LoadError.wrongKind }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        // Core ML's runtime loader needs a compiled `.mlmodelc`, not the
        // `.mlpackage` produced by coremltools. Compile once and cache the
        // result alongside the source so the next launch skips the recompile.
        let compiledURL: URL
        do {
            compiledURL = try Self.compiledModel(for: modelURL, identity: descriptor.identity)
        } catch {
            throw LoadError.loadFailed("Compile step failed: \(error)")
        }
        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: cfg)
        } catch {
            throw LoadError.loadFailed(String(describing: error))
        }
        self.identity = descriptor.identity
        self.inputSize = CGSize(width: descriptor.inputWidth, height: descriptor.inputHeight)
        self.featureDim = descriptor.featureDim
        self.normalization = descriptor.pixelNormalization

        let inputs = model.modelDescription.inputDescriptionsByName
        guard let first = inputs.first else { throw LoadError.noInputs }
        self.inputName = first.key

        let outputs = model.modelDescription.outputDescriptionsByName
        guard let firstOutput = outputs.first else { throw LoadError.noOutputs }
        self.outputName = firstOutput.key
    }

    func extract(_ image: CGImage) async throws -> FeatureExtractionResult {
        let array = try makeInputArray(image)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: array)]
        )
        let prediction = try await model.prediction(from: provider)

        guard let featureValue = prediction.featureValue(for: outputName),
              let multiArray = featureValue.multiArrayValue else {
            throw ExtractError.unexpectedOutput("Output is not a multi-array")
        }
        let count = multiArray.count
        var floats = [Float](repeating: 0, count: count)
        for i in 0..<count {
            floats[i] = Float(truncating: multiArray[i])
        }
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return FeatureExtractionResult(
            data: data,
            dim: count,
            elementType: 1, // float32
            identity: identity
        )
    }

    // MARK: - Compilation cache

    /// Returns a URL to the compiled `.mlmodelc` for `modelURL`. If the source
    /// is already a `.mlmodelc`, returns it as-is. Otherwise compiles into the
    /// app's cache directory and returns the cached path. Recompiles if the
    /// source `.mlpackage` is newer than the cached output.
    private static func compiledModel(for modelURL: URL,
                                      identity: ExtractorIdentity) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" {
            return modelURL
        }
        let fm = FileManager.default
        let cacheDir = try cacheDirectory(for: identity)
        let cachedName = modelURL.deletingPathExtension().lastPathComponent + ".mlmodelc"
        let cachedURL = cacheDir.appendingPathComponent(cachedName, isDirectory: true)

        if fm.fileExists(atPath: cachedURL.path),
           let cachedAttrs = try? fm.attributesOfItem(atPath: cachedURL.path),
           let srcAttrs = try? fm.attributesOfItem(atPath: modelURL.path),
           let cachedDate = cachedAttrs[.modificationDate] as? Date,
           let srcDate = srcAttrs[.modificationDate] as? Date,
           cachedDate >= srcDate {
            return cachedURL
        }

        print("[coreml] compiling \(modelURL.lastPathComponent) → \(cachedURL.path)")
        let freshURL = try MLModel.compileModel(at: modelURL)
        if fm.fileExists(atPath: cachedURL.path) {
            try? fm.removeItem(at: cachedURL)
        }
        try fm.moveItem(at: freshURL, to: cachedURL)
        return cachedURL
    }

    private static func cacheDirectory(for identity: ExtractorIdentity) throws -> URL {
        let fm = FileManager.default
        let caches = try fm.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let bundleID = Bundle.main.bundleIdentifier ?? "PaNIN-detector.PaNIN-detector"
        let dir = caches
            .appendingPathComponent(bundleID, isDirectory: true)
            .appendingPathComponent("CompiledExtractors", isDirectory: true)
            .appendingPathComponent(identity.stringForm, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Helpers

    /// Build a `[1, 3, H, W]` MLMultiArray with the descriptor's normalisation.
    private func makeInputArray(_ cgImage: CGImage) throws -> MLMultiArray {
        let H = Int(inputSize.height)
        let W = Int(inputSize.width)
        let shape: [NSNumber] = [1, 3, NSNumber(value: H), NSNumber(value: W)]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

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
