import Foundation
import CoreML

/// Runs the three VISTA / MicePan UNets (converted to Core ML) over a single
/// normalized 512×512 tile and reduces their per-pixel probability maps to a
/// tissue-type composition, matching the paper's thresholds and combine order.
///
/// The `.mlpackage` files are loaded at runtime from Application Support (the app
/// is not sandboxed), compiled to `.mlmodelc` once and cached — the same pattern
/// as `CoreMLFeatureExtractor`. This keeps the models out of the app bundle /
/// pbxproj entirely.
nonisolated final class VISTASegmenter: @unchecked Sendable {

    /// Tissue classes in combine-priority order (normal wins over metaplasia
    /// wins over neoplasia — as in ProcessImages.py), each with its published
    /// probability threshold and display color.
    struct TissueClass {
        let name: String
        let modelFile: String
        let threshold: Float
        let color: AnnotationColor
    }

    static let tileSize = 512

    static let neoplasia  = TissueClass(name: "Neoplasia",  modelFile: "VISTA-neoplasia.mlpackage",  threshold: 0.7, color: AnnotationColor(r: 230, g: 210, b: 30))
    static let metaplasia = TissueClass(name: "Metaplasia", modelFile: "VISTA-metaplasia.mlpackage", threshold: 0.5, color: AnnotationColor(r: 222, g: 31,  b: 123))
    static let normal     = TissueClass(name: "Normal",     modelFile: "VISTA-normal.mlpackage",     threshold: 0.3, color: AnnotationColor(r: 122, g: 230, b: 213))
    static let stroma     = AnnotationColor(r: 19, g: 16, b: 163)

    /// All labels the segmenter can emit, with their heatmap colors.
    static var classColors: [String: AnnotationColor] {
        [neoplasia.name: neoplasia.color,
         metaplasia.name: metaplasia.color,
         normal.name: normal.color,
         "Stroma": stroma]
    }

    enum LoadError: Error, CustomStringConvertible {
        case modelsNotInstalled(URL)
        case compileFailed(String)
        case loadFailed(String)
        var description: String {
            switch self {
            case .modelsNotInstalled(let dir):
                return "VISTA models not found. Put VISTA-neoplasia/metaplasia/normal.mlpackage in:\n\(dir.path)"
            case .compileFailed(let s): return "Failed to compile a VISTA model: \(s)"
            case .loadFailed(let s):    return "Failed to load a VISTA model: \(s)"
            }
        }
    }

    // model + its io names + threshold, in combine order [normal, metaplasia, neoplasia].
    private struct Loaded {
        let cls: TissueClass
        let model: MLModel
        let inputName: String
        let outputName: String
    }
    private let models: [Loaded]

    /// The directory the models are loaded from: `<AppSupport>/<bundle>/VISTA/`.
    static var modelsDirectory: URL {
        let base = (try? FileManager.default.url(for: .applicationSupportDirectory,
                                                 in: .userDomainMask,
                                                 appropriateFor: nil, create: true))
            ?? FileManager.default.homeDirectoryForCurrentUser
        let bundleID = Bundle.main.bundleIdentifier ?? "PaNIN-detector.PaNIN-detector"
        return base.appendingPathComponent(bundleID, isDirectory: true)
            .appendingPathComponent("VISTA", isDirectory: true)
    }

    static var isInstalled: Bool {
        let dir = modelsDirectory
        for c in [neoplasia, metaplasia, normal] {
            if !FileManager.default.fileExists(atPath: dir.appendingPathComponent(c.modelFile).path) {
                return false
            }
        }
        return true
    }

    init() throws {
        let dir = Self.modelsDirectory
        guard Self.isInstalled else { throw LoadError.modelsNotInstalled(dir) }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        // Combine order: normal first so it wins ties, then metaplasia, then neoplasia.
        var loaded: [Loaded] = []
        for cls in [Self.normal, Self.metaplasia, Self.neoplasia] {
            let src = dir.appendingPathComponent(cls.modelFile)
            let compiled: URL
            do { compiled = try Self.compiledModel(for: src) }
            catch { throw LoadError.compileFailed("\(cls.modelFile): \(error)") }
            let model: MLModel
            do { model = try MLModel(contentsOf: compiled, configuration: cfg) }
            catch { throw LoadError.loadFailed("\(cls.modelFile): \(error)") }
            let inName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
            let outName = model.modelDescription.outputDescriptionsByName.keys.first ?? "mask"
            loaded.append(Loaded(cls: cls, model: model, inputName: inName, outputName: outName))
        }
        self.models = loaded
    }

    // MARK: - Inference

    struct TileResult {
        let dominant: String
        let maxFraction: Float
        /// Fraction of the tile's tissue (non-white) area per class.
        let fractions: [String: Float]
    }

    /// Segment one normalized RGBA8 tile (512×512). Returns nil if the tile is
    /// essentially all background (no tissue).
    func segment(tileRGBA: [UInt8]) throws -> TileResult? {
        let side = Self.tileSize
        let n = side * side
        guard tileRGBA.count >= n * 4 else { return nil }

        guard let tensor = Self.inputTensor(fromRGBA: tileRGBA, side: side) else { return nil }
        let provider = { (inName: String) throws -> MLFeatureProvider in
            try MLDictionaryFeatureProvider(dictionary: [inName: MLFeatureValue(multiArray: tensor)])
        }

        // Run the three models; threshold each into a boolean map.
        var maps: [String: [Bool]] = [:]
        for l in models {
            let out = try l.model.prediction(from: try provider(l.inputName))
            guard let arr = out.featureValue(for: l.outputName)?.multiArrayValue else { continue }
            maps[l.cls.name] = Self.threshold(arr, above: l.cls.threshold, count: n)
        }
        guard let neoM = maps[Self.neoplasia.name],
              let metaM = maps[Self.metaplasia.name],
              let normM = maps[Self.normal.name] else { return nil }

        // Whitespace from the normalized tile (all channels >= 200), matching the
        // Python combine step.
        var counts = [Self.neoplasia.name: 0, Self.metaplasia.name: 0,
                      Self.normal.name: 0, "Stroma": 0]
        var tissuePixels = 0
        for i in 0..<n {
            let r = tileRGBA[i * 4 + 0], g = tileRGBA[i * 4 + 1], b = tileRGBA[i * 4 + 2]
            if r >= 200 && g >= 200 && b >= 200 { continue }  // white → skip
            tissuePixels += 1
            // priority: normal > metaplasia > neoplasia; else stroma.
            if normM[i] { counts[Self.normal.name]! += 1 }
            else if metaM[i] { counts[Self.metaplasia.name]! += 1 }
            else if neoM[i] { counts[Self.neoplasia.name]! += 1 }
            else { counts["Stroma"]! += 1 }
        }
        guard tissuePixels > 0 else { return nil }

        var fractions: [String: Float] = [:]
        for (k, v) in counts { fractions[k] = Float(v) / Float(tissuePixels) }
        let dominantPair = fractions.max { $0.value < $1.value }!
        return TileResult(dominant: dominantPair.key,
                          maxFraction: dominantPair.value,
                          fractions: fractions)
    }

    /// Threshold the model's `[1, H, W, 1]` probability map into a row-major
    /// (y*W + x) boolean mask. Must honor the array's `strides`: Core ML / ANE
    /// returns a padded, non-contiguous Float16 buffer (e.g. strides
    /// [.., 16384, 32, 1] for a 512-wide map), so a linear read grabs padding
    /// and yields garbage.
    private static func threshold(_ arr: MLMultiArray, above t: Float, count: Int) -> [Bool] {
        // shape [1, H, W, 1]; strides give the element offsets per axis.
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        guard shape.count == 4 else { return [Bool](repeating: false, count: count) }
        let H = shape[1], W = shape[2]
        let sY = strides[1], sX = strides[2]
        var out = [Bool](repeating: false, count: H * W)

        func fill<T>(_ ptr: UnsafePointer<T>, _ toFloat: (T) -> Float) {
            for y in 0..<H {
                let rowBase = y * sY
                let outRow = y * W
                for x in 0..<W {
                    out[outRow + x] = toFloat(ptr[rowBase + x * sX]) >= t
                }
            }
        }
        switch arr.dataType {
        case .float16:
            arr.dataPointer.withMemoryRebound(to: Float16.self, capacity: arr.count) {
                fill($0) { Float($0) }
            }
        case .float32:
            arr.dataPointer.withMemoryRebound(to: Float32.self, capacity: arr.count) {
                fill($0) { $0 }
            }
        case .double:
            arr.dataPointer.withMemoryRebound(to: Double.self, capacity: arr.count) {
                fill($0) { Float($0) }
            }
        default:
            for y in 0..<H { for x in 0..<W {
                out[y * W + x] = arr[[0, y, x, 0] as [NSNumber]].floatValue >= t
            }}
        }
        return out
    }

    // MARK: - Input tensor

    /// Build an NHWC `[1, side, side, 3]` Float32 MLMultiArray of raw 0-255 RGB
    /// values from a tightly-packed RGBA8 tile. The model divides by 255
    /// internally. A plain tensor (not an image input) avoids Core ML's image
    /// color management, which otherwise gamma-shifted the pixels.
    private static func inputTensor(fromRGBA rgba: [UInt8], side: Int) -> MLMultiArray? {
        guard let arr = try? MLMultiArray(shape: [1, NSNumber(value: side), NSNumber(value: side), 3],
                                          dataType: .float32) else { return nil }
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float32.self)
        let n = side * side
        for i in 0..<n {
            let s = i * 4
            let d = i * 3
            ptr[d + 0] = Float32(rgba[s + 0]) // R
            ptr[d + 1] = Float32(rgba[s + 1]) // G
            ptr[d + 2] = Float32(rgba[s + 2]) // B
        }
        return arr
    }

    // MARK: - Compile cache (mirrors CoreMLFeatureExtractor)

    private static func compiledModel(for modelURL: URL) throws -> URL {
        if modelURL.pathExtension == "mlmodelc" { return modelURL }
        let fm = FileManager.default
        let caches = try fm.url(for: .cachesDirectory, in: .userDomainMask,
                                appropriateFor: nil, create: true)
        let bundleID = Bundle.main.bundleIdentifier ?? "PaNIN-detector.PaNIN-detector"
        let cacheDir = caches.appendingPathComponent(bundleID, isDirectory: true)
            .appendingPathComponent("CompiledVISTA", isDirectory: true)
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let cachedURL = cacheDir.appendingPathComponent(
            modelURL.deletingPathExtension().lastPathComponent + ".mlmodelc", isDirectory: true)

        if fm.fileExists(atPath: cachedURL.path),
           let ca = try? fm.attributesOfItem(atPath: cachedURL.path),
           let sa = try? fm.attributesOfItem(atPath: modelURL.path),
           let cd = ca[.modificationDate] as? Date,
           let sd = sa[.modificationDate] as? Date, cd >= sd {
            return cachedURL
        }
        let fresh = try MLModel.compileModel(at: modelURL)
        if fm.fileExists(atPath: cachedURL.path) { try? fm.removeItem(at: cachedURL) }
        try fm.moveItem(at: fresh, to: cachedURL)
        return cachedURL
    }
}
