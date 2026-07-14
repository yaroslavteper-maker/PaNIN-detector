import Foundation

/// Saves / loads `MLClassifier` as a `.paninmodel.json` file. Weights and
/// biases ride as base64 Float32 blobs so a model file is one self-contained
/// JSON artifact (no sidecar binaries).
enum ModelSerializer {
    static let currentVersion = 1
    /// File extension used when saving a classifier with the new naming
    /// (`.cl`). Legacy `*.paninmodel.json` files still load via the open
    /// panel since the underlying format is unchanged.
    static let suggestedExtension = "cl"

    struct File: Codable {
        let version: Int
        let id: String
        let createdAt: Date
        let classLabels: [String]
        let featureDim: Int
        let classCount: Int
        let weightsBase64: String
        let biasesBase64: String
        let featureExtractorRevision: Int
        let metrics: TrainingMetrics
        /// Optional for backward compat — models written before multi-extractor
        /// support omit this. Readers treat `nil` as legacy Vision at
        /// `featureExtractorRevision`.
        let extractorIdentity: String?
        /// Provenance: `"logistic"` or `"centroid"`. Optional for backward
        /// compat — `nil` is treated as `.logistic` on load.
        let kind: String?
        /// Embedded t-SNE plot data, present only for centroid classifiers
        /// that were promoted from a live t-SNE run. Loading restores this
        /// into the embedding store so the plot window reopens identically.
        let embeddingSnapshot: EmbeddingSnapshot?
    }

    enum SerializerError: Error, CustomStringConvertible {
        case base64Failed
        case versionMismatch(Int)
        case decodeFailed(String)
        var description: String {
            switch self {
            case .base64Failed: return "Model file weight/bias blob was malformed."
            case .versionMismatch(let v): return "Model file version \(v) is not supported."
            case .decodeFailed(let s): return "Could not parse model file: \(s)"
            }
        }
    }

    static func save(_ classifier: MLClassifier, to url: URL) throws {
        let file = File(
            version: currentVersion,
            id: classifier.id.uuidString,
            createdAt: classifier.createdAt,
            classLabels: classifier.classLabels,
            featureDim: classifier.featureDim,
            classCount: classifier.classCount,
            weightsBase64: floatsToData(classifier.weights).base64EncodedString(),
            biasesBase64: floatsToData(classifier.biases).base64EncodedString(),
            featureExtractorRevision: classifier.featureExtractorRevision,
            metrics: classifier.metrics,
            extractorIdentity: classifier.extractorIdentity,
            kind: classifier.kind,
            embeddingSnapshot: classifier.embeddingSnapshot
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(file)
        try data.write(to: url, options: [.atomic])
    }

    static func load(from url: URL) throws -> MLClassifier {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let file: File
        do {
            file = try decoder.decode(File.self, from: data)
        } catch {
            throw SerializerError.decodeFailed(String(describing: error))
        }
        guard file.version == currentVersion else {
            throw SerializerError.versionMismatch(file.version)
        }
        guard let wData = Data(base64Encoded: file.weightsBase64),
              let bData = Data(base64Encoded: file.biasesBase64) else {
            throw SerializerError.base64Failed
        }
        let id = UUID(uuidString: file.id) ?? UUID()
        return MLClassifier(
            id: id,
            weights: dataToFloats(wData),
            biases: dataToFloats(bData),
            classLabels: file.classLabels,
            featureDim: file.featureDim,
            metrics: file.metrics,
            featureExtractorRevision: file.featureExtractorRevision,
            extractorIdentity: file.extractorIdentity,
            kind: file.kind,
            embeddingSnapshot: file.embeddingSnapshot,
            createdAt: file.createdAt
        )
    }

    private static func floatsToData(_ floats: [Float]) -> Data {
        floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private static func dataToFloats(_ data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return out
    }
}
