import Foundation
import SwiftData

/// JSON encode/decode for the patch bank. Each saved file is a self-contained
/// snapshot of all `MLPatch` rows. Feature vectors travel as base64 blobs so
/// the file is a single plain-text artifact.
enum BankSerializer {
    static let currentVersion = 1
    /// File extension used when saving a bank with the new naming (`.bank`).
    /// Legacy `*.paninbank.json` files still load via the open panel since
    /// the underlying JSON format is unchanged.
    static let suggestedExtension = "bank"

    struct File: Codable {
        let version: Int
        let createdAt: Date
        let patchCount: Int
        let patches: [PatchDTO]
    }

    struct PatchDTO: Codable {
        let id: String
        let slidePath: String
        let slideName: String
        let annotationID: String
        let classification: String
        let patchX: Int64
        let patchY: Int64
        let patchLevel: Int32
        let patchSizeLevel: Int
        let featureBase64: String
        let featureDim: Int
        let featureElementType: Int
        let extractorRevision: Int
        let createdAt: Date
        /// Optional for backward compat with older files (legacy = 0.0).
        let whiteFraction: Float?
        /// Optional for backward compat — older banks predate multi-extractor
        /// support. Readers treat `nil` as legacy Vision at `extractorRevision`.
        let extractorIdentity: String?
    }

    enum SerializerError: Error, CustomStringConvertible {
        case decodeFailed(String)
        case base64Failed
        case versionMismatch(Int)
        var description: String {
            switch self {
            case .decodeFailed(let s): return "Could not parse bank file: \(s)"
            case .base64Failed:        return "Feature vector base64 was malformed."
            case .versionMismatch(let v): return "Bank file version \(v) is not supported."
            }
        }
    }

    @MainActor
    static func export(context: ModelContext, to url: URL) throws -> Int {
        let descriptor = FetchDescriptor<MLPatch>()
        let rows = try context.fetch(descriptor)
        let file = File(
            version: currentVersion,
            createdAt: Date(),
            patchCount: rows.count,
            patches: rows.map { p in
                PatchDTO(
                    id: p.id.uuidString,
                    slidePath: p.slidePath,
                    slideName: p.slideName,
                    annotationID: p.annotationID.uuidString,
                    classification: p.classification,
                    patchX: p.patchX,
                    patchY: p.patchY,
                    patchLevel: p.patchLevel,
                    patchSizeLevel: p.patchSizeLevel,
                    featureBase64: p.featureData.base64EncodedString(),
                    featureDim: p.featureDim,
                    featureElementType: p.featureElementType,
                    extractorRevision: p.extractorRevision,
                    createdAt: p.createdAt,
                    whiteFraction: p.whiteFraction,
                    extractorIdentity: p.extractorIdentity
                )
            }
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(file)
        try data.write(to: url, options: [.atomic])
        return rows.count
    }

    enum ImportMode: Sendable {
        case replace
        case append
    }

    @MainActor
    @discardableResult
    static func importFile(_ url: URL, into context: ModelContext, mode: ImportMode) throws -> Int {
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

        if mode == .replace {
            try context.delete(model: MLPatch.self)
        }

        var imported = 0
        for dto in file.patches {
            guard let featureData = Data(base64Encoded: dto.featureBase64) else {
                continue
            }
            let row = MLPatch(
                slidePath: dto.slidePath,
                slideName: dto.slideName,
                annotationID: UUID(uuidString: dto.annotationID) ?? UUID(),
                classification: dto.classification,
                patchX: dto.patchX,
                patchY: dto.patchY,
                patchLevel: dto.patchLevel,
                patchSizeLevel: dto.patchSizeLevel,
                featureData: featureData,
                featureDim: dto.featureDim,
                featureElementType: dto.featureElementType,
                extractorRevision: dto.extractorRevision,
                extractorIdentity: dto.extractorIdentity,
                whiteFraction: dto.whiteFraction ?? 0.0
            )
            context.insert(row)
            imported += 1
        }
        try context.save()
        return imported
    }
}
