import Foundation
import SwiftData

/// One labeled feature vector in the bank. Stored in Application Support via
/// SwiftData. Annotations themselves are NOT stored here — they live in the
/// per-slide GeoJSON sidecar. We denormalize `classification` so the sidebar
/// can group/count without joins.
@Model
final class MLPatch {
    @Attribute(.unique) var id: UUID
    var slidePath: String
    var slideName: String
    var annotationID: UUID
    var classification: String

    /// Top-left of the patch in level-0 slide-data-Y coordinates
    /// (i.e. after the same Y-mirror the exporter uses).
    var patchX: Int64
    var patchY: Int64
    var patchLevel: Int32
    /// Patch dimensions in *level* pixels (e.g. 224).
    var patchSizeLevel: Int

    /// Raw bytes of the feature print element array (float or float16).
    var featureData: Data
    var featureDim: Int
    /// Raw value of `VNElementType` (0 = unknown, 1 = float, 2 = float16, …).
    var featureElementType: Int
    /// `VNGenerateImageFeaturePrintRequest.revision` at extraction time.
    /// Kept for backward compat; new code prefers `extractorIdentity`.
    var extractorRevision: Int

    /// Stable identity of the extractor that produced `featureData`, e.g.
    /// `"vision:vision:r2"` or `"coreml:uni-v1:r1"`. Optional so older rows
    /// (Vision-only era) migrate cleanly — readers should treat `nil` as
    /// a legacy Vision identity at `extractorRevision`.
    var extractorIdentity: String? = nil

    var createdAt: Date

    /// Fraction (0…1) of pixels in the patch whose R, G, B are all above the
    /// near-white threshold (default 220). Legacy patches predating this field
    /// default to 0, so they're treated as "all tissue" by any threshold filter.
    var whiteFraction: Float = 0.0

    init(slidePath: String,
         slideName: String,
         annotationID: UUID,
         classification: String,
         patchX: Int64,
         patchY: Int64,
         patchLevel: Int32,
         patchSizeLevel: Int,
         featureData: Data,
         featureDim: Int,
         featureElementType: Int,
         extractorRevision: Int,
         extractorIdentity: String? = nil,
         whiteFraction: Float = 0.0) {
        self.id = UUID()
        self.slidePath = slidePath
        self.slideName = slideName
        self.annotationID = annotationID
        self.classification = classification
        self.patchX = patchX
        self.patchY = patchY
        self.patchLevel = patchLevel
        self.patchSizeLevel = patchSizeLevel
        self.featureData = featureData
        self.featureDim = featureDim
        self.featureElementType = featureElementType
        self.extractorRevision = extractorRevision
        self.extractorIdentity = extractorIdentity
        self.whiteFraction = whiteFraction
        self.createdAt = Date()
    }

    /// Resolved identity for this patch — uses the stored string when present,
    /// otherwise synthesizes a legacy Vision identity from `extractorRevision`.
    var resolvedExtractorIdentity: ExtractorIdentity {
        if let s = extractorIdentity, let id = ExtractorIdentity.parse(s) {
            return id
        }
        return .legacyVision(revision: extractorRevision)
    }
}
