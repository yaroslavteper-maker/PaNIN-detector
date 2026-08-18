import Foundation
import Observation
import SwiftUI

/// Holds the most recently computed t-SNE embedding plus live progress fields
/// so the config sheet, progress modal, and plot window can all read off the
/// same source of truth.
@Observable
final class EmbeddingStore: @unchecked Sendable {
    struct Embedding: Sendable {
        let points: [SIMD2<Float>]   // length N
        let labels: [String]         // length N
        let slideNames: [String]     // length N
        let extractorIdentity: ExtractorIdentity
        let perplexity: Double
        let iterations: Int
        let totalAvailable: Int      // points before subsampling
        let createdAt: Date

        var pointCount: Int { points.count }

        var perClass: [String: Int] {
            Dictionary(grouping: labels, by: { $0 }).mapValues { $0.count }
        }
    }

    var current: Embedding?
    /// Per-class centroids computed alongside the most recent embedding.
    /// Lives in the same store so the plot window can read both off one source.
    var centroids: CentroidSet?
    var isComputing: Bool = false
    var progress: Float = 0
    var status: String = ""
}
