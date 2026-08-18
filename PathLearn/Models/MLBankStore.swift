import Foundation
import Observation
import SwiftData

/// Aggregate stats over the persistent `MLPatch` bank. Refreshed lazily after
/// extractions or when the Bank panel appears.
@Observable
final class MLBankStore {
    var totalPatches: Int = 0
    var perClass: [String: Int] = [:]
    var perSlide: [String: Int] = [:]
    /// Counts of patches grouped by their resolved extractor identity
    /// (e.g. `"vision:vision:r2"` or `"coreml:uni-v1:r1"`).
    var perExtractor: [String: Int] = [:]
    /// Nested counts: extractor identity → class → patch count. Used by the
    /// Train sheet to drive per-extractor class summaries when the bank holds
    /// patches from more than one extractor.
    var perExtractorPerClass: [String: [String: Int]] = [:]
    var lastRefreshed: Date?

    func refresh(context: ModelContext) {
        do {
            let descriptor = FetchDescriptor<MLPatch>()
            let all = try context.fetch(descriptor)
            totalPatches = all.count
            perClass = Dictionary(grouping: all, by: \.classification)
                .mapValues { $0.count }
            perSlide = Dictionary(grouping: all, by: \.slideName)
                .mapValues { $0.count }
            let byExtractor = Dictionary(grouping: all, by: { $0.resolvedExtractorIdentity.stringForm })
            perExtractor = byExtractor.mapValues { $0.count }
            perExtractorPerClass = byExtractor.mapValues { rows in
                Dictionary(grouping: rows, by: \.classification).mapValues { $0.count }
            }
            lastRefreshed = Date()
        } catch {
            print("[MLBankStore] refresh failed: \(error)")
        }
    }

    func clearAll(context: ModelContext) {
        do {
            try context.delete(model: MLPatch.self)
            try context.save()
            refresh(context: context)
        } catch {
            print("[MLBankStore] clearAll failed: \(error)")
        }
    }

    func removePatches(forClass name: String, context: ModelContext) {
        do {
            let descriptor = FetchDescriptor<MLPatch>(
                predicate: #Predicate<MLPatch> { $0.classification == name }
            )
            let rows = try context.fetch(descriptor)
            for row in rows { context.delete(row) }
            try context.save()
            refresh(context: context)
        } catch {
            print("[MLBankStore] removePatches(forClass:) failed: \(error)")
        }
    }

    func removePatches(forSlide slideName: String, context: ModelContext) {
        do {
            let descriptor = FetchDescriptor<MLPatch>(
                predicate: #Predicate<MLPatch> { $0.slideName == slideName }
            )
            let rows = try context.fetch(descriptor)
            for row in rows { context.delete(row) }
            try context.save()
            refresh(context: context)
        } catch {
            print("[MLBankStore] removePatches(forSlide:) failed: \(error)")
        }
    }
}
