import Foundation
import Observation
import Vision

/// In-process registry of available feature extractors. The built-in Vision
/// extractor is always present; Core ML extractors are discovered by scanning
/// `~/Library/Application Support/<bundle>/Extractors/` for
/// `*.paninextractor.json` sidecars at launch.
///
/// Lookups in the pipelines resolve a requested `ExtractorIdentity` to a
/// concrete extractor instance — falling back to Vision when no match is
/// found (legacy banks/models have no stored identity).
@Observable
final class ExtractorRegistry: @unchecked Sendable {
    static let shared = ExtractorRegistry()

    struct DiscoveredExtractor: Identifiable, Sendable {
        let id: String       // identity.stringForm
        let identity: ExtractorIdentity
        let descriptor: ExtractorDescriptor
        let descriptorURL: URL
        let modelURL: URL?   // nil for vision-kind descriptors
    }

    /// All known extractors — the always-present built-in Vision entry first,
    /// then any user-installed Core ML entries discovered on disk.
    private(set) var available: [DiscoveredExtractor] = []

    /// In-memory cache of instantiated extractors keyed by identity string.
    /// CoreML model loads are expensive — reuse across pipeline runs.
    private var cache: [String: any AnyFeatureExtractor] = [:]
    private let cacheLock = NSLock()

    private init() {
        // Always seed with the built-in Vision entry so the UI can list it.
        let visionRev = VNGenerateImageFeaturePrintRequest.currentRevision
        let visionID = ExtractorIdentity.legacyVision(revision: visionRev)
        let visionDescriptor = ExtractorDescriptor(
            identity: visionID,
            kind: .vision,
            inputWidth: 299, inputHeight: 299,
            featureDim: 2048,
            pixelNormalization: .imageNet,
            modelFilename: nil
        )
        available = [
            DiscoveredExtractor(
                id: visionID.stringForm,
                identity: visionID,
                descriptor: visionDescriptor,
                descriptorURL: URL(fileURLWithPath: "/dev/null"),
                modelURL: nil
            )
        ]
    }

    /// Filesystem location where user-installed extractors live. Created on
    /// first access so it shows up in Finder even before any extractor has
    /// been added.
    static var extractorsDirectory: URL {
        let fm = FileManager.default
        let support = (try? fm.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )) ?? fm.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        let bundleID = Bundle.main.bundleIdentifier ?? "PaNIN-detector.PaNIN-detector"
        let dir = support
            .appendingPathComponent(bundleID, isDirectory: true)
            .appendingPathComponent("Extractors", isDirectory: true)
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Re-scan the extractors directory. Safe to call multiple times.
    @discardableResult
    func discover() -> [DiscoveredExtractor] {
        let visionEntry = available.first { $0.descriptor.kind == .vision }
        var found: [DiscoveredExtractor] = []
        let fm = FileManager.default
        let dir = Self.extractorsDirectory
        guard let enumerator = fm.enumerator(
            at: dir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            available = visionEntry.map { [$0] } ?? []
            return available
        }
        for case let url as URL in enumerator {
            guard url.pathExtension == "json",
                  url.lastPathComponent.hasSuffix(".paninextractor.json") else { continue }
            do {
                let descriptor = try ExtractorDescriptor.load(from: url)
                let modelURL: URL? = descriptor.modelFilename.flatMap { name in
                    url.deletingLastPathComponent().appendingPathComponent(name)
                }
                found.append(DiscoveredExtractor(
                    id: descriptor.identity.stringForm,
                    identity: descriptor.identity,
                    descriptor: descriptor,
                    descriptorURL: url,
                    modelURL: modelURL
                ))
            } catch {
                print("[extractors] skipped \(url.lastPathComponent): \(error)")
            }
        }
        var all: [DiscoveredExtractor] = []
        if let visionEntry { all.append(visionEntry) }
        all.append(contentsOf: found.sorted { $0.id < $1.id })
        available = all
        print("[extractors] discovered \(found.count) Core ML extractor(s) in \(dir.path)")
        return available
    }

    /// Resolve an identity to a concrete extractor. Vision identities always
    /// resolve to a fresh `FeatureExtractor` at the requested revision.
    /// CoreML identities require a matching discovered descriptor.
    enum ResolveError: Error, CustomStringConvertible {
        case notFound(String)
        case loadFailed(String, String)
        var description: String {
            switch self {
            case .notFound(let id): return "No installed extractor matches identity \(id)."
            case .loadFailed(let id, let msg): return "Failed to load extractor \(id): \(msg)"
            }
        }
    }

    func extractor(for identity: ExtractorIdentity) throws -> any AnyFeatureExtractor {
        let key = identity.stringForm
        cacheLock.lock()
        if let cached = cache[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()

        let made: any AnyFeatureExtractor
        switch identity.kind {
        case "vision":
            made = FeatureExtractor(revision: identity.revision)
        case "coreml":
            guard let entry = available.first(where: { $0.id == key }) else {
                throw ResolveError.notFound(key)
            }
            guard let modelURL = entry.modelURL else {
                throw ResolveError.notFound(key)
            }
            do {
                made = try CoreMLFeatureExtractor(modelURL: modelURL, descriptor: entry.descriptor)
            } catch {
                throw ResolveError.loadFailed(key, String(describing: error))
            }
        default:
            throw ResolveError.notFound(key)
        }
        cacheLock.lock()
        cache[key] = made
        cacheLock.unlock()
        return made
    }

    /// Forget the loaded instance for an identity. Handy when the descriptor
    /// or `.mlpackage` is replaced on disk.
    func invalidate(_ identity: ExtractorIdentity) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        cache.removeValue(forKey: identity.stringForm)
    }
}
