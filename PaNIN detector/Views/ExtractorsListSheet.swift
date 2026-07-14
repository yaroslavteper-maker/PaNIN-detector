import SwiftUI
import AppKit

/// Small panel that lists every feature extractor the app knows about.
/// Built-in Vision always appears first; Core ML extractors discovered in
/// `~/Library/Application Support/<bundle>/Extractors/` follow.
struct ExtractorsListSheet: View {
    let onClose: () -> Void

    private let registry = ExtractorRegistry.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("Feature extractors", systemImage: "cpu")
                    .font(.headline)
                Spacer()
                Button {
                    registry.discover()
                } label: {
                    Label("Rescan", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([ExtractorRegistry.extractorsDirectory])
                } label: {
                    Label("Reveal Folder", systemImage: "folder")
                }
                .buttonStyle(.bordered)
            }

            Text("Drop `.paninextractor.json` sidecars next to their `.mlpackage` files into the Extractors folder, then click Rescan. The Python helper in `PathologyFeatures/Tools/convert_to_coreml.py` produces both files.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Divider()

            if registry.available.isEmpty {
                Text("No extractors found.")
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 30)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(registry.available) { entry in
                            row(entry)
                        }
                    }
                }
            }

            HStack {
                Spacer()
                Button("Close") { onClose() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(minWidth: 540, minHeight: 360)
    }

    @ViewBuilder
    private func row(_ entry: ExtractorRegistry.DiscoveredExtractor) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Image(systemName: entry.descriptor.kind == .vision ? "wand.and.stars" : "cube.box")
                    .foregroundStyle(.secondary)
                Text(entry.identity.name)
                    .font(.body.bold())
                Text(entry.identity.kind == "vision" ? "built-in" : "Core ML")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(Color.secondary.opacity(0.12), in: Capsule())
                Spacer()
                Text(entry.id)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
            }
            HStack(spacing: 12) {
                Text("\(entry.descriptor.featureDim)-dim")
                Text("input \(entry.descriptor.inputWidth)×\(entry.descriptor.inputHeight)")
                if entry.descriptor.kind == .coreml {
                    Text("μ=\(formatRGB(entry.descriptor.pixelNormalization.meanRGB)) σ=\(formatRGB(entry.descriptor.pixelNormalization.stdRGB))")
                        .lineLimit(1)
                }
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)

            if entry.descriptor.kind == .coreml, let modelURL = entry.modelURL {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Text(modelURL.path)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Reveal") {
                        NSWorkspace.shared.activateFileViewerSelecting([modelURL])
                    }
                    .buttonStyle(.borderless)
                    .font(.caption2)
                }
            }
        }
        .padding(8)
        .background(Color(nsColor: .separatorColor).opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 6))
    }

    private func formatRGB(_ v: [Float]) -> String {
        let parts = v.prefix(3).map { String(format: "%.2f", $0) }
        return "[\(parts.joined(separator: ","))]"
    }
}
