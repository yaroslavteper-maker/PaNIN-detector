import SwiftUI
import AppKit
import Observation

@Observable
final class PreviewState: @unchecked Sendable {
    var image: CGImage?
    var annotation: Annotation?
    var isGenerating = false
    var sourceSlideName: String?
    /// The slide-pixel rect the preview image represents — used to map the
    /// polygon onto the displayed image.
    var outputBBox: CGRect?
    /// Full slide dimensions, for the mini-map overlay.
    var slideDimensions: CGSize?
}

struct AnnotationPreviewWindow: View {
    @Environment(PreviewState.self) private var state

    var body: some View {
        Group {
            if state.isGenerating {
                ProgressView("Rendering preview…")
                    .controlSize(.large)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let image = state.image {
                imageView(image)
            } else {
                ContentUnavailableView(
                    "No preview yet",
                    systemImage: "photo.stack",
                    description: Text("Draw an annotation, or select one and press ⇧⌘P.")
                )
            }
        }
        .frame(minWidth: 480, minHeight: 480)
        .navigationTitle(title)
    }

    private var title: String {
        let cls = state.annotation?.classification ?? "Annotation"
        if let slide = state.sourceSlideName, !slide.isEmpty {
            return "\(cls) — \(slide)"
        }
        return cls
    }

    @ViewBuilder
    private func imageView(_ image: CGImage) -> some View {
        VStack(spacing: 0) {
            let aspect = max(0.01, CGFloat(image.width) / CGFloat(image.height))
            Image(decorative: image, scale: 1.0)
                .resizable()
                .interpolation(.medium)
                .aspectRatio(aspect, contentMode: .fit)
                .padding(8)
            Divider()
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    if let ann = state.annotation {
                        let bbox = ann.boundingBoxInSlidePixels
                        HStack(spacing: 4) {
                            Label("\(ann.classification)", systemImage: "tag")
                                .foregroundStyle(.secondary)
                            Text("• \(Int(ann.areaInSlidePixels)) px²")
                                .monospacedDigit().foregroundStyle(.secondary)
                            Text("• bbox \(Int(bbox.width))×\(Int(bbox.height))")
                                .monospacedDigit().foregroundStyle(.secondary)
                        }
                        if let out = state.outputBBox {
                            Text("Read from slide (\(Int(out.minX)), \(Int(out.minY))) — \(Int(out.width))×\(Int(out.height)) slide-px")
                                .monospacedDigit().foregroundStyle(.tertiary)
                        }
                    }
                    Text("Preview: \(image.width)×\(image.height) px")
                        .monospacedDigit().foregroundStyle(.tertiary)
                }
                Spacer()
                slideMinimap()
            }
            .font(.caption)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
    }

    @ViewBuilder
    private func slideMinimap() -> some View {
        if let dim = state.slideDimensions,
           let bbox = state.outputBBox,
           dim.width > 0, dim.height > 0 {
            let mapW: CGFloat = 120
            let mapH = mapW * dim.height / dim.width
            let sx = mapW / dim.width
            let sy = mapH / dim.height
            ZStack(alignment: .topLeading) {
                Rectangle()
                    .strokeBorder(Color.secondary, lineWidth: 1)
                    .background(Color.secondary.opacity(0.08))
                Rectangle()
                    .fill(Color.red.opacity(0.5))
                    .border(Color.red, width: 1)
                    .frame(width: max(2, bbox.width * sx),
                           height: max(2, bbox.height * sy))
                    .offset(x: bbox.minX * sx, y: bbox.minY * sy)
            }
            .frame(width: mapW, height: mapH)
            .overlay(alignment: .top) {
                Text("position on slide")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.top, -14)
            }
        }
    }

}
