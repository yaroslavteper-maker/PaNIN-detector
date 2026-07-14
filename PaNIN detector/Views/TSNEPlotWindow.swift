import SwiftUI

/// Separate window scene that draws the most recent t-SNE embedding as a
/// scatter plot. Points are coloured by their classification (using the
/// profile palette where possible). Mouse-drag pans, scroll wheel zooms.
struct TSNEPlotWindow: View {
    @Environment(EmbeddingStore.self) private var store
    @Environment(ProfileStore.self) private var profileStore

    @State private var zoom: CGFloat = 1.0
    @State private var pan: CGSize = .zero
    @State private var showCentroids: Bool = true
    @State private var showDecisionRegions: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if store.isComputing {
                progressOverlay
            } else if let emb = store.current {
                plot(emb)
            } else {
                ContentUnavailableView(
                    "No embedding yet",
                    systemImage: "chart.dots.scatter",
                    description: Text("Run Analysis ▸ t-SNE Classifier… to compute one.")
                )
            }
        }
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        if let emb = store.current {
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("\(emb.pointCount) patches")
                        .font(.body.bold())
                    Text("Extractor: \(emb.extractorIdentity.stringForm) • perplexity \(Int(emb.perplexity)) • \(emb.iterations) iters")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if emb.totalAvailable > emb.pointCount {
                        Text("Stratified-subsampled from \(emb.totalAvailable).")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                Spacer()
                VStack(alignment: .leading, spacing: 4) {
                    Toggle("Centroids", isOn: $showCentroids)
                        .toggleStyle(.checkbox)
                        .disabled(store.centroids == nil)
                    Toggle("Decision regions", isOn: $showDecisionRegions)
                        .toggleStyle(.checkbox)
                        .disabled(store.centroids == nil)
                    if let cset = store.centroids {
                        Label("Centroid accuracy: \(String(format: "%.1f", cset.trainAccuracy * 100))%",
                              systemImage: "checkmark.seal")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                .font(.caption)
                legend(emb)
                resetButton
            }
            .padding(12)
        } else {
            HStack {
                Label("t-SNE Plot", systemImage: "chart.dots.scatter")
                    .font(.headline)
                Spacer()
            }
            .padding(12)
        }
    }

    private var resetButton: some View {
        Button {
            zoom = 1.0
            pan = .zero
        } label: {
            Image(systemName: "arrow.uturn.backward")
        }
        .buttonStyle(.bordered)
        .help("Reset zoom and pan")
    }

    // MARK: - Legend

    private func legend(_ emb: EmbeddingStore.Embedding) -> some View {
        let counts = emb.perClass.sorted { $0.key < $1.key }
        return VStack(alignment: .leading, spacing: 2) {
            ForEach(counts, id: \.key) { entry in
                HStack(spacing: 4) {
                    Circle()
                        .fill(color(for: entry.key))
                        .frame(width: 8, height: 8)
                    Text(entry.key)
                        .font(.caption)
                    Text("\(entry.value)")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(6)
        .background(Color(nsColor: .separatorColor).opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Plot

    @ViewBuilder
    private func plot(_ emb: EmbeddingStore.Embedding) -> some View {
        GeometryReader { geo in
            Canvas { ctx, _ in
                drawScatter(ctx, size: geo.size, embedding: emb)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .gesture(
                DragGesture()
                    .onChanged { value in
                        pan = CGSize(
                            width: pan.width + value.translation.width - dragAccum.width,
                            height: pan.height + value.translation.height - dragAccum.height
                        )
                        dragAccum = value.translation
                    }
                    .onEnded { _ in dragAccum = .zero }
            )
            .gesture(MagnificationGesture()
                .onChanged { v in
                    zoom = max(0.1, min(20, magnifyStart * v))
                }
                .onEnded { _ in magnifyStart = zoom }
            )
        }
    }

    @State private var dragAccum: CGSize = .zero
    @State private var magnifyStart: CGFloat = 1.0

    private func drawScatter(_ ctx: GraphicsContext,
                             size: CGSize,
                             embedding: EmbeddingStore.Embedding) {
        let pts = embedding.points
        guard !pts.isEmpty else { return }
        var minX: Float = .infinity, maxX: Float = -.infinity
        var minY: Float = .infinity, maxY: Float = -.infinity
        for p in pts {
            if p.x < minX { minX = p.x }
            if p.x > maxX { maxX = p.x }
            if p.y < minY { minY = p.y }
            if p.y > maxY { maxY = p.y }
        }
        let spanX = max(1e-6, maxX - minX)
        let spanY = max(1e-6, maxY - minY)
        let margin: CGFloat = 24
        let plotW = max(1, size.width - margin * 2)
        let plotH = max(1, size.height - margin * 2)
        let cx = size.width / 2
        let cy = size.height / 2

        // Closure converts a t-SNE point to canvas coordinates.
        let project: (Float, Float) -> CGPoint = { px, py in
            let nx = (CGFloat((px - minX) / spanX) - 0.5) * plotW
            let ny = (CGFloat((py - minY) / spanY) - 0.5) * plotH
            return CGPoint(x: cx + nx * zoom + pan.width,
                           y: cy + ny * zoom + pan.height)
        }

        // 1. Optional decision regions: coarse Voronoi over centroid plot
        // positions, painted as faint coloured cells. Cheap to compute (a
        // grid sweep — not per-pixel) so the toggle stays responsive.
        if showDecisionRegions, let cset = store.centroids, !cset.centroids.isEmpty {
            drawDecisionRegions(ctx, size: size, project: project, centroids: cset.centroids)
        }

        // 2. The patch scatter.
        for (i, p) in pts.enumerated() {
            let pt = project(p.x, p.y)
            if pt.x < -4 || pt.x > size.width + 4 || pt.y < -4 || pt.y > size.height + 4 { continue }
            let label = embedding.labels[i]
            let r: CGFloat = 2.4
            ctx.fill(
                Path(ellipseIn: CGRect(x: pt.x - r, y: pt.y - r, width: r * 2, height: r * 2)),
                with: .color(color(for: label))
            )
        }

        // 3. Centroid markers.
        if showCentroids, let cset = store.centroids {
            for c in cset.centroids {
                let pt = project(c.plotPosition.x, c.plotPosition.y)
                let outer: CGFloat = 10
                let inner: CGFloat = 4
                let col = color(for: c.label)
                // Hollow ring + filled centre dot.
                ctx.stroke(
                    Path(ellipseIn: CGRect(x: pt.x - outer, y: pt.y - outer,
                                           width: outer * 2, height: outer * 2)),
                    with: .color(col),
                    lineWidth: 2
                )
                ctx.fill(
                    Path(ellipseIn: CGRect(x: pt.x - inner, y: pt.y - inner,
                                           width: inner * 2, height: inner * 2)),
                    with: .color(col)
                )
                // Label above the marker.
                let text = Text(c.label)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(col)
                ctx.draw(text, at: CGPoint(x: pt.x, y: pt.y - outer - 8), anchor: .center)
            }
        }
    }

    /// Coarse Voronoi tessellation of the plot region by nearest centroid in
    /// 2D plot space. Drawn as low-opacity cells so the underlying scatter
    /// stays readable. Resolution is intentionally low (~60×60 cells) so the
    /// overlay is cheap even when re-running on pan / zoom.
    private func drawDecisionRegions(_ ctx: GraphicsContext,
                                     size: CGSize,
                                     project: (Float, Float) -> CGPoint,
                                     centroids: [Centroid]) {
        let cellsX = 60
        let cellsY = max(20, Int(60 * size.height / max(size.width, 1)))
        let cellW = size.width / CGFloat(cellsX)
        let cellH = size.height / CGFloat(cellsY)
        // Project centroid plot positions once.
        let cps: [(CGPoint, Color)] = centroids.map { c in
            (project(c.plotPosition.x, c.plotPosition.y), color(for: c.label))
        }
        for gy in 0..<cellsY {
            for gx in 0..<cellsX {
                let px = CGFloat(gx) * cellW + cellW * 0.5
                let py = CGFloat(gy) * cellH + cellH * 0.5
                var bestIdx = 0
                var bestD2: CGFloat = .greatestFiniteMagnitude
                for (i, cp) in cps.enumerated() {
                    let dx = cp.0.x - px
                    let dy = cp.0.y - py
                    let d2 = dx * dx + dy * dy
                    if d2 < bestD2 { bestD2 = d2; bestIdx = i }
                }
                let cellRect = CGRect(
                    x: CGFloat(gx) * cellW,
                    y: CGFloat(gy) * cellH,
                    width: cellW + 0.5,
                    height: cellH + 0.5
                )
                ctx.fill(Path(cellRect), with: .color(cps[bestIdx].1.opacity(0.10)))
            }
        }
    }

    // MARK: - Helpers

    private func color(for label: String) -> Color {
        if let cls = profileStore.profile.classes.first(where: { $0.name == label }) {
            return Color(
                red: Double(cls.color.r) / 255,
                green: Double(cls.color.g) / 255,
                blue: Double(cls.color.b) / 255
            )
        }
        // Deterministic fallback so unknown classes still get a stable colour.
        let h = abs(label.hashValue) % 360
        return Color(hue: Double(h) / 360, saturation: 0.7, brightness: 0.85)
    }

    // MARK: - Progress

    private var progressOverlay: some View {
        VStack(spacing: 10) {
            ProgressView(value: store.progress)
                .progressViewStyle(.linear)
                .frame(maxWidth: 400)
            Text(store.status)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor))
    }
}
