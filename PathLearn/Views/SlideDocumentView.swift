import AppKit
import QuartzCore

@MainActor
final class SlideDocumentView: NSView {
    let slide: SlideImage
    let store: AnnotationStore

    private(set) var tiledView: SlideTiledView!
    private(set) var overlay: AnnotationOverlayView!

    override var isFlipped: Bool { true }

    init(slide: SlideImage, store: AnnotationStore) {
        self.slide = slide
        self.store = store
        super.init(frame: NSRect(origin: .zero, size: slide.dimensions))
        wantsLayer = true

        tiledView = SlideTiledView(slide: slide)
        tiledView.frame = bounds
        tiledView.autoresizingMask = [.width, .height]
        addSubview(tiledView)

        overlay = AnnotationOverlayView(slide: slide, store: store)
        overlay.frame = bounds
        overlay.autoresizingMask = [.width, .height]
        addSubview(overlay)
    }
    required init?(coder: NSCoder) { fatalError() }
}

/// Layer-hosting view whose backing layer is our custom CATiledLayer subclass.
@MainActor
final class SlideTiledView: NSView {
    let slide: SlideImage

    override var isFlipped: Bool { true }

    init(slide: SlideImage) {
        self.slide = slide
        super.init(frame: NSRect(origin: .zero, size: slide.dimensions))
        wantsLayer = true
        if let tl = layer as? SlideTiledLayer {
            tl.slide = slide
            tl.tileSize = CGSize(width: 512, height: 512)
            tl.levelsOfDetail = max(1, Int(slide.levelCount))
            tl.levelsOfDetailBias = 0
            tl.frame = bounds
            tl.contentsScale = 1
            tl.needsDisplayOnBoundsChange = true
            tl.backgroundColor = NSColor.windowBackgroundColor.cgColor

            print("[SlideTiledView] slide dims=\(slide.dimensions) levels=\(slide.levelCount) " +
                  "downsamples=\(slide.levelDownsamples)")
        }
    }
    required init?(coder: NSCoder) { fatalError() }

    override func makeBackingLayer() -> CALayer { SlideTiledLayer() }
}

/// Custom CATiledLayer that draws each tile by reading the appropriate pyramid
/// level from OpenSlide. `draw(in:)` is called from a background queue.
final class SlideTiledLayer: CATiledLayer, @unchecked Sendable {
    var slide: SlideImage?

    override class func fadeDuration() -> CFTimeInterval { 0 }

    override init() { super.init() }

    override init(layer: Any) {
        if let other = layer as? SlideTiledLayer {
            self.slide = other.slide
        }
        super.init(layer: layer)
    }

    required init?(coder: NSCoder) { fatalError() }

    override func draw(in ctx: CGContext) {
        guard let slide = slide else { return }

        let scale = sqrt(abs(ctx.ctm.a * ctx.ctm.d - ctx.ctm.b * ctx.ctm.c))
        let downsample = 1.0 / max(scale, 0.0000001)
        let level = slide.bestLevel(forDownsample: downsample)
        let lds = slide.levelDownsamples[Int(level)]

        // CATiledLayer's per-tile CGContext is Y-up (CoreAnimation default),
        // independent of the host NSView's `isFlipped`. Anchor Y to the top of
        // the tile in slide-pixel space.
        let r = ctx.boundingBoxOfClipPath
        let slideH = slide.dimensions.height
        let x0 = Int64(floor(r.minX))
        let y0 = Int64(floor(slideH - r.maxY))
        let w  = max(1, Int(ceil(r.width  / lds)))
        let h  = max(1, Int(ceil(r.height / lds)))

        if let img = slide.readRegion(x: x0, y: y0, level: level, width: w, height: h) {
            ctx.draw(img, in: r)
        }
    }
}
