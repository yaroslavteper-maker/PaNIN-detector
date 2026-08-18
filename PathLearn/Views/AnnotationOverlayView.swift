import AppKit

enum AnnotationTool: String, CaseIterable, Identifiable, Sendable {
    case pan
    case lasso
    case polygon

    var id: String { rawValue }
    var label: String {
        switch self {
        case .pan: return "Pan"
        case .lasso: return "Lasso"
        case .polygon: return "Polygon"
        }
    }
    var systemImage: String {
        switch self {
        case .pan: return "hand.draw"
        case .lasso: return "lasso"
        case .polygon: return "pentagon"
        }
    }
}

@MainActor
final class AnnotationOverlayView: NSView {
    let slide: SlideImage
    let store: AnnotationStore
    /// Set by `SlideCanvasView`; the overlay reads it to render the prediction
    /// heatmap underneath the annotations.
    var predictionStore: PredictionStore?
    /// Auto-proposed predicted regions, drawn dashed to distinguish from the
    /// manually-drawn annotations. Display-only (not editable on the canvas).
    var predictedStore: AnnotationStore?
    /// User-tunable stroke thickness/color (set by `SlideCanvasView`).
    var renderSettings: RenderSettings?

    var tool: AnnotationTool = .pan {
        didSet {
            print("[overlay] tool -> \(tool)")
            // Cancel in-progress shapes when switching tools mid-construction.
            if oldValue == .lasso && tool != .lasso {
                isDrawing = false
                inProgress = []
            }
            if oldValue == .polygon && tool != .polygon {
                cancelPolygon()
            }
            window?.invalidateCursorRects(for: self)
            needsDisplay = true
        }
    }

    /// Called when the user finishes a lasso or polygon (≥ 3 points).
    var onPolygonComplete: (([CGPoint]) -> Void)?

    private var inProgress: [CGPoint] = []
    private var isDrawing = false

    // Pan tool state
    private var panAnchorWindow: NSPoint = .zero
    private var panAnchorOrigin: NSPoint = .zero
    private var isPanning = false

    // Polygon tool state
    private var isPolygonInProgress = false
    private var polygonCursor: CGPoint?
    private var trackingArea: NSTrackingArea?

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    init(slide: SlideImage, store: AnnotationStore) {
        self.slide = slide
        self.store = store
        super.init(frame: .zero)
        wantsLayer = true
        layer?.isOpaque = false
        layer?.backgroundColor = .clear
    }
    required init?(coder: NSCoder) { fatalError() }

    // Accept clicks for both tools. Trackpad scroll/magnify gestures still reach
    // the enclosing NSScrollView via the responder chain.
    override func hitTest(_ point: NSPoint) -> NSView? {
        super.hitTest(point)
    }

    override func resetCursorRects() {
        let cursor: NSCursor
        switch tool {
        case .pan:     cursor = .openHand
        case .lasso:   cursor = .crosshair
        case .polygon: cursor = .crosshair
        }
        addCursorRect(bounds, cursor: cursor)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area = trackingArea { removeTrackingArea(area) }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
        trackingArea = area
    }

    // MARK: Mouse

    override func mouseDown(with event: NSEvent) {
        let windowPt = event.locationInWindow
        let p = convert(windowPt, from: nil)
        let mag = enclosingScrollView?.magnification ?? -1
        let scrollOrigin = enclosingScrollView?.contentView.bounds.origin ?? .zero
        print(String(format:
            "[overlay] mouseDown tool=%@ window=(%.1f, %.1f) overlay=(%.1f, %.1f) mag=%.4f scrollOrigin=(%.1f, %.1f) slideDim=(%.0f x %.0f)",
            String(describing: tool),
            windowPt.x, windowPt.y, p.x, p.y, mag,
            scrollOrigin.x, scrollOrigin.y,
            slide.dimensions.width, slide.dimensions.height))
        switch tool {
        case .pan:
            beginPan(at: windowPt)
        case .lasso:
            inProgress = [p]
            isDrawing = true
            needsDisplay = true
        case .polygon:
            handlePolygonClick(at: p, clickCount: event.clickCount)
        }
    }

    override func mouseMoved(with event: NSEvent) {
        guard tool == .polygon, isPolygonInProgress else {
            return super.mouseMoved(with: event)
        }
        let p = convert(event.locationInWindow, from: nil)
        polygonCursor = p
        needsDisplay = true
    }

    override func keyDown(with event: NSEvent) {
        if tool == .polygon, isPolygonInProgress {
            switch event.keyCode {
            case 36, 76:   // Return, Numpad Enter
                commitPolygon()
                return
            case 53:       // Escape
                cancelPolygon()
                return
            case 51:       // Delete / Backspace — remove last vertex
                if !inProgress.isEmpty {
                    inProgress.removeLast()
                    if inProgress.isEmpty { isPolygonInProgress = false }
                    needsDisplay = true
                }
                return
            default:
                break
            }
        }
        super.keyDown(with: event)
    }

    // MARK: Polygon helpers

    private func handlePolygonClick(at p: CGPoint, clickCount: Int) {
        // Double-click anywhere closes the polygon.
        if clickCount == 2, isPolygonInProgress, inProgress.count >= 3 {
            commitPolygon()
            return
        }

        // Click near the first vertex (in screen pixels) also closes.
        if isPolygonInProgress, inProgress.count >= 3,
           let first = inProgress.first,
           screenDistance(p, first) < 10 {
            commitPolygon()
            return
        }

        if !isPolygonInProgress {
            // Take keyboard focus so Return / Esc / Backspace work.
            window?.makeFirstResponder(self)
        }

        inProgress.append(p)
        polygonCursor = p
        isPolygonInProgress = true
        needsDisplay = true
    }

    private func commitPolygon() {
        let pts = inProgress
        inProgress = []
        polygonCursor = nil
        isPolygonInProgress = false
        needsDisplay = true
        print("[overlay] polygon finished: \(pts.count) points")
        if pts.count >= 3 {
            onPolygonComplete?(pts)
        }
    }

    private func cancelPolygon() {
        inProgress = []
        polygonCursor = nil
        isPolygonInProgress = false
        needsDisplay = true
    }

    /// Distance in screen pixels (i.e. converts back through magnification).
    private func screenDistance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let zoom = (enclosingScrollView as? SlideScrollView)?.magnification ?? 1
        return hypot(a.x - b.x, a.y - b.y) * zoom
    }

    override func mouseDragged(with event: NSEvent) {
        switch tool {
        case .pan:
            guard isPanning else { return }
            continuePan(to: event.locationInWindow)
        case .lasso:
            guard isDrawing else { return }
            let p = convert(event.locationInWindow, from: nil)
            if let last = inProgress.last {
                let minStep = pointSpacingForCurrentZoom()
                if hypot(p.x - last.x, p.y - last.y) >= minStep {
                    inProgress.append(p)
                    needsDisplay = true
                }
            }
        case .polygon:
            // Polygon ignores drags; vertices are placed by clicks.
            break
        }
    }

    override func mouseUp(with event: NSEvent) {
        switch tool {
        case .pan:
            isPanning = false
            NSCursor.openHand.set()
        case .lasso:
            guard isDrawing else { return }
            let p = convert(event.locationInWindow, from: nil)
            if inProgress.last != p { inProgress.append(p) }
            isDrawing = false
            let pts = inProgress
            inProgress = []
            needsDisplay = true
            print("[overlay] lasso finished: \(pts.count) points")
            if pts.count >= 3 {
                onPolygonComplete?(pts)
            }
        case .polygon:
            // Polygon vertices commit on mouseDown, nothing on mouseUp.
            break
        }
    }

    // MARK: Pan helpers

    private func beginPan(at windowPoint: NSPoint) {
        guard let scrollView = enclosingScrollView else { return }
        panAnchorWindow = windowPoint
        panAnchorOrigin = scrollView.contentView.bounds.origin
        isPanning = true
        NSCursor.closedHand.set()
    }

    private func continuePan(to windowPoint: NSPoint) {
        guard let scrollView = enclosingScrollView else { return }
        let dxWindow = windowPoint.x - panAnchorWindow.x
        let dyWindow = windowPoint.y - panAnchorWindow.y
        let mag = scrollView.magnification
        // Window coords are Y-up; doc view is flipped (Y-down). Negate dy.
        let dxDoc = dxWindow / mag
        let dyDoc = -dyWindow / mag
        let newOrigin = NSPoint(
            x: panAnchorOrigin.x - dxDoc,
            y: panAnchorOrigin.y - dyDoc
        )
        scrollView.contentView.scroll(to: newOrigin)
        scrollView.reflectScrolledClipView(scrollView.contentView)
    }

    private func pointSpacingForCurrentZoom() -> CGFloat {
        let zoom = (enclosingScrollView as? SlideScrollView)?.magnification ?? 1
        return max(1, 1.5 / max(zoom, 0.0001))
    }

    // MARK: Drawing

    override func draw(_ dirtyRect: NSRect) {
        guard let ctx = NSGraphicsContext.current?.cgContext else { return }
        // Heatmap goes underneath so polygon outlines stay on top.
        drawPredictions(in: ctx)
        // Predicted regions (dashed), below the manual annotations.
        if let pstore = predictedStore {
            for ann in pstore.annotations where ann.isVisible {
                drawPolygon(
                    ann.points,
                    color: ann.color,
                    selected: ann.id == pstore.selectedID,
                    closed: true,
                    dashed: true,
                    in: ctx
                )
            }
        }
        let visible = store.annotations.filter(\.isVisible)
        for ann in visible {
            drawPolygon(
                ann.points,
                color: ann.color,
                selected: ann.id == store.selectedID,
                closed: true,
                in: ctx
            )
        }
        if isDrawing, inProgress.count >= 2 {
            drawPolygon(
                inProgress,
                color: store.currentColor,
                selected: true,
                closed: false,
                in: ctx
            )
        }
        if isPolygonInProgress, !inProgress.isEmpty {
            drawPolygonInProgress(in: ctx)
        }
    }

    private func drawPolygon(_ points: [CGPoint],
                             color: AnnotationColor,
                             selected: Bool,
                             closed: Bool,
                             dashed: Bool = false,
                             in ctx: CGContext) {
        guard points.count >= 2 else { return }
        let path = CGMutablePath()
        path.move(to: points[0])
        for p in points.dropFirst() { path.addLine(to: p) }
        if closed { path.closeSubpath() }

        // Stroke thickness + color come from RenderSettings; selected outlines
        // are scaled slightly thicker. Fill carries the class color.
        let settings = renderSettings
        let scale = sqrt(abs(ctx.ctm.a * ctx.ctm.d - ctx.ctm.b * ctx.ctm.c))
        let baseThickness = CGFloat(settings?.strokeThickness ?? 8.0)
        let mult = CGFloat(settings?.selectedThicknessMultiplier ?? 1.25)
        let basePts: CGFloat = selected ? baseThickness * mult : baseThickness
        let strokeW = basePts / max(scale, 0.0001)

        let sc = settings?.strokeColor ?? AnnotationColor(r: 0, g: 0, b: 0)
        let fillAlpha = CGFloat(settings?.fillOpacity ?? 0.3)

        ctx.saveGState()
        ctx.setLineWidth(strokeW)
        ctx.setLineJoin(.round)
        ctx.setLineCap(.round)
        if dashed {
            ctx.setLineDash(phase: 0, lengths: [strokeW * 3, strokeW * 2])
        }
        ctx.setStrokeColor(red: CGFloat(sc.r) / 255,
                           green: CGFloat(sc.g) / 255,
                           blue: CGFloat(sc.b) / 255,
                           alpha: 1)
        if closed {
            ctx.setFillColor(red: CGFloat(color.r)/255,
                             green: CGFloat(color.g)/255,
                             blue: CGFloat(color.b)/255,
                             alpha: fillAlpha)
            ctx.addPath(path)
            ctx.drawPath(using: .fillStroke)
        } else {
            ctx.addPath(path)
            ctx.strokePath()
        }
        ctx.restoreGState()
    }

    private func drawPolygonInProgress(in ctx: CGContext) {
        let color = store.currentColor
        let settings = renderSettings
        let scale = sqrt(abs(ctx.ctm.a * ctx.ctm.d - ctx.ctm.b * ctx.ctm.c))
        let baseThickness = CGFloat(settings?.strokeThickness ?? 8.0)
        let strokeW: CGFloat = (baseThickness * 0.75) / max(scale, 0.0001)
        let sqHalf: CGFloat  = 6.0 / max(scale, 0.0001)
        let sc = settings?.strokeColor ?? AnnotationColor(r: 0, g: 0, b: 0)

        ctx.saveGState()
        ctx.setLineWidth(strokeW)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        // Solid segments between placed vertices
        if inProgress.count >= 2 {
            ctx.setStrokeColor(red: CGFloat(sc.r) / 255,
                               green: CGFloat(sc.g) / 255,
                               blue: CGFloat(sc.b) / 255,
                               alpha: 1)
            let path = CGMutablePath()
            path.move(to: inProgress[0])
            for p in inProgress.dropFirst() { path.addLine(to: p) }
            ctx.addPath(path)
            ctx.strokePath()
        }

        // Dashed rubber-band from last vertex to current cursor
        if let cursor = polygonCursor, let last = inProgress.last {
            ctx.setLineDash(phase: 0, lengths: [strokeW * 2, strokeW * 1.5])
            ctx.setStrokeColor(red: CGFloat(sc.r) / 255,
                               green: CGFloat(sc.g) / 255,
                               blue: CGFloat(sc.b) / 255,
                               alpha: 0.6)
            ctx.move(to: last)
            ctx.addLine(to: cursor)
            ctx.strokePath()
            ctx.setLineDash(phase: 0, lengths: [])
        }

        // Vertex markers — squares around each placed point.
        // First vertex slightly larger as a snap-to-close target.
        ctx.setFillColor(
            red: CGFloat(color.r) / 255,
            green: CGFloat(color.g) / 255,
            blue: CGFloat(color.b) / 255,
            alpha: 1
        )
        ctx.setStrokeColor(red: CGFloat(sc.r) / 255,
                           green: CGFloat(sc.g) / 255,
                           blue: CGFloat(sc.b) / 255,
                           alpha: 1)
        ctx.setLineWidth(strokeW * 0.6)
        for (i, p) in inProgress.enumerated() {
            let h = i == 0 ? sqHalf * 1.4 : sqHalf
            let rect = CGRect(x: p.x - h, y: p.y - h, width: 2 * h, height: 2 * h)
            ctx.fill(rect)
            ctx.stroke(rect)
        }
        ctx.restoreGState()
    }

    private func drawPredictions(in ctx: CGContext) {
        guard let ps = predictionStore,
              ps.isVisible,
              !ps.predictions.isEmpty else { return }
        let slideH = slide.dimensions.height
        let minProb = ps.minProbability
        ctx.saveGState()
        for pred in ps.predictions {
            guard pred.maxProbability >= minProb else { continue }
            guard ps.isClassVisible(pred.predictedLabel) else { continue }
            let color = ps.classColors[pred.predictedLabel] ?? .defaultColor
            let size = CGFloat(pred.sizeLevel0)
            let x = CGFloat(pred.dataX)
            // Mirror Y back to overlay space so the patch lines up with the
            // pixels it was read from (same convention as the rest of the app).
            let y = slideH - CGFloat(pred.dataY) - size
            let rect = CGRect(x: x, y: y, width: size, height: size)
            let alpha = max(0.1, min(0.75, CGFloat(pred.maxProbability)))
            ctx.setFillColor(
                red: CGFloat(color.r) / 255,
                green: CGFloat(color.g) / 255,
                blue: CGFloat(color.b) / 255,
                alpha: alpha * 0.6
            )
            ctx.fill(rect)
        }
        ctx.restoreGState()
    }
}
