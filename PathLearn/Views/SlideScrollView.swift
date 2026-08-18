import AppKit

extension Notification.Name {
    static let zoomInRequested      = Notification.Name("PaNINDetector.zoomIn")
    static let zoomOutRequested     = Notification.Name("PaNINDetector.zoomOut")
    static let fitToWindowRequested = Notification.Name("PaNINDetector.fitToWindow")
}

@MainActor
final class SlideScrollView: NSScrollView {
    let slide: SlideImage
    let store: AnnotationStore
    private(set) var slideDocument: SlideDocumentView!

    private var observers: [NSObjectProtocol] = []

    init(slide: SlideImage, store: AnnotationStore) {
        self.slide = slide
        self.store = store
        super.init(frame: .zero)
        commonInit()
    }
    required init?(coder: NSCoder) { fatalError() }

    private func commonInit() {
        hasVerticalScroller = true
        hasHorizontalScroller = true
        autohidesScrollers = false
        allowsMagnification = true
        minMagnification = 0.005
        maxMagnification = 20
        backgroundColor = .windowBackgroundColor
        drawsBackground = true
        scrollerStyle = .overlay

        let doc = SlideDocumentView(slide: slide, store: store)
        documentView = doc
        slideDocument = doc

        registerForZoomNotifications()
        DispatchQueue.main.async { [weak self] in self?.fitToWindow() }
    }

    deinit {
        for o in observers {
            NotificationCenter.default.removeObserver(o)
        }
    }

    private func registerForZoomNotifications() {
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: .zoomInRequested, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.zoomIn() }
        })
        observers.append(nc.addObserver(forName: .zoomOutRequested, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.zoomOut() }
        })
        observers.append(nc.addObserver(forName: .fitToWindowRequested, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.fitToWindow() }
        })
    }

    func fitToWindow() {
        let dim = slide.dimensions
        guard dim.width > 0, dim.height > 0 else { return }
        let visible = contentView.bounds.size
        guard visible.width > 1, visible.height > 1 else {
            DispatchQueue.main.async { [weak self] in self?.fitToWindow() }
            return
        }
        magnify(toFit: NSRect(origin: .zero, size: dim))
    }

    func zoomIn(factor: CGFloat = 1.5) {
        let center = NSPoint(x: contentView.bounds.midX, y: contentView.bounds.midY)
        setMagnification(min(magnification * factor, maxMagnification), centeredAt: center)
    }

    func zoomOut(factor: CGFloat = 1.5) {
        let center = NSPoint(x: contentView.bounds.midX, y: contentView.bounds.midY)
        setMagnification(max(magnification / factor, minMagnification), centeredAt: center)
    }

    // ⌘+scrollWheel = zoom (Magic Mouse swipe, trackpad two-finger scroll, etc.).
    // Plain scroll = default pan.
    override func scrollWheel(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY : event.deltaY * 10
            let factor = pow(1.0015, delta)
            let newMag = min(max(magnification * factor, minMagnification), maxMagnification)
            let point = contentView.convert(event.locationInWindow, from: nil)
            setMagnification(newMag, centeredAt: point)
        } else {
            super.scrollWheel(with: event)
        }
    }
}
