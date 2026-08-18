import SwiftUI
import AppKit

struct SlideCanvasView: NSViewRepresentable {
    let slide: SlideImage
    let store: AnnotationStore
    let predictedStore: AnnotationStore
    let predictionStore: PredictionStore
    let renderSettings: RenderSettings
    @Binding var tool: AnnotationTool
    let onPolygonComplete: ([CGPoint]) -> Void

    func makeNSView(context: Context) -> SlideScrollView {
        let sv = SlideScrollView(slide: slide, store: store)
        sv.slideDocument.overlay.tool = tool
        sv.slideDocument.overlay.onPolygonComplete = onPolygonComplete
        sv.slideDocument.overlay.predictionStore = predictionStore
        sv.slideDocument.overlay.predictedStore = predictedStore
        sv.slideDocument.overlay.renderSettings = renderSettings
        return sv
    }

    func updateNSView(_ nsView: SlideScrollView, context: Context) {
        nsView.slideDocument.overlay.tool = tool
        nsView.slideDocument.overlay.onPolygonComplete = onPolygonComplete
        nsView.slideDocument.overlay.predictionStore = predictionStore
        nsView.slideDocument.overlay.predictedStore = predictedStore
        nsView.slideDocument.overlay.renderSettings = renderSettings
        // Re-read observable state so any change to annotations, predictions,
        // or stroke style forces an overlay redraw.
        _ = store.annotations.count
        _ = store.selectedID
        _ = predictedStore.annotations.count
        _ = predictedStore.selectedID
        _ = predictionStore.displayedID
        _ = predictionStore.predictions.count
        _ = predictionStore.isVisible
        _ = predictionStore.hiddenClasses
        _ = predictionStore.minProbability
        _ = renderSettings.strokeThickness
        _ = renderSettings.strokeColor
        nsView.slideDocument.overlay.needsDisplay = true
    }
}
