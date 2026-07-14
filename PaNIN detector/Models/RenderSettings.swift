import Foundation
import Observation

/// User-tunable rendering knobs for annotation outlines on the canvas.
/// Lives at the `ContentView` level; the overlay reads it every redraw.
@Observable
final class RenderSettings: @unchecked Sendable {
    /// Stroke thickness in points (screen space). Will be divided by the
    /// current zoom so the line stays a constant visual width at any zoom.
    var strokeThickness: Float = 8.0
    var strokeColor: AnnotationColor = AnnotationColor(r: 0, g: 0, b: 0)
    var fillOpacity: Float = 0.3

    /// Selected outlines are drawn slightly thicker than unselected ones.
    var selectedThicknessMultiplier: Float = 1.25
}
