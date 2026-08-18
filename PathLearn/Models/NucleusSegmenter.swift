import Foundation
import CoreGraphics

/// A detected nucleus.
struct DetectedNucleus: Sendable {
    let area: Int          // pixels at the analysis level
    let cx: Double
    let cy: Double
}

/// Result of segmenting nuclei in a region.
struct NucleiResult: Sendable {
    let nuclei: [DetectedNucleus]
    /// Total nucleus pixel count (for crowding / area-fraction metrics).
    let nucleusPixelCount: Int
}

/// The swappable segmentation backend. The geometry descriptors depend only on
/// this protocol, so a learned instance segmenter (StarDist / HoVer-Net via
/// Core ML) can replace `ClassicalNucleusSegmenter` without touching callers.
protocol NucleusSegmenting: Sendable {
    /// Detect nuclei inside `tissueMask` (row-major, true = analyze this pixel).
    /// `stain` provides per-pixel hematoxylin already deconvolved.
    nonisolated func segment(stain: StainDeconvolution.Result, tissueMask: [Bool]) -> NucleiResult
}

/// Classical nucleus detector: threshold the hematoxylin channel, then
/// connected-components. Fast and dependency-free. Under-segments touching
/// nuclei (a known classical limitation) but captures density, crowding, and
/// size variation — the signals grading cares about — well enough to test the
/// hypothesis. Swap in a DL segmenter later for cleaner instance separation.
nonisolated struct ClassicalNucleusSegmenter: NucleusSegmenting {
    /// Hematoxylin concentration above `mean + k·std` (over tissue) counts as
    /// nucleus. Adaptive so it tolerates staining/scanner variation.
    let thresholdK: Float
    /// Minimum blob area (pixels) to count as a nucleus — filters speckle.
    let minNucleusArea: Int

    init(thresholdK: Float = 0.5, minNucleusArea: Int = 6) {
        self.thresholdK = thresholdK
        self.minNucleusArea = minNucleusArea
    }

    func segment(stain: StainDeconvolution.Result, tissueMask: [Bool]) -> NucleiResult {
        let w = stain.width, h = stain.height
        let hema = stain.hematoxylin

        // Adaptive threshold from tissue-pixel statistics.
        var sum: Double = 0
        var count = 0
        for i in 0..<hema.count where tissueMask[i] {
            sum += Double(hema[i]); count += 1
        }
        guard count > 0 else { return NucleiResult(nuclei: [], nucleusPixelCount: 0) }
        let mean = sum / Double(count)
        var varSum: Double = 0
        for i in 0..<hema.count where tissueMask[i] {
            let d = Double(hema[i]) - mean; varSum += d * d
        }
        let std = (varSum / Double(count)).squareRoot()
        let threshold = Float(mean + Double(thresholdK) * std)

        var nucMask = [Bool](repeating: false, count: hema.count)
        var pixelCount = 0
        for i in 0..<hema.count where tissueMask[i] && hema[i] >= threshold {
            nucMask[i] = true
            pixelCount += 1
        }

        let blobs = ConnectedComponents.label(mask: nucMask, width: w, height: h,
                                               minArea: minNucleusArea)
        let nuclei = blobs.map { DetectedNucleus(area: $0.area, cx: $0.cx, cy: $0.cy) }
        return NucleiResult(nuclei: nuclei, nucleusPixelCount: pixelCount)
    }
}
