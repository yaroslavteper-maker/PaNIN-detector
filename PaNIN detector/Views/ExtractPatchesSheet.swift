import SwiftUI

struct ExtractPatchesSheet: View {
    let slide: SlideImage
    let allAnnotations: [Annotation]
    let hasSelection: Bool
    let selectedAnnotation: Annotation?
    let initialExtractorIdentity: ExtractorIdentity?
    let onCancel: () -> Void
    let onExtract: (PatchExtractionPipeline.Config, Scope) -> Void

    enum Scope: Hashable, CaseIterable, Identifiable {
        case all, selected
        var id: String { String(describing: self) }
    }

    @State private var patchSize: Int = 224
    @State private var stride: Int = 224
    @State private var levelIndex: Int
    @State private var scope: Scope
    @State private var skipWhitePatches: Bool = false
    @State private var whiteThreshold: Double = 0.75
    /// stringForm of the chosen extractor identity. Empty = built-in Vision.
    @State private var extractorIdentityKey: String

    private let registry = ExtractorRegistry.shared

    init(slide: SlideImage,
         allAnnotations: [Annotation],
         hasSelection: Bool,
         selectedAnnotation: Annotation?,
         initialExtractorIdentity: ExtractorIdentity? = nil,
         onCancel: @escaping () -> Void,
         onExtract: @escaping (PatchExtractionPipeline.Config, Scope) -> Void) {
        self.slide = slide
        self.allAnnotations = allAnnotations
        self.hasSelection = hasSelection
        self.selectedAnnotation = selectedAnnotation
        self.initialExtractorIdentity = initialExtractorIdentity
        self.onCancel = onCancel
        self.onExtract = onExtract
        // Pick the pyramid level whose downsample is closest to 4×.
        let bestLevel = Int(slide.bestLevel(forDownsample: 4.0))
        _levelIndex = State(initialValue: max(0, min(bestLevel, Int(slide.levelCount) - 1)))
        _scope = State(initialValue: hasSelection ? .selected : .all)
        _extractorIdentityKey = State(initialValue: initialExtractorIdentity?.stringForm ?? "")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Extract patches into the bank").font(.headline)

            HStack {
                Text("Extractor:").frame(width: 90, alignment: .trailing)
                Picker("", selection: $extractorIdentityKey) {
                    ForEach(registry.available) { entry in
                        Text(extractorLabel(entry))
                            .tag(entry.id)
                    }
                }
                .labelsHidden()
                Button {
                    registry.discover()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Re-scan the extractors directory")
            }

            HStack {
                Text("Scope:").frame(width: 90, alignment: .trailing)
                Picker("", selection: $scope) {
                    ForEach(Scope.allCases) { s in
                        Text(scopeLabel(s)).tag(s)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            HStack {
                Text("Patch size:").frame(width: 90, alignment: .trailing)
                TextField("", value: $patchSize, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                Stepper("", value: $patchSize, in: 32...2048, step: 32)
                    .labelsHidden()
                Text("level-\(levelIndex) px")
                    .foregroundStyle(.secondary).font(.caption)
            }

            HStack {
                Text("Stride:").frame(width: 90, alignment: .trailing)
                TextField("", value: $stride, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                Stepper("", value: $stride, in: 32...2048, step: 32)
                    .labelsHidden()
                Text("level-\(levelIndex) px")
                    .foregroundStyle(.secondary).font(.caption)
            }

            HStack {
                Text("Level:").frame(width: 90, alignment: .trailing)
                Picker("", selection: $levelIndex) {
                    ForEach(0..<Int(slide.levelCount), id: \.self) { i in
                        let ds = slide.levelDownsamples[i]
                        Text(levelLabel(index: i, downsample: ds)).tag(i)
                    }
                }
                .labelsHidden()
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Toggle("Skip mostly-white patches above", isOn: $skipWhitePatches)
                        .toggleStyle(.checkbox)
                        .font(.caption)
                    Slider(value: $whiteThreshold, in: 0.1...0.99, step: 0.05)
                        .controlSize(.mini)
                        .disabled(!skipWhitePatches)
                    Text("\(Int((whiteThreshold * 100).rounded()))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 38, alignment: .trailing)
                }
                Text("Each patch gets a whiteness score; mostly-white background patches will be skipped before feature extraction.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            Text(estimateLine)
                .font(.caption)
                .foregroundStyle(.secondary)

            if estimatedCount > 2000 {
                Label("Estimate exceeds 2000 patches — consider raising the stride.",
                      systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Extract") {
                    let cutoff: Float? = skipWhitePatches ? Float(whiteThreshold) : nil
                    let identity: ExtractorIdentity? =
                        registry.available.first(where: { $0.id == extractorIdentityKey })?.identity
                    onExtract(
                        PatchExtractionPipeline.Config(
                            patchSizeLevel: max(32, patchSize),
                            strideLevel: max(32, stride),
                            level: Int32(levelIndex),
                            maxWhiteFraction: cutoff,
                            extractorIdentity: identity
                        ),
                        scope
                    )
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(extractAnnotationCount == 0)
            }
        }
        .padding(20)
        .frame(minWidth: 500)
    }

    private var extractAnnotations: [Annotation] {
        switch scope {
        case .all:      return allAnnotations
        case .selected: return selectedAnnotation.map { [$0] } ?? []
        }
    }

    private var extractAnnotationCount: Int {
        extractAnnotations.count
    }

    private var estimatedCount: Int {
        let lds = slide.levelDownsamples[max(0, min(levelIndex, slide.levelDownsamples.count - 1))]
        var sum = 0
        for ann in extractAnnotations {
            sum += PatchSampler.estimatedPatchCount(
                polygonOverlayY: ann.points,
                patchSizeLevel: patchSize,
                strideLevel: stride,
                levelDownsample: lds
            )
        }
        return sum
    }

    private var estimateLine: String {
        let n = extractAnnotationCount
        return "Will scan \(n) annotation\(n == 1 ? "" : "s") — estimated ≈ \(estimatedCount) patches (upper bound)."
    }

    private func scopeLabel(_ s: Scope) -> String {
        switch s {
        case .all:
            return "All (\(allAnnotations.count))"
        case .selected:
            return hasSelection ? "Selected" : "Selected (none)"
        }
    }

    private func extractorLabel(_ entry: ExtractorRegistry.DiscoveredExtractor) -> String {
        let id = entry.identity
        switch id.kind {
        case "vision":
            return "Vision (built-in, r\(id.revision), \(entry.descriptor.featureDim)-dim)"
        case "coreml":
            return "\(id.name) — r\(id.revision) (\(entry.descriptor.featureDim)-dim Core ML)"
        default:
            return id.stringForm
        }
    }

    private func levelLabel(index: Int, downsample: Double) -> String {
        let dims = slide.levelDimensions[index]
        let dimStr = "\(Int(dims.width))×\(Int(dims.height))"
        if index == 0 {
            return "Level 0 (full resolution, \(dimStr))"
        }
        return "Level \(index) – \(String(format: "%.1f", downsample))× downsample (\(dimStr))"
    }
}
