import SwiftUI

struct PredictAnnotationSheet: View {
    let slide: SlideImage
    let annotation: Annotation
    let classifier: MLClassifier
    let onCancel: () -> Void
    let onRun: (PredictionPipeline.MultiPassConfig, Float) -> Void
    let onManageExtractors: () -> Void

    @State private var defaultPatchSize: Int = 224
    @State private var defaultStride: Int = 224
    @State private var levelIndex: Int
    @State private var minConfidence: Double = 0.0
    @State private var perClassSettings: [String: (patch: Int, stride: Int)] = [:]
    @State private var enabledClasses: Set<String> = []

    init(slide: SlideImage,
         annotation: Annotation,
         classifier: MLClassifier,
         onCancel: @escaping () -> Void,
         onRun: @escaping (PredictionPipeline.MultiPassConfig, Float) -> Void,
         onManageExtractors: @escaping () -> Void = {}) {
        self.slide = slide
        self.annotation = annotation
        self.classifier = classifier
        self.onCancel = onCancel
        self.onRun = onRun
        self.onManageExtractors = onManageExtractors
        let best = Int(slide.bestLevel(forDownsample: 4.0))
        _levelIndex = State(initialValue: max(0, min(best, Int(slide.levelCount) - 1)))
        var settings: [String: (patch: Int, stride: Int)] = [:]
        for label in classifier.classLabels {
            settings[label] = (224, 224)
        }
        _perClassSettings = State(initialValue: settings)
        _enabledClasses = State(initialValue: Set(classifier.classLabels))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Predict over annotation").font(.headline)

            VStack(alignment: .leading, spacing: 2) {
                Text("Annotation").font(.caption.bold()).foregroundStyle(.secondary)
                Text("Class “\(annotation.classification)” — area ≈ \(Int(annotation.areaInSlidePixels)) px²")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Classifier: \(classifier.classCount) classes, val \(Int((classifier.metrics.valAccuracy * 100).rounded()))%")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    Label("Extractor: \(classifierIdentity.stringForm)", systemImage: "cpu")
                        .font(.caption2)
                        .foregroundStyle(extractorAvailable ? Color.secondary : Color.red)
                    if !extractorAvailable {
                        Button("Manage…") { onManageExtractors() }
                            .buttonStyle(.borderless)
                            .font(.caption2)
                    }
                }
                if !extractorAvailable {
                    Text("Required extractor is not installed. Drop its `.paninextractor.json` + `.mlpackage` files into the Extractors directory and rescan.")
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }

            Divider()

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

            Text("Per-class patch & stride (in level-\(levelIndex) px)")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .padding(.top, 4)

            HStack {
                Text("Default:")
                    .frame(width: 90, alignment: .trailing)
                Text("Patch").font(.caption2).foregroundStyle(.secondary)
                TextField("", value: $defaultPatchSize, format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 64)
                Text("Stride").font(.caption2).foregroundStyle(.secondary)
                TextField("", value: $defaultStride, format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 64)
                Button("Apply to all") {
                    for label in perClassSettings.keys {
                        perClassSettings[label] = (
                            max(32, defaultPatchSize),
                            max(16, defaultStride)
                        )
                    }
                }
                .buttonStyle(.borderless)
                Spacer()
                Button(allClassesEnabled ? "Disable all" : "Enable all") {
                    if allClassesEnabled {
                        enabledClasses = []
                    } else {
                        enabledClasses = Set(classifier.classLabels)
                    }
                }
                .buttonStyle(.borderless)
                .font(.caption2)
            }

            classTable

            HStack {
                Text("Min confidence:")
                    .frame(width: 110, alignment: .trailing)
                Slider(value: $minConfidence, in: 0.0...0.99, step: 0.05)
                Text("\(Int((minConfidence * 100).rounded()))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 40, alignment: .trailing)
            }
            Text("Patches below this confidence are hidden on the heatmap. You can adjust live in the Classifier panel after the run.")
                .font(.caption2)
                .foregroundStyle(.tertiary)

            Text(estimate)
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Run") {
                    var perClass: [String: PredictionPipeline.PassConfig] = [:]
                    for label in enabledClasses {
                        let settings = perClassSettings[label] ?? (224, 224)
                        perClass[label] = PredictionPipeline.PassConfig(
                            patchSizeLevel: max(32, settings.patch),
                            strideLevel: max(16, settings.stride),
                            level: Int32(levelIndex)
                        )
                    }
                    onRun(
                        PredictionPipeline.MultiPassConfig(perClass: perClass),
                        Float(minConfidence)
                    )
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(enabledClasses.isEmpty || !extractorAvailable)
            }
        }
        .padding(20)
        .frame(minWidth: 540, idealHeight: 540)
    }

    private var classifierIdentity: ExtractorIdentity {
        classifier.resolvedExtractorIdentity
    }

    private var extractorAvailable: Bool {
        // Vision is always considered available — the registry will pick the
        // current revision if the stored one is unsupported (rare).
        if classifierIdentity.kind == "vision" { return true }
        return ExtractorRegistry.shared.available.contains {
            $0.identity == classifierIdentity
        }
    }

    @ViewBuilder
    private var classTable: some View {
        VStack(spacing: 2) {
            HStack {
                Text("On")
                    .frame(width: 24, alignment: .center)
                Text("Class")
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text("Patch")
                    .frame(width: 64)
                Text("Stride")
                    .frame(width: 64)
                Text("Pass")
                    .frame(width: 30, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 4)

            // Color-code rows by their pass tuple so the user can quickly see
            // which classes share work.
            let passIndex = passIndexByClass
            ForEach(classifier.classLabels, id: \.self) { label in
                let isEnabled = enabledClasses.contains(label)
                HStack {
                    Toggle("", isOn: Binding(
                        get: { enabledClasses.contains(label) },
                        set: { on in
                            if on { enabledClasses.insert(label) }
                            else  { enabledClasses.remove(label) }
                        }
                    ))
                    .toggleStyle(.checkbox)
                    .labelsHidden()
                    .frame(width: 24, alignment: .center)

                    Circle()
                        .fill(isEnabled ? passColor(passIndex[label] ?? 0) : Color.secondary.opacity(0.3))
                        .frame(width: 8, height: 8)
                    Text(label)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundStyle(isEnabled ? .primary : .tertiary)
                    TextField("", value: Binding(
                        get: { perClassSettings[label]?.patch ?? 224 },
                        set: { perClassSettings[label] = (max(32, $0), perClassSettings[label]?.stride ?? 224) }
                    ), format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 64)
                    .disabled(!isEnabled)
                    TextField("", value: Binding(
                        get: { perClassSettings[label]?.stride ?? 224 },
                        set: { perClassSettings[label] = (perClassSettings[label]?.patch ?? 224, max(16, $0)) }
                    ), format: .number)
                    .textFieldStyle(.roundedBorder).frame(width: 64)
                    .disabled(!isEnabled)
                    Text(isEnabled ? "\((passIndex[label] ?? 0) + 1)" : "—")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 30, alignment: .trailing)
                }
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
            }
        }
        .padding(.vertical, 4)
        .background(Color(nsColor: .separatorColor).opacity(0.08), in: RoundedRectangle(cornerRadius: 4))
    }

    private var allClassesEnabled: Bool {
        enabledClasses.count == classifier.classLabels.count
    }

    private var passIndexByClass: [String: Int] {
        // Assign each unique (patch, stride) a pass index in stable order.
        // Only enabled classes participate in pass numbering.
        var index: [String: Int] = [:]
        var tupleToIndex: [String: Int] = [:]
        var next = 0
        for label in classifier.classLabels where enabledClasses.contains(label) {
            let s = perClassSettings[label] ?? (224, 224)
            let key = "\(s.patch)x\(s.stride)"
            if let i = tupleToIndex[key] {
                index[label] = i
            } else {
                tupleToIndex[key] = next
                index[label] = next
                next += 1
            }
        }
        return index
    }

    private func passColor(_ i: Int) -> Color {
        let palette: [Color] = [.blue, .orange, .green, .pink, .purple, .teal, .indigo, .yellow]
        return palette[i % palette.count]
    }

    private var estimate: String {
        if enabledClasses.isEmpty {
            return "No classes selected — enable at least one in the table above."
        }
        let lds = slide.levelDownsamples[max(0, min(levelIndex, slide.levelDownsamples.count - 1))]
        // Sum unique passes' estimates (classes that share a config share work).
        var seen: Set<String> = []
        var total = 0
        for label in classifier.classLabels where enabledClasses.contains(label) {
            let s = perClassSettings[label] ?? (224, 224)
            let key = "\(s.patch)x\(s.stride)"
            if seen.contains(key) { continue }
            seen.insert(key)
            total += PatchSampler.estimatedPatchCount(
                polygonOverlayY: annotation.points,
                patchSizeLevel: s.patch,
                strideLevel: s.stride,
                levelDownsample: lds
            )
        }
        let passCount = seen.count
        return "≈ \(total) patches across \(passCount) pass\(passCount == 1 ? "" : "es") for \(enabledClasses.count) of \(classifier.classLabels.count) class\(classifier.classLabels.count == 1 ? "" : "es") (upper bound)."
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
