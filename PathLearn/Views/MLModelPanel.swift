import SwiftUI

struct MLModelPanel: View {
    @Bindable var classifierStore: MLClassifierStore
    @Bindable var bankStore: MLBankStore
    @Bindable var predictionStore: PredictionStore

    var body: some View {
        VStack(spacing: 0) {
            if classifierStore.isTraining {
                trainingProgress
            } else if classifierStore.slotsFilled > 0 {
                kindPicker
                if let classifier = classifierStore.classifier {
                    summary(classifier)
                }
            } else {
                emptyState
            }
            Divider()
            predictionSection
            Divider()
            footer
        }
    }

    /// Segmented control between the two classifier slots. Only renders when
    /// both slots are populated; when only one is filled, there's no choice
    /// to make and the picker stays hidden.
    @ViewBuilder
    private var kindPicker: some View {
        let hasLogistic = classifierStore.logisticClassifier != nil
        let hasCentroid = classifierStore.centroidClassifier != nil
        if hasLogistic && hasCentroid {
            HStack(spacing: 8) {
                Text("Use:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker("", selection: Binding(
                    get: { classifierStore.activeKind },
                    set: { classifierStore.setActive($0) }
                )) {
                    Text(label(for: .logistic, classifierStore.logisticClassifier))
                        .tag(MLClassifier.Kind.logistic)
                    Text(label(for: .centroid, classifierStore.centroidClassifier))
                        .tag(MLClassifier.Kind.centroid)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            .padding(.horizontal, 12)
            .padding(.top, 10)
        } else if hasLogistic {
            badge("Logistic — only this slot populated. Promote centroids in the t-SNE window to fill the other.")
        } else if hasCentroid {
            badge("Centroid — only this slot populated. Run Analysis ▸ Regression Classifier… to fill the other.")
        }
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .foregroundStyle(.tertiary)
            .padding(.horizontal, 12)
            .padding(.top, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func label(for kind: MLClassifier.Kind, _ c: MLClassifier?) -> String {
        let valPct = c.map { Int(($0.metrics.valAccuracy * 100).rounded()) }
        switch (kind, valPct) {
        case (.logistic, let v?):  return "Logistic (\(v)%)"
        case (.centroid, let v?):  return "Centroid (\(v)%)"
        case (.logistic, nil):     return "Logistic"
        case (.centroid, nil):     return "Centroid"
        }
    }

    private var predictionSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "eyedropper.halffull")
                    .foregroundStyle(.secondary)
                Text(predictionHeadline)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !predictionStore.predictions.isEmpty {
                    Toggle("", isOn: Binding(
                        get: { predictionStore.isVisible },
                        set: { predictionStore.isVisible = $0 }
                    ))
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .labelsHidden()
                    .help("Show / hide heatmap on canvas")
                    Button {
                        NotificationCenter.default.post(name: .savePredictionsAsAnnotationsRequested, object: nil)
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .buttonStyle(.plain)
                    .help("Save displayed predictions as annotations")
                    Button {
                        predictionStore.reset()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.plain)
                    .help("Clear predictions")
                }
            }
            if !passInfoLines.isEmpty {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(passInfoLines, id: \.self) { line in
                        Text(line)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                            .lineLimit(2)
                            .truncationMode(.tail)
                    }
                }
            }
            if !predictionStore.predictions.isEmpty {
                HStack(spacing: 6) {
                    Text("Min confidence")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Slider(
                        value: Binding(
                            get: { Double(predictionStore.minProbability) },
                            set: { predictionStore.minProbability = Float($0) }
                        ),
                        in: 0.0...0.99,
                        step: 0.01,
                        onEditingChanged: { editing in
                            if !editing { predictionStore.commitEdits() }
                        }
                    )
                    .controlSize(.mini)
                    Text("\(Int((Double(predictionStore.minProbability) * 100).rounded()))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 32, alignment: .trailing)
                }
                let visibleCount = predictionStore.predictions
                    .filter { $0.maxProbability >= predictionStore.minProbability }
                    .count
                let totalCount = predictionStore.predictions.count
                if predictionStore.minProbability > 0 {
                    Text("Showing \(visibleCount) of \(totalCount) patches")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                HStack(spacing: 6) {
                    Text("Show on slide")
                        .font(.caption2.bold())
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("All") { predictionStore.showAllClasses() }
                        .buttonStyle(.plain)
                        .font(.caption2)
                        .foregroundStyle(.tint)
                        .disabled(predictionStore.hiddenClasses.isEmpty)
                    Text("·").font(.caption2).foregroundStyle(.tertiary)
                    Button("None") { predictionStore.hideAllClasses() }
                        .buttonStyle(.plain)
                        .font(.caption2)
                        .foregroundStyle(.tint)
                        .disabled(allClassesHidden)
                }
                .padding(.top, 2)
                ForEach(filteredPerClass, id: \.key) { entry in
                    HStack(spacing: 6) {
                        Toggle("", isOn: classVisibleBinding(for: entry.key))
                            .toggleStyle(.checkbox)
                            .labelsHidden()
                            .controlSize(.mini)
                            .help("Show “\(entry.key)” on the slide")
                        ColorPicker(
                            "",
                            selection: classColorBinding(for: entry.key),
                            supportsOpacity: false
                        )
                        .labelsHidden()
                        .controlSize(.mini)
                        .frame(width: 18, height: 18)
                        Text(entry.key)
                            .font(.caption2)
                            .foregroundStyle(predictionStore.isClassVisible(entry.key) ? .primary : .secondary)
                        Spacer()
                        Text(entry.value.label)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if predictionStore.isPredicting {
                ProgressView(value: predictionStore.progressFraction)
                Text("\(predictionStore.progressCurrent) / \(predictionStore.progressTotal)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var predictionHeadline: String {
        if predictionStore.isPredicting {
            return "Running…"
        }
        if predictionStore.predictions.isEmpty {
            return "No prediction yet — Analysis ▸ Classify on Annotation (⌥⌘R)"
        }
        let n = predictionStore.predictions.count
        let src = predictionStore.sourceAnnotationClass ?? "annotation"
        return "\(n) patches predicted on “\(src)”"
    }

    private var passInfoLines: [String] {
        predictionStore.passes.map { p in
            let ds = String(format: "%.1f", p.levelDownsample)
            let classList = p.classes.joined(separator: ", ")
            return "patch \(p.patchSizeLevel)px · stride \(p.strideLevel)px · L\(p.level) ×\(ds) — \(classList)"
        }
    }

    /// True when every predicted class is currently hidden.
    private var allClassesHidden: Bool {
        let labels = Set(predictionStore.predictions.map(\.predictedLabel))
        return !labels.isEmpty && labels.isSubset(of: predictionStore.hiddenClasses)
    }

    private func classVisibleBinding(for label: String) -> Binding<Bool> {
        Binding(
            get: { predictionStore.isClassVisible(label) },
            set: { predictionStore.setClassVisible(label, $0) }
        )
    }

    private func classColorBinding(for label: String) -> Binding<Color> {
        Binding(
            get: {
                let ac = predictionStore.classColors[label] ?? .defaultColor
                return Color(
                    red: Double(ac.r) / 255,
                    green: Double(ac.g) / 255,
                    blue: Double(ac.b) / 255
                )
            },
            set: { newColor in
                let ns = NSColor(newColor).usingColorSpace(.sRGB) ?? .red
                predictionStore.setClassColor(label, AnnotationColor(
                    r: Int(round(ns.redComponent * 255)),
                    g: Int(round(ns.greenComponent * 255)),
                    b: Int(round(ns.blueComponent * 255))
                ))
            }
        )
    }

    private func swiftUIColor(_ ac: AnnotationColor?) -> Color {
        guard let ac else { return .gray }
        return Color(red: Double(ac.r)/255, green: Double(ac.g)/255, blue: Double(ac.b)/255)
    }

    /// Per-class counts: visible / total when a threshold is in effect,
    /// otherwise just the total.
    private var filteredPerClass: [(key: String, value: (label: String, visible: Int))] {
        let thr = predictionStore.minProbability
        var bothCounts: [String: (visible: Int, total: Int)] = [:]
        for p in predictionStore.predictions {
            var entry = bothCounts[p.predictedLabel] ?? (0, 0)
            entry.total += 1
            if p.maxProbability >= thr { entry.visible += 1 }
            bothCounts[p.predictedLabel] = entry
        }
        return bothCounts
            .sorted { $0.key < $1.key }
            .map { entry in
                let label = thr > 0
                    ? "\(entry.value.visible) / \(entry.value.total)"
                    : "\(entry.value.total)"
                return (key: entry.key, value: (label: label, visible: entry.value.visible))
            }
    }

    // MARK: Subviews

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "brain")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("No classifier loaded")
                .foregroundStyle(.secondary)
            Text("Train one via Analysis ▸ Regression Classifier… (⌥⌘T) once the bank has ≥ 2 classes with ≥ 2 patches each, or run Analysis ▸ t-SNE Classifier… and promote its centroids.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var trainingProgress: some View {
        VStack(spacing: 12) {
            ProgressView(value: Double(classifierStore.trainingProgress))
                .frame(width: 220)
            Text(classifierStore.trainingStatus.isEmpty ? "Training…" : classifierStore.trainingStatus)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func summary(_ c: MLClassifier) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header(c)
                Divider()
                accuracySection(c.metrics)
                Divider()
                perClassSection(c.metrics)
                if c.metrics.classLabels.count <= 8 {
                    Divider()
                    confusionSection(c.metrics)
                }
            }
            .padding(12)
        }
    }

    private func header(_ c: MLClassifier) -> some View {
        let id = c.resolvedExtractorIdentity
        let available = id.kind == "vision" ||
            ExtractorRegistry.shared.available.contains(where: { $0.identity == id })
        let kindIcon = c.classifierKind == .centroid ? "scope" : "brain"
        let kindTitle = "\(c.classifierKind.displayName) classifier"
        return VStack(alignment: .leading, spacing: 2) {
            Label(kindTitle, systemImage: kindIcon)
                .font(.headline)
            Text("\(c.classCount) classes • \(c.metrics.trainCount + c.metrics.valCount) patches • \(c.featureDim)-dim features")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button {
                NotificationCenter.default.post(name: .showExtractorsRequested, object: nil)
            } label: {
                Label("Extractor: \(id.stringForm)", systemImage: "cpu")
                    .font(.caption2)
                    .foregroundStyle(available ? Color.secondary : Color.red)
            }
            .buttonStyle(.plain)
            .help(available ? "Open extractors panel" : "Required extractor is not installed — click to manage")
            Text(c.createdAt.formatted(date: .numeric, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.tertiary)
            if c.classifierKind == .centroid {
                Button {
                    NotificationCenter.default.post(name: .showTSNEPlotRequested, object: nil)
                } label: {
                    Label("View t-SNE plot", systemImage: "chart.dots.scatter")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .padding(.top, 4)
            }
        }
    }

    private func accuracySection(_ m: TrainingMetrics) -> some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Train").font(.caption).foregroundStyle(.secondary)
                Text("\(Int((m.trainAccuracy * 100).rounded()))%")
                    .font(.title2.monospacedDigit())
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Val").font(.caption).foregroundStyle(.secondary)
                Text("\(Int((m.valAccuracy * 100).rounded()))%")
                    .font(.title2.monospacedDigit())
                    .foregroundStyle(accuracyColor(m.valAccuracy))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Loss").font(.caption).foregroundStyle(.secondary)
                Text(String(format: "%.3f", m.finalLoss))
                    .font(.caption.monospacedDigit())
            }
            Spacer()
        }
    }

    private func accuracyColor(_ v: Float) -> Color {
        if v >= 0.8 { return .green }
        if v >= 0.6 { return .orange }
        return .red
    }

    private func perClassSection(_ m: TrainingMetrics) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Per-class (val)").font(.caption.bold()).foregroundStyle(.secondary)
            ForEach(m.classLabels, id: \.self) { label in
                let p = m.perClassPrecision[label] ?? 0
                let r = m.perClassRecall[label] ?? 0
                let f = m.perClassF1[label] ?? 0
                HStack {
                    Text(label).font(.caption).lineLimit(1)
                    Spacer()
                    Text("P \(Int((p * 100).rounded()))%  R \(Int((r * 100).rounded()))%  F1 \(Int((f * 100).rounded()))%")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func confusionSection(_ m: TrainingMetrics) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Confusion (val)").font(.caption.bold()).foregroundStyle(.secondary)
            Text("Rows = true, columns = predicted").font(.caption2).foregroundStyle(.tertiary)
            VStack(spacing: 1) {
                HStack(spacing: 1) {
                    Text("").frame(width: 64, height: 18)
                    ForEach(Array(m.classLabels.enumerated()), id: \.offset) { _, label in
                        Text(label)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 44, height: 18)
                            .lineLimit(1)
                    }
                }
                ForEach(Array(m.classLabels.enumerated()), id: \.offset) { i, label in
                    HStack(spacing: 1) {
                        Text(label)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .frame(width: 64, height: 22, alignment: .trailing)
                            .lineLimit(1)
                        ForEach(Array(m.classLabels.enumerated()), id: \.offset) { j, _ in
                            let v = m.confusionMatrix[i][j]
                            Text("\(v)")
                                .font(.caption2.monospacedDigit())
                                .frame(width: 44, height: 22)
                                .background(i == j ? Color.green.opacity(0.22) : Color.gray.opacity(0.06))
                        }
                    }
                }
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Image(systemName: "brain")
                .foregroundStyle(.secondary)
            if let c = classifierStore.classifier {
                Text("Val \(Int((c.metrics.valAccuracy * 100).rounded()))% • \(c.classCount) classes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("No model")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                NotificationCenter.default.post(name: .showExtractorsRequested, object: nil)
            } label: {
                Image(systemName: "cpu")
            }
            .buttonStyle(.plain)
            .help("Manage feature extractors…")

            Button {
                NotificationCenter.default.post(name: .trainModelRequested, object: nil)
            } label: {
                Image(systemName: "play.circle")
            }
            .buttonStyle(.plain)
            .disabled(classifierStore.isTraining || bankStore.totalPatches < 4)
            .help("Train Regression Classifier (⌥⌘T)")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }
}
