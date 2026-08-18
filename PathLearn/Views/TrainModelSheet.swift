import SwiftUI

struct TrainModelSheet: View {
    let bankSummary: BankSummary
    /// True when the active profile has at least one class flagged "Exclude /
    /// null" — enables the null-exclusion control.
    var hasNullClasses: Bool = false
    let onCancel: () -> Void
    let onTrain: (ClassifierTrainer.Config) -> Void

    struct BankSummary {
        let total: Int
        let perClass: [String: Int]
        /// Patch counts grouped by extractor identity string. Empty for
        /// callers that haven't been updated yet — UI treats that as
        /// "unknown" rather than mixed.
        let perExtractor: [String: Int]
        /// Nested counts: extractor identity → class → patch count. Drives
        /// per-extractor summaries when the bank holds patches from more
        /// than one extractor.
        let perExtractorPerClass: [String: [String: Int]]

        init(total: Int,
             perClass: [String: Int],
             perExtractor: [String: Int] = [:],
             perExtractorPerClass: [String: [String: Int]] = [:]) {
            self.total = total
            self.perClass = perClass
            self.perExtractor = perExtractor
            self.perExtractorPerClass = perExtractorPerClass
        }
    }

    @State private var iterations: Int = 200
    @State private var learningRate: Double = 0.1
    @State private var l2: Double = 0.001
    @State private var valFraction: Double = 0.2
    @State private var skipWhitePatches: Bool = false
    @State private var whiteThreshold: Double = 0.75
    @State private var poolPerAnnotation: Bool = false
    @State private var excludeNull: Bool = false
    @State private var nullThreshold: Double = 0.85
    @State private var enabledClasses: Set<String>
    /// Which extractor's patches to train on. Empty string means the bank
    /// holds patches from only one extractor (or none) — no filter applied.
    @State private var selectedExtractorKey: String

    init(bankSummary: BankSummary,
         hasNullClasses: Bool = false,
         onCancel: @escaping () -> Void,
         onTrain: @escaping (ClassifierTrainer.Config) -> Void) {
        self.bankSummary = bankSummary
        self.hasNullClasses = hasNullClasses
        self.onCancel = onCancel
        self.onTrain = onTrain
        // Default to the most-populated extractor when the bank is mixed.
        let initialKey = bankSummary.perExtractor.count > 1
            ? bankSummary.perExtractor.max(by: { $0.value < $1.value })?.key ?? ""
            : ""
        _selectedExtractorKey = State(initialValue: initialKey)
        // Enable every class that has patches under the chosen extractor (or
        // all classes when no filter is active).
        let initialClasses: [String: Int] = {
            if !initialKey.isEmpty,
               let perClass = bankSummary.perExtractorPerClass[initialKey] {
                return perClass
            }
            return bankSummary.perClass
        }()
        _enabledClasses = State(initialValue: Set(initialClasses.keys))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Train classifier").font(.headline)

            summary

            Divider()

            HStack {
                Text("Iterations:").frame(width: 110, alignment: .trailing)
                TextField("", value: $iterations, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                Stepper("", value: $iterations, in: 10...10000, step: 50).labelsHidden()
            }

            HStack {
                Text("Learning rate:").frame(width: 110, alignment: .trailing)
                TextField("", value: $learningRate, format: .number.precision(.fractionLength(4)))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }

            HStack {
                Text("L2 weight:").frame(width: 110, alignment: .trailing)
                TextField("", value: $l2, format: .number.precision(.fractionLength(4)))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }

            HStack {
                Text("Validation:").frame(width: 110, alignment: .trailing)
                Slider(value: $valFraction, in: 0.05...0.5)
                Text("\(Int(valFraction * 100))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 40, alignment: .trailing)
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
                Text("Patches without a stored whiteness score (extracted before this filter existed) always pass — re-extract them to score them.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Toggle("Classify whole annotation (pooled)", isOn: $poolPerAnnotation)
                    .toggleStyle(.checkbox)
                    .font(.caption)
                Text("Pools each annotation's patches (mean/max/std) into one example and makes a single per-annotation decision — better for lesion-level calls like PanIN grade. Needs ≥2 annotations per class.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if hasNullClasses {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Toggle("Exclude patches similar to null class at", isOn: $excludeNull)
                            .toggleStyle(.checkbox)
                            .font(.caption)
                        Slider(value: $nullThreshold, in: 0.5...0.99, step: 0.01)
                            .controlSize(.mini)
                            .disabled(!excludeNull)
                        Text("\(Int((nullThreshold * 100).rounded()))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(width: 38, alignment: .trailing)
                    }
                    Text("Patches whose cosine similarity to a null-class (e.g. lumen) patch is at/above this cutoff are dropped from training, and the null reference is saved on the model to filter predictions too. Null-class patches are never trained as a real class regardless.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            if extractorMixed {
                HStack(spacing: 6) {
                    Image(systemName: "cpu")
                        .foregroundStyle(.secondary)
                    Text("Train on:")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("", selection: $selectedExtractorKey) {
                        ForEach(sortedExtractorKeys, id: \.self) { key in
                            let count = bankSummary.perExtractor[key] ?? 0
                            Text("\(key) (\(count))").tag(key)
                        }
                    }
                    .labelsHidden()
                    .onChange(of: selectedExtractorKey) { _, _ in
                        // Reset to all classes available under the new filter.
                        enabledClasses = Set(activeClasses.keys)
                    }
                }
            } else if let extractorLine {
                Label(extractorLine, systemImage: "cpu")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if isBinarizeMode {
                Label(binarizeNote, systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.blue)
            } else if !canTrain {
                Label(disabledReason, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Train") {
                    let cutoff: Float? = skipWhitePatches ? Float(whiteThreshold) : nil
                    let classFilter: Set<String>? =
                        allClassesEnabled ? nil : enabledClasses
                    onTrain(ClassifierTrainer.Config(
                        iterations: iterations,
                        learningRate: Float(learningRate),
                        l2: Float(l2),
                        valFraction: Float(valFraction),
                        maxWhiteFraction: cutoff,
                        enabledClasses: classFilter,
                        binarizeAgainstRest: isBinarizeMode,
                        extractorIdentity: trainExtractorIdentity,
                        poolPerAnnotation: poolPerAnnotation,
                        nullThreshold: (hasNullClasses && excludeNull) ? Float(nullThreshold) : nil
                    ))
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(!canTrain)
            }
        }
        .padding(20)
        .frame(minWidth: 480)
    }

    private var summary: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Bank contents")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Button(allClassesEnabled ? "Disable all" : "Enable all") {
                    if allClassesEnabled {
                        enabledClasses = []
                    } else {
                        enabledClasses = Set(activeClasses.keys)
                    }
                }
                .buttonStyle(.borderless)
                .font(.caption2)
            }
            Text(headerLine)
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(activeClasses.sorted(by: { $0.key < $1.key }), id: \.key) { entry in
                let isOn = enabledClasses.contains(entry.key)
                HStack(spacing: 6) {
                    Toggle("", isOn: Binding(
                        get: { enabledClasses.contains(entry.key) },
                        set: { on in
                            if on { enabledClasses.insert(entry.key) }
                            else  { enabledClasses.remove(entry.key) }
                        }
                    ))
                    .toggleStyle(.checkbox)
                    .labelsHidden()
                    Text(entry.key)
                        .foregroundStyle(isOn ? .primary : .tertiary)
                    Spacer()
                    Text("\(entry.value)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                .font(.caption)
            }
        }
    }

    /// Class counts under the currently selected extractor filter. When no
    /// extractor filter applies (bank has zero or one extractor), this is
    /// just `bankSummary.perClass`.
    private var activeClasses: [String: Int] {
        if extractorMixed,
           let filtered = bankSummary.perExtractorPerClass[selectedExtractorKey] {
            return filtered
        }
        return bankSummary.perClass
    }

    private var activeTotal: Int {
        if extractorMixed {
            return bankSummary.perExtractor[selectedExtractorKey] ?? 0
        }
        return bankSummary.total
    }

    private var allClassesEnabled: Bool {
        enabledClasses.count == activeClasses.count
    }

    private var selectedPatchCount: Int {
        activeClasses
            .filter { enabledClasses.contains($0.key) }
            .values
            .reduce(0, +)
    }

    private var selectedClassCount: Int {
        enabledClasses.intersection(activeClasses.keys).count
    }

    private var minPerSelectedClass: Int {
        let counts = activeClasses
            .filter { enabledClasses.contains($0.key) }
            .values
        return counts.min() ?? 0
    }

    private var headerLine: String {
        "\(selectedPatchCount) of \(activeTotal) patches in \(selectedClassCount) of \(activeClasses.count) classes selected"
    }

    private var sortedExtractorKeys: [String] {
        bankSummary.perExtractor.keys.sorted()
    }

    /// True when the user has selected exactly 1 class AND there are other
    /// classes in the bank to act as a synthetic "Background" negative class.
    private var isBinarizeMode: Bool {
        selectedClassCount == 1 && activeClasses.count >= 2
    }

    private var binarizeTargetClass: String {
        enabledClasses.first ?? ""
    }

    private var binarizeBackgroundCount: Int {
        activeClasses
            .filter { !enabledClasses.contains($0.key) }
            .values
            .reduce(0, +)
    }

    private var binarizeTargetCount: Int {
        activeClasses[binarizeTargetClass] ?? 0
    }

    private var binarizeNote: String {
        "Will train binary classifier: “\(binarizeTargetClass)” (\(binarizeTargetCount) pts) vs. “Background” (\(binarizeBackgroundCount) pts from other classes)."
    }

    private var canTrain: Bool {
        if isBinarizeMode {
            return binarizeTargetCount >= 2 && binarizeBackgroundCount >= 2
        }
        return selectedClassCount >= 2 &&
               selectedPatchCount >= 4 &&
               minPerSelectedClass >= 2
    }

    /// Identity string to pass into `ClassifierTrainer.Config.extractorIdentity`.
    /// `nil` when the bank has zero or one extractor (no filter needed).
    private var trainExtractorIdentity: String? {
        extractorMixed ? selectedExtractorKey : nil
    }

    private var extractorMixed: Bool {
        bankSummary.perExtractor.count > 1
    }

    private var extractorLine: String? {
        guard !bankSummary.perExtractor.isEmpty,
              let only = bankSummary.perExtractor.first,
              bankSummary.perExtractor.count == 1 else {
            return nil
        }
        return "Extractor: \(only.key)"
    }

    private var disabledReason: String {
        if selectedClassCount == 0 {
            return "Select at least 1 class. With 1 class, the trainer will use all other patches as a synthetic Background."
        }
        if isBinarizeMode {
            if binarizeTargetCount < 2 {
                return "“\(binarizeTargetClass)” needs at least 2 patches."
            }
            if binarizeBackgroundCount < 2 {
                return "Need at least 2 patches in non-target classes to act as Background."
            }
            return ""
        }
        if selectedClassCount < 2 {
            return "Select at least 2 classes (or 1 class with at least 2 other-class patches in the bank to use as Background)."
        }
        if selectedPatchCount < 4 {
            return "Selected classes have fewer than 4 patches total."
        }
        if minPerSelectedClass < 2 {
            return "Each selected class needs at least 2 patches."
        }
        return ""
    }
}
