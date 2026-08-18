import SwiftUI

/// Sheet that configures a t-SNE run before kicking it off. Mirrors the train
/// sheet's pattern: extractor picker only appears when the bank is mixed,
/// counts adapt to the selected extractor.
struct TSNEConfigSheet: View {
    struct BankInfo {
        let total: Int
        let perExtractor: [String: Int]
        let perExtractorPerClass: [String: [String: Int]]
        let perClass: [String: Int]
    }

    struct RunRequest {
        let config: TSNE.Config
        let extractorIdentity: String?
        let maxCount: Int
        /// If set, patches whose pixel-whiteness exceeds this fraction are
        /// excluded before sampling — same semantics as the whiteness filter
        /// in the Train Regression Classifier sheet.
        let maxWhiteFraction: Float?
        /// If set, patches with a stored nucleus count below this are excluded.
        /// Patches with no measured count (legacy) pass.
        let minNuclei: Int?
    }

    let info: BankInfo
    let onCancel: () -> Void
    let onRun: (RunRequest) -> Void

    @State private var perplexity: Double = 30
    @State private var iterations: Int = 1000
    @State private var maxCount: Int = 5000
    @State private var skipWhitePatches: Bool = false
    @State private var whiteThreshold: Double = 0.75
    @State private var requireNuclei: Bool = false
    @State private var minNuclei: Int = 5
    @State private var selectedExtractorKey: String

    init(info: BankInfo,
         onCancel: @escaping () -> Void,
         onRun: @escaping (RunRequest) -> Void) {
        self.info = info
        self.onCancel = onCancel
        self.onRun = onRun
        let initialKey = info.perExtractor.count > 1
            ? info.perExtractor.max(by: { $0.value < $1.value })?.key ?? ""
            : ""
        _selectedExtractorKey = State(initialValue: initialKey)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("t-SNE analysis").font(.headline)

            if extractorMixed {
                HStack(spacing: 6) {
                    Image(systemName: "cpu").foregroundStyle(.secondary)
                    Text("Use patches from:")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("", selection: $selectedExtractorKey) {
                        ForEach(sortedExtractorKeys, id: \.self) { key in
                            let n = info.perExtractor[key] ?? 0
                            Text("\(key) (\(n))").tag(key)
                        }
                    }
                    .labelsHidden()
                }
            } else if let only = info.perExtractor.first {
                Label("Extractor: \(only.key)", systemImage: "cpu")
                    .font(.caption2).foregroundStyle(.tertiary)
            }

            Divider()

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                GridRow {
                    Text("Perplexity")
                    HStack {
                        Slider(value: $perplexity, in: 5...50, step: 1)
                        Text("\(Int(perplexity))")
                            .frame(width: 36, alignment: .trailing)
                            .monospacedDigit()
                    }
                }
                GridRow {
                    Text("Iterations")
                    HStack {
                        Slider(value: Binding(get: { Double(iterations) },
                                              set: { iterations = Int($0) }),
                               in: 250...2000, step: 50)
                        Text("\(iterations)")
                            .frame(width: 50, alignment: .trailing)
                            .monospacedDigit()
                    }
                }
                GridRow {
                    Text("Max patches")
                    HStack {
                        Slider(value: Binding(get: { Double(maxCount) },
                                              set: { maxCount = Int($0) }),
                               in: 200...10000, step: 100)
                        Text("\(maxCount)")
                            .frame(width: 50, alignment: .trailing)
                            .monospacedDigit()
                    }
                }
                GridRow {
                    Toggle("Skip white patches", isOn: $skipWhitePatches)
                    HStack {
                        Slider(value: $whiteThreshold, in: 0.30...0.95, step: 0.05)
                            .disabled(!skipWhitePatches)
                        Text("\(Int(whiteThreshold * 100))%")
                            .frame(width: 50, alignment: .trailing)
                            .monospacedDigit()
                            .foregroundStyle(skipWhitePatches ? .primary : .secondary)
                    }
                }
                GridRow {
                    Toggle("Require nuclei", isOn: $requireNuclei)
                    HStack {
                        Stepper(value: $minNuclei, in: 1...200) {
                            Text("≥ \(minNuclei) per patch")
                                .font(.callout.monospacedDigit())
                                .foregroundStyle(requireNuclei ? .primary : .secondary)
                        }
                        .disabled(!requireNuclei)
                    }
                }
            }

            Text(summaryLine)
                .font(.caption)
                .foregroundStyle(.secondary)
            if !canRun {
                Label(disabledReason, systemImage: "exclamationmark.triangle")
                    .font(.caption).foregroundStyle(.orange)
            }
            Text("Subsampling is stratified per class, so rare classes don't get drowned. O(N²) per iteration — keep `Max patches` low for fast feedback.")
                .font(.caption2)
                .foregroundStyle(.tertiary)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Run") {
                    let cutoff: Float? = skipWhitePatches ? Float(whiteThreshold) : nil
                    onRun(RunRequest(
                        config: TSNE.Config(
                            perplexity: perplexity,
                            iterations: iterations
                        ),
                        extractorIdentity: extractorMixed ? selectedExtractorKey : nil,
                        maxCount: maxCount,
                        maxWhiteFraction: cutoff,
                        minNuclei: requireNuclei ? max(1, minNuclei) : nil
                    ))
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(!canRun)
            }
        }
        .padding(20)
        .frame(minWidth: 460)
    }

    // MARK: - Derived

    private var extractorMixed: Bool { info.perExtractor.count > 1 }
    private var sortedExtractorKeys: [String] { info.perExtractor.keys.sorted() }

    private var availableForExtractor: Int {
        if extractorMixed { return info.perExtractor[selectedExtractorKey] ?? 0 }
        return info.total
    }

    private var willUse: Int { min(availableForExtractor, maxCount) }

    private var summaryLine: String {
        let cls = activeClasses.count
        return "Will embed \(willUse) of \(availableForExtractor) patches across \(cls) classes."
    }

    private var activeClasses: [String: Int] {
        if extractorMixed,
           let perClass = info.perExtractorPerClass[selectedExtractorKey] {
            return perClass
        }
        return info.perClass
    }

    private var canRun: Bool {
        availableForExtractor >= 10 && perplexity * 3 < Double(willUse)
    }

    private var disabledReason: String {
        if availableForExtractor < 10 {
            return "Need at least 10 patches under the selected extractor."
        }
        if perplexity * 3 >= Double(willUse) {
            return "Perplexity \(Int(perplexity)) is too high for \(willUse) points (must be < N/3)."
        }
        return ""
    }
}
