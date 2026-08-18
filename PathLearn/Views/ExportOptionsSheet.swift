import SwiftUI

struct ExportOptionsSheet: View {
    let slide: SlideImage
    let totalAnnotationCount: Int
    let hasSelection: Bool
    let outputDirectory: URL
    let onCancel: () -> Void
    let onExport: (AnnotationExporter.Options, Scope) -> Void

    @State private var formatKind: FormatKind = .png
    @State private var jpegQuality: Double = 0.9
    @State private var level: Int = 0
    @State private var padding: Int = 0
    @State private var cropShape: AnnotationExporter.CropShape = .square
    @State private var scope: Scope

    init(slide: SlideImage,
         totalAnnotationCount: Int,
         hasSelection: Bool,
         outputDirectory: URL,
         onCancel: @escaping () -> Void,
         onExport: @escaping (AnnotationExporter.Options, Scope) -> Void) {
        self.slide = slide
        self.totalAnnotationCount = totalAnnotationCount
        self.hasSelection = hasSelection
        self.outputDirectory = outputDirectory
        self.onCancel = onCancel
        self.onExport = onExport
        _scope = State(initialValue: hasSelection ? .selected : .all)
    }

    enum FormatKind: String, CaseIterable, Identifiable {
        case png, jpeg
        var id: String { rawValue }
        var label: String { rawValue.uppercased() }
    }

    enum Scope: String, CaseIterable, Identifiable, Hashable {
        case all, selected
        var id: String { rawValue }
        var label: String {
            self == .all ? "All annotations" : "Selected annotation only"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export annotations").font(.headline)

            VStack(alignment: .leading, spacing: 2) {
                Text("Saving to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(outputDirectory.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Divider()

            HStack {
                Text("Scope:").frame(width: 80, alignment: .trailing)
                Picker("", selection: $scope) {
                    ForEach(Scope.allCases) { s in
                        Text(scopeLabel(s)).tag(s)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            HStack {
                Text("Shape:").frame(width: 80, alignment: .trailing)
                Picker("", selection: $cropShape) {
                    ForEach(AnnotationExporter.CropShape.allCases) { s in
                        Text(s.label).tag(s)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            HStack {
                Text("Format:").frame(width: 80, alignment: .trailing)
                Picker("", selection: $formatKind) {
                    ForEach(FormatKind.allCases) { k in
                        Text(k.label).tag(k)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }

            if formatKind == .jpeg {
                HStack {
                    Text("Quality:").frame(width: 80, alignment: .trailing)
                    Slider(value: $jpegQuality, in: 0.5...1.0)
                    Text("\(Int(jpegQuality * 100))%")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                        .frame(width: 42, alignment: .trailing)
                }
            }

            HStack {
                Text("Detail:").frame(width: 80, alignment: .trailing)
                Picker("", selection: $level) {
                    ForEach(0..<Int(slide.levelCount), id: \.self) { i in
                        let ds = slide.levelDownsamples[i]
                        Text(levelLabel(index: i, downsample: ds))
                            .tag(i)
                    }
                }
                .labelsHidden()
            }

            HStack {
                Text("Padding:").frame(width: 80, alignment: .trailing)
                TextField("0", value: $padding, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                Stepper("", value: $padding, in: 0...100_000, step: 50)
                    .labelsHidden()
                Text("slide px (each side)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(layoutHint)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.top, 2)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Export") {
                    let format: AnnotationExporter.Format =
                        formatKind == .png ? .png : .jpeg(quality: jpegQuality)
                    onExport(AnnotationExporter.Options(
                        format: format,
                        level: Int32(level),
                        paddingLevel0: max(0, padding),
                        cropShape: cropShape
                    ), scope)
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .disabled(exportCount == 0)
            }
        }
        .padding(20)
        .frame(minWidth: 480)
    }

    private var exportCount: Int {
        switch scope {
        case .all:      return totalAnnotationCount
        case .selected: return hasSelection ? 1 : 0
        }
    }

    private func scopeLabel(_ s: Scope) -> String {
        switch s {
        case .all:
            return "All (\(totalAnnotationCount))"
        case .selected:
            return hasSelection ? "Selected" : "Selected (none)"
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

    private var layoutHint: String {
        let n = exportCount
        return "Will export \(n) annotation\(n == 1 ? "" : "s")  •  \(outputDirectory.lastPathComponent)/<slide>/<class>/<annotation>.\(formatKind == .png ? "png" : "jpg")"
    }
}
