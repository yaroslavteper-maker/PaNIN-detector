import SwiftUI
import SwiftData

struct AnnotationSidebar: View {
    @Bindable var store: AnnotationStore
    @Bindable var predictedStore: AnnotationStore
    @Bindable var profileStore: ProfileStore
    @Bindable var bankStore: MLBankStore
    @Bindable var classifierStore: MLClassifierStore
    @Bindable var predictionStore: PredictionStore
    @Bindable var geometryStore: GeometryFeatureStore
    @Bindable var renderSettings: RenderSettings
    @Environment(\.modelContext) private var modelContext
    @State private var selectedTab: SidebarTab = .annotations

    enum SidebarTab: String, CaseIterable, Identifiable {
        case annotations, predicted, classes, bank, geometry, model
        var id: String { rawValue }
        var label: String {
            switch self {
            case .annotations: return "Annotations"
            case .predicted:   return "Predicted"
            case .classes:     return "Classes"
            case .bank:        return "Bank"
            case .geometry:    return "Geometry"
            case .model:       return "Classifier"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $selectedTab) {
                ForEach(SidebarTab.allCases) { t in
                    Text(t.label).tag(t)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 8)
            .padding(.top, 8)
            .padding(.bottom, 4)

            Group {
                switch selectedTab {
                case .annotations:
                    AnnotationsTableView(
                        store: store,
                        profileStore: profileStore,
                        renderSettings: renderSettings
                    )
                case .predicted:
                    AnnotationsTableView(
                        store: predictedStore,
                        profileStore: profileStore,
                        renderSettings: renderSettings,
                        emptyMessage: "No predicted regions yet",
                        emptyHint: "Run a per-patch prediction, then Analysis ▸ Propose & Grade PanIN Regions."
                    )
                case .classes:
                    ClassesListView(profileStore: profileStore, annotationStore: store)
                case .bank:
                    MLBankPanel(bankStore: bankStore, modelContext: modelContext)
                case .geometry:
                    GeometryBankPanel(store: geometryStore)
                case .model:
                    MLModelPanel(
                        classifierStore: classifierStore,
                        bankStore: bankStore,
                        predictionStore: predictionStore
                    )
                }
            }
        }
    }
}

private struct AnnotationsTableView: View {
    @Bindable var store: AnnotationStore
    @Bindable var profileStore: ProfileStore
    @Bindable var renderSettings: RenderSettings
    var emptyMessage: String = "No annotations yet"
    var emptyHint: String = "Select the Lasso tool, then drag on the slide."
    @State private var editing: Annotation?
    @State private var showingDeleteChecked = false
    @State private var sortOrder: [KeyPathComparator<Annotation>] =
        [KeyPathComparator(\Annotation.classification, order: .forward)]

    private var sortedAnnotations: [Annotation] {
        store.annotations.sorted(using: sortOrder)
    }

    var body: some View {
        VStack(spacing: 0) {
            if store.annotations.isEmpty {
                emptyState
            } else {
                table
            }
            Divider()
            styleSection
            Divider()
            footer
        }
        .sheet(item: $editing) { ann in
            AnnotationEditorSheet(
                annotation: ann,
                existingLabels: store.existingLabels,
                profileStore: profileStore,
                onCancel: { editing = nil },
                onSave: { updated in
                    store.update(updated)
                    editing = nil
                }
            )
        }
        .confirmationDialog(
            "Delete \(store.checkedCount) checked annotation\(store.checkedCount == 1 ? "" : "s")?",
            isPresented: $showingDeleteChecked,
            titleVisibility: .visible
        ) {
            Button("Delete \(store.checkedCount)", role: .destructive) { store.removeChecked() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Removes every annotation whose checkbox is checked (visible). Uncheck the ones you want to keep first.")
        }
    }

    private var styleSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "paintbrush")
                    .foregroundStyle(.secondary)
                Text("Outline")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(
                    value: Binding(
                        get: { Double(renderSettings.strokeThickness) },
                        set: { renderSettings.strokeThickness = Float($0) }
                    ),
                    in: 1...150
                )
                .controlSize(.mini)
                Text("\(Int(renderSettings.strokeThickness))pt")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 40, alignment: .trailing)
                ColorPicker(
                    "",
                    selection: Binding(
                        get: {
                            Color(
                                red: Double(renderSettings.strokeColor.r) / 255,
                                green: Double(renderSettings.strokeColor.g) / 255,
                                blue: Double(renderSettings.strokeColor.b) / 255
                            )
                        },
                        set: { newColor in
                            let ns = NSColor(newColor).usingColorSpace(.sRGB) ?? .black
                            renderSettings.strokeColor = AnnotationColor(
                                r: Int(round(ns.redComponent * 255)),
                                g: Int(round(ns.greenComponent * 255)),
                                b: Int(round(ns.blueComponent * 255))
                            )
                        }
                    ),
                    supportsOpacity: false
                )
                .labelsHidden()
                .frame(width: 34)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "lasso")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text(emptyMessage)
                .foregroundStyle(.secondary)
            Text(emptyHint)
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var table: some View {
        Table(sortedAnnotations,
              selection: $store.selectedID,
              sortOrder: $sortOrder) {
            TableColumn("") { ann in
                Toggle("", isOn: Binding(
                    get: { ann.isVisible },
                    set: { store.setVisibility(ann.id, visible: $0) }
                ))
                .toggleStyle(.checkbox)
                .labelsHidden()
            }
            .width(22)

            TableColumn("Class", value: \.classification) { ann in
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color(red: Double(ann.color.r)/255,
                                    green: Double(ann.color.g)/255,
                                    blue: Double(ann.color.b)/255))
                        .frame(width: 10, height: 10)
                    Text(ann.classification)
                        .lineLimit(1)
                }
            }

            TableColumn("Name") { ann in
                if let name = ann.name, !name.isEmpty {
                    Text(name).lineLimit(1)
                } else {
                    Text("—").foregroundStyle(.tertiary)
                }
            }

            TableColumn("Pts") { ann in
                Text("\(ann.points.count)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .width(44)

            TableColumn("Area (px²)") { ann in
                let bbox = ann.boundingBoxInSlidePixels
                Text(ann.areaShort)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .help("Polygon area: \(Int(ann.areaInSlidePixels)) px²\nBounding box: \(Int(bbox.width)) × \(Int(bbox.height)) px")
            }
            .width(72)
        }
        .contextMenu(forSelectionType: UUID.self) { ids in
            if let id = ids.first ?? store.selectedID,
               let ann = store.annotations.first(where: { $0.id == id }) {
                Button("Edit…") { editing = ann }
                Button(ann.isVisible ? "Hide" : "Show") {
                    store.setVisibility(id, visible: !ann.isVisible)
                }
                Divider()
                Button("Delete", role: .destructive) { store.remove(id: id) }
            }
        } primaryAction: { ids in
            if let id = ids.first, let ann = store.annotations.first(where: { $0.id == id }) {
                editing = ann
            }
        }
        .onDeleteCommand {
            if let id = store.selectedID { store.remove(id: id) }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text(footerText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if let url = store.sidecarURL {
                    Image(systemName: "externaldrive")
                        .foregroundStyle(.secondary)
                        .font(.caption2)
                        .help("Saved to: \(url.path)")
                }
                Button(role: .destructive) {
                    showingDeleteChecked = true
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .disabled(store.checkedCount == 0)
                .help("Delete all checked (visible) annotations")
            }
            if let selectedAnnotation {
                let bbox = selectedAnnotation.boundingBoxInSlidePixels
                Text("Selected: \(Int(selectedAnnotation.areaInSlidePixels)) px²  •  bbox \(Int(bbox.width))×\(Int(bbox.height))")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var selectedAnnotation: Annotation? {
        guard let id = store.selectedID else { return nil }
        return store.annotations.first { $0.id == id }
    }

    private var footerText: String {
        let n = store.annotations.count
        return "\(n) annotation\(n == 1 ? "" : "s")"
    }
}
