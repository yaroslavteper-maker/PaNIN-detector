import SwiftUI
import AppKit
import UniformTypeIdentifiers
import Observation
import SwiftData

extension Notification.Name {
    static let openSlideRequested              = Notification.Name("PaNINDetector.openSlideRequested")
    static let saveProfileRequested            = Notification.Name("PaNINDetector.saveProfile")
    static let openProfileRequested            = Notification.Name("PaNINDetector.openProfile")
    static let exportAnnotationsRequested      = Notification.Name("PaNINDetector.exportAnnotations")
    static let showAnnotationPreviewRequested  = Notification.Name("PaNINDetector.showAnnotationPreview")
    static let extractPatchesRequested         = Notification.Name("PaNINDetector.extractPatches")
    static let saveBankRequested               = Notification.Name("PaNINDetector.saveBank")
    static let openBankRequested               = Notification.Name("PaNINDetector.openBank")
    static let newBankRequested                = Notification.Name("PaNINDetector.newBank")
    static let revealBankRequested             = Notification.Name("PaNINDetector.revealBank")
    static let trainModelRequested             = Notification.Name("PaNINDetector.trainModel")
    static let saveModelRequested              = Notification.Name("PaNINDetector.saveModel")
    static let openModelRequested              = Notification.Name("PaNINDetector.openModel")
    static let predictAnnotationRequested      = Notification.Name("PaNINDetector.predictAnnotation")
    static let predictVISTARequested           = Notification.Name("PaNINDetector.predictVISTA")
    static let clearPredictionsRequested       = Notification.Name("PaNINDetector.clearPredictions")
    static let clearClassesAndAnnotationsRequested = Notification.Name("PaNINDetector.clearClassesAndAnnotations")
    static let showExtractorsRequested         = Notification.Name("PaNINDetector.showExtractors")
    static let closeSlideRequested             = Notification.Name("PaNINDetector.closeSlide")
    static let tsneAnalysisRequested           = Notification.Name("PaNINDetector.tsneAnalysis")
    static let promoteCentroidsRequested       = Notification.Name("PaNINDetector.promoteCentroids")
    static let showTSNEPlotRequested           = Notification.Name("PaNINDetector.showTSNEPlot")
    static let importAnnotationsRequested      = Notification.Name("PaNINDetector.importAnnotations")
    static let exportGeoJSONRequested          = Notification.Name("PaNINDetector.exportGeoJSON")
    static let showHelpRequested               = Notification.Name("PaNINDetector.showHelp")
    static let savePredictionsAsAnnotationsRequested = Notification.Name("PaNINDetector.savePredictionsAsAnnotations")
    static let togglePredictionHeatmapRequested = Notification.Name("PaNINDetector.togglePredictionHeatmap")
    static let extractGeometryRequested         = Notification.Name("PaNINDetector.extractGeometry")
    static let trainGeometryModelRequested      = Notification.Name("PaNINDetector.trainGeometryModel")
    static let revealGeometryBankRequested      = Notification.Name("PaNINDetector.revealGeometryBank")
    static let proposeAndGradeRegionsRequested  = Notification.Name("PaNINDetector.proposeAndGradeRegions")
}

private struct PendingPolygon: Identifiable {
    let id = UUID()
    let points: [CGPoint]
}

private struct ExportDirRequest: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ProposeRegionsRequest: Identifiable {
    let id = UUID()
    let labels: [String]
    let counts: [String: Int]
}

@Observable
final class ExportSession: @unchecked Sendable {
    var isExporting = false
    var current = 0
    var total = 0
    var report: AnnotationExporter.Report?

    var progressFraction: Double {
        total > 0 ? Double(current) / Double(total) : 0
    }
}

@Observable
final class ExtractionSession: @unchecked Sendable {
    var isExtracting = false
    var current = 0
    var total = 0
    var report: PatchExtractionPipeline.Report?

    var progressFraction: Double {
        total > 0 ? Double(current) / Double(total) : 0
    }
}

private struct ExtractRequest: Identifiable {
    let id = UUID()
}

private struct BankImportRequest: Identifiable {
    let id = UUID()
    let url: URL
}

private struct AnnotationImportRequest: Identifiable {
    let id = UUID()
    let url: URL
    let incoming: [Annotation]
}

private struct TrainModelRequest: Identifiable {
    let id = UUID()
}

private struct PredictRequest: Identifiable {
    let id = UUID()
    let annotation: Annotation
}

private struct ExtractorsRequest: Identifiable {
    let id = UUID()
}

private struct TSNERequest: Identifiable {
    let id = UUID()
}

struct ContentView: View {
    @State private var slide: SlideImage?
    @State private var store = AnnotationStore()
    /// Auto-proposed + geometry-graded PanIN regions, shown in their own tab.
    @State private var predictedStore = AnnotationStore()
    @State private var bankStore = MLBankStore()
    @State private var classifierStore = MLClassifierStore()
    @State private var predictionStore = PredictionStore()
    @State private var geometryStore = GeometryFeatureStore()
    @Environment(ProfileStore.self) private var profileStore
    @Environment(EmbeddingStore.self) private var embeddingStore
    @State private var renderSettings = RenderSettings()
    @State private var exportSession = ExportSession()
    @State private var extractionSession = ExtractionSession()
    @State private var tool: AnnotationTool = .pan
    @State private var pending: PendingPolygon?
    @State private var pendingExport: ExportDirRequest?
    @State private var pendingExtract: ExtractRequest?
    @State private var pendingBankImport: BankImportRequest?
    @State private var pendingAnnotationImport: AnnotationImportRequest?
    @State private var pendingTrainModel: TrainModelRequest?
    @State private var pendingPredict: PredictRequest?
    @State private var pendingProposeRegions: ProposeRegionsRequest?
    @State private var pendingExtractors: ExtractorsRequest?
    @State private var pendingTSNE: TSNERequest?
    @State private var tsneTask: Task<Void, Never>?
    @State private var showingNewBankConfirmation = false
    @State private var showingClearAllConfirmation = false
    /// Predicted regions staged for conversion into persistent annotations,
    /// awaiting the user's confirmation (the count can be large).
    @State private var pendingPredictionAnnotations: [Annotation]?
    @State private var errorMessage: String?
    /// Neutral, non-error feedback (e.g. geometry extract/train summaries).
    @State private var infoMessage: String?
    @State private var slideURL: URL?
    /// User-selected extractor identity for the next patch extraction. `nil`
    /// means the built-in Vision extractor at its current revision.
    @State private var selectedExtractorIdentity: ExtractorIdentity?

    @Environment(PreviewState.self) private var previewState
    @Environment(\.openWindow) private var openWindow
    @Environment(\.modelContext) private var modelContext

    /// The split-view shell (sidebar + slide canvas). Kept as its own property
    /// so the large `body` modifier chain type-checks in reasonable time.
    private var splitView: some View {
        NavigationSplitView {
            AnnotationSidebar(
                store: store,
                predictedStore: predictedStore,
                profileStore: profileStore,
                bankStore: bankStore,
                classifierStore: classifierStore,
                predictionStore: predictionStore,
                geometryStore: geometryStore,
                renderSettings: renderSettings
            )
            .navigationSplitViewColumnWidth(min: 280, ideal: 380)
        } detail: {
            detail
                .toolbar {
                    ToolbarItem(placement: .primaryAction) {
                        Button {
                            openPanel()
                        } label: {
                            Label("Open Slide", systemImage: "folder")
                        }
                        .help("Open a whole-slide image (⌘O)")
                    }
                }
        }
        .navigationTitle(slideURL?.lastPathComponent ?? "PathLearn")
        .onChange(of: store.selectedID) { _, newID in
            // Show the selected annotation's saved prediction module (or the
            // empty state if it was never classified).
            predictionStore.displayedID = newID
        }
    }

    var body: some View {
        bodyStageC(bodyStageB(bodyStageA(splitView)))
    }

    // The presentation modifier chain is split across stages so each sub-chain
    // type-checks in reasonable time (SwiftUI's inference cost grows sharply
    // with chain length).
    @ViewBuilder
    private func bodyStageA(_ base: some View) -> some View {
        base
        .sheet(item: $pending) { box in
            LabelPickerSheet(
                initialLabel: store.currentLabel,
                initialColor: store.currentColor,
                existingLabels: store.existingLabels,
                profileStore: profileStore,
                onCancel: { pending = nil },
                onCommit: { label, color in
                    store.currentLabel = label
                    store.currentColor = color
                    let ann = Annotation(points: box.points,
                                         classification: label,
                                         color: color)
                    store.add(ann)
                    pending = nil
                }
            )
        }
        .sheet(item: $pendingExport) { req in
            if let slide {
                ExportOptionsSheet(
                    slide: slide,
                    totalAnnotationCount: store.annotations.count,
                    hasSelection: store.selectedID != nil,
                    outputDirectory: req.url,
                    onCancel: { pendingExport = nil },
                    onExport: { opts, scope in
                        let dir = req.url
                        pendingExport = nil
                        let annotationsToExport: [Annotation]
                        switch scope {
                        case .all:
                            annotationsToExport = store.annotations
                        case .selected:
                            if let id = store.selectedID,
                               let ann = store.annotations.first(where: { $0.id == id }) {
                                annotationsToExport = [ann]
                            } else {
                                annotationsToExport = []
                            }
                        }
                        runExport(to: dir, annotations: annotationsToExport, with: opts)
                    }
                )
            }
        }
        .sheet(isPresented: exportingBinding) {
            ExportProgressView(session: exportSession)
        }
        .alert("Export complete",
               isPresented: exportDoneBinding,
               presenting: exportSession.report) { report in
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([report.outputDirectory])
                exportSession.report = nil
            }
            Button("OK") { exportSession.report = nil }
        } message: { report in
            let skipped = report.skippedCount
            let saved = report.savedCount
            let plural = saved == 1 ? "" : "s"
            if skipped > 0 {
                Text("Saved \(saved) annotation\(plural), skipped \(skipped).")
            } else {
                Text("Saved \(saved) annotation\(plural).")
            }
        }
        .alert("Could not open file",
               isPresented: Binding(get: { errorMessage != nil },
                                    set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
        .alert("PathLearn",
               isPresented: Binding(get: { infoMessage != nil },
                                    set: { if !$0 { infoMessage = nil } })) {
            Button("OK") { infoMessage = nil }
        } message: {
            Text(infoMessage ?? "")
        }
        .task {
            // Launch with no slide loaded — user picks one via File ▸ Open Slide… (⌘O).
            for await _ in NotificationCenter.default.notifications(named: .openSlideRequested) {
                openPanel()
            }
        }
        .task { await runNotificationHandlers() }
    }

    @ViewBuilder
    private func bodyStageB(_ base: some View) -> some View {
        base
        .sheet(item: $pendingPredict) { req in
            if let slide, let classifier = classifierStore.classifier {
                PredictAnnotationSheet(
                    slide: slide,
                    annotation: req.annotation,
                    classifier: classifier,
                    onCancel: { pendingPredict = nil },
                    onRun: { multiPass, minConf in
                        let ann = req.annotation
                        pendingPredict = nil
                        runPrediction(annotation: ann, multiPass: multiPass, minConfidence: minConf)
                    },
                    onManageExtractors: {
                        pendingPredict = nil
                        pendingExtractors = ExtractorsRequest()
                    }
                )
            }
        }
        .sheet(item: $pendingProposeRegions) { req in
            ProposeRegionsSheet(
                labels: req.labels,
                counts: req.counts,
                onCancel: { pendingProposeRegions = nil },
                onRun: { selected in
                    pendingProposeRegions = nil
                    proposeAndGradeRegions(candidateLabels: selected)
                }
            )
        }
        .sheet(item: $pendingTrainModel) { _ in
            TrainModelSheet(
                bankSummary: TrainModelSheet.BankSummary(
                    total: bankStore.totalPatches,
                    perClass: bankStore.perClass,
                    perExtractor: bankStore.perExtractor,
                    perExtractorPerClass: bankStore.perExtractorPerClass
                ),
                hasNullClasses: hasNullClasses,
                onCancel: { pendingTrainModel = nil },
                onTrain: { config in
                    pendingTrainModel = nil
                    runTrainModel(config: config)
                }
            )
        }
        .confirmationDialog(
            "Start a new bank?",
            isPresented: $showingNewBankConfirmation,
            titleVisibility: .visible
        ) {
            Button("Save current bank first…") {
                showingNewBankConfirmation = false
                saveBankThenClear()
            }
            Button("Discard \(bankStore.totalPatches) patches and start fresh", role: .destructive) {
                bankStore.clearAll(context: modelContext)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Starting a new bank clears the \(bankStore.totalPatches) patches in the current working bank. Annotations and saved bank files on disk are not affected.")
        }
        .confirmationDialog(
            "Clear all classes and annotations?",
            isPresented: $showingClearAllConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear \(store.annotations.count) annotation\(store.annotations.count == 1 ? "" : "s") and \(profileStore.profile.classes.count) class\(profileStore.profile.classes.count == 1 ? "" : "es")", role: .destructive) {
                clearAllClassesAndAnnotations()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Deletes every annotation on the open slide (its sidecar file is emptied) and every class in the active profile. The patch bank, trained models, and saved profile files on disk are not affected. This cannot be undone.")
        }
        .confirmationDialog(
            "Save predicted regions as annotations?",
            isPresented: savePredictionsPromptBinding,
            titleVisibility: .visible,
            presenting: pendingPredictionAnnotations
        ) { anns in
            Button("Add \(anns.count) annotation\(anns.count == 1 ? "" : "s")") {
                store.addBatch(anns)
                pendingPredictionAnnotations = nil
            }
            Button("Cancel", role: .cancel) { pendingPredictionAnnotations = nil }
        } message: { anns in
            Text("Each displayed prediction becomes a square annotation, all assigned the scanned annotation's class, saved to this slide's annotation file. This adds \(anns.count) object\(anns.count == 1 ? "" : "s").")
        }
        .alert("Bank import",
               isPresented: bankImportPromptBinding,
               presenting: pendingBankImport) { req in
            Button("Replace current bank", role: .destructive) {
                performBankImport(req.url, mode: .replace)
                pendingBankImport = nil
            }
            Button("Append to current bank") {
                performBankImport(req.url, mode: .append)
                pendingBankImport = nil
            }
            Button("Cancel", role: .cancel) { pendingBankImport = nil }
        } message: { req in
            Text("The current bank has \(bankStore.totalPatches) patches. How should the imported file be applied?")
        }
    }

    @ViewBuilder
    private func bodyStageC(_ base: some View) -> some View {
        base
        .alert("Import annotations",
               isPresented: annotationImportPromptBinding,
               presenting: pendingAnnotationImport) { req in
            Button("Replace current annotations", role: .destructive) {
                store.importMerge(req.incoming, replace: true)
                pendingAnnotationImport = nil
            }
            Button("Append to existing") {
                store.importMerge(req.incoming, replace: false)
                pendingAnnotationImport = nil
            }
            Button("Cancel", role: .cancel) { pendingAnnotationImport = nil }
        } message: { req in
            Text("The slide currently has \(store.annotations.count) annotations and the file contains \(req.incoming.count). How should it be applied?")
        }
        .sheet(item: $pendingExtract) { _ in
            if let slide {
                ExtractPatchesSheet(
                    slide: slide,
                    allAnnotations: store.annotations,
                    hasSelection: store.selectedID != nil,
                    selectedAnnotation: store.annotations.first(where: { $0.id == store.selectedID }),
                    initialExtractorIdentity: selectedExtractorIdentity,
                    onCancel: { pendingExtract = nil },
                    onExtract: { config, scope in
                        pendingExtract = nil
                        selectedExtractorIdentity = config.extractorIdentity
                        let annotations: [Annotation]
                        switch scope {
                        case .all:
                            annotations = store.annotations
                        case .selected:
                            if let id = store.selectedID,
                               let ann = store.annotations.first(where: { $0.id == id }) {
                                annotations = [ann]
                            } else {
                                annotations = []
                            }
                        }
                        runExtraction(annotations: annotations, config: config)
                    }
                )
            }
        }
        .sheet(isPresented: extractingBinding) {
            ExtractionProgressView(session: extractionSession)
        }
        .sheet(isPresented: tsneComputingBinding) {
            TSNEProgressView(store: embeddingStore, onStop: { tsneTask?.cancel() })
        }
        .alert("Extraction complete",
               isPresented: extractDoneBinding,
               presenting: extractionSession.report) { report in
            Button("OK") { extractionSession.report = nil }
        } message: { report in
            let perClass = report.perClass
                .sorted { $0.key < $1.key }
                .map { "\($0.key): \($0.value)" }
                .joined(separator: ", ")
            if !perClass.isEmpty {
                Text("Saved \(report.saved) patches (\(perClass)). Skipped \(report.skipped).")
            } else {
                Text("Saved \(report.saved) patches. Skipped \(report.skipped).")
            }
        }
        .sheet(item: $pendingExtractors) { _ in
            ExtractorsListSheet(onClose: { pendingExtractors = nil })
        }
        .sheet(item: $pendingTSNE) { _ in
            TSNEConfigSheet(
                info: TSNEConfigSheet.BankInfo(
                    total: bankStore.totalPatches,
                    perExtractor: bankStore.perExtractor,
                    perExtractorPerClass: bankStore.perExtractorPerClass,
                    perClass: bankStore.perClass
                ),
                onCancel: { pendingTSNE = nil },
                onRun: { req in
                    pendingTSNE = nil
                    runTSNE(request: req)
                }
            )
        }
    }

    // MARK: Annotation preview

    private func showPreviewForSelectedAnnotation() {
        guard let id = store.selectedID,
              let ann = store.annotations.first(where: { $0.id == id }) else {
            errorMessage = "Select an annotation first."
            return
        }
        showPreview(for: ann)
    }

    private func showPreview(for ann: Annotation) {
        guard let slide = slide else { return }
        let slideName = slideURL?.deletingPathExtension().lastPathComponent
        let state = previewState

        state.isGenerating = true
        state.annotation = ann
        state.sourceSlideName = slideName
        state.outputBBox = nil
        state.slideDimensions = slide.dimensions
        openWindow(id: "annotation-preview")

        Task.detached {
            let result = await AnnotationExporter.renderPreview(
                for: ann,
                slide: slide,
                cropShape: .square,
                paddingLevel0: 0,
                targetMaxDim: 1024
            )
            await MainActor.run {
                state.image = result?.image
                state.outputBBox = result?.outputBBox
                state.isGenerating = false
            }
        }
    }

    private static let lastSlideURLKey = "PaNINDetector.lastSlideURL"

    private var exportingBinding: Binding<Bool> {
        Binding(get: { exportSession.isExporting },
                set: { exportSession.isExporting = $0 })
    }

    private var exportDoneBinding: Binding<Bool> {
        Binding(get: { exportSession.report != nil },
                set: { if !$0 { exportSession.report = nil } })
    }

    private var extractingBinding: Binding<Bool> {
        Binding(get: { extractionSession.isExtracting },
                set: { extractionSession.isExtracting = $0 })
    }

    private var tsneComputingBinding: Binding<Bool> {
        Binding(get: { embeddingStore.isComputing },
                set: { embeddingStore.isComputing = $0 })
    }

    private var extractDoneBinding: Binding<Bool> {
        Binding(get: { extractionSession.report != nil },
                set: { if !$0 { extractionSession.report = nil } })
    }

    private var bankImportPromptBinding: Binding<Bool> {
        Binding(get: { pendingBankImport != nil },
                set: { if !$0 { pendingBankImport = nil } })
    }

    private var savePredictionsPromptBinding: Binding<Bool> {
        Binding(get: { pendingPredictionAnnotations != nil },
                set: { if !$0 { pendingPredictionAnnotations = nil } })
    }

    private var annotationImportPromptBinding: Binding<Bool> {
        Binding(get: { pendingAnnotationImport != nil },
                set: { if !$0 { pendingAnnotationImport = nil } })
    }

    @ViewBuilder
    private var detail: some View {
        if let slide {
            let _ = store.annotations.count
            let _ = store.selectedID
            ZStack(alignment: .top) {
                SlideCanvasView(slide: slide, store: store, predictedStore: predictedStore, predictionStore: predictionStore, renderSettings: renderSettings, tool: $tool) { pts in
                    pending = PendingPolygon(points: pts)
                }
                .id(slide.url)
                FloatingToolBar(tool: $tool, predictionStore: predictionStore)
                    .padding(.top, 8)
            }
        } else {
            ContentUnavailableView(
                "No slide open",
                systemImage: "photo.stack",
                description: Text("Use File ▸ Open Slide… (⌘O) or the toolbar button")
            )
        }
    }

    private func openPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open whole-slide image"
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        var types: [UTType] = []
        for ext in ["svs", "tif", "tiff", "ndpi", "mrxs", "scn", "vms", "vmu", "bif", "dcm"] {
            if let t = UTType(filenameExtension: ext) { types.append(t) }
        }
        if !types.isEmpty { panel.allowedContentTypes = types }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        openSlide(at: url)
    }

    private func openSlide(at url: URL) {
        do {
            let s = try SlideImage(url: url)
            slide = s
            slideURL = url
            store.bind(toSlideURL: url)
            // Predicted (auto-proposed) regions live in their own companion file
            // and their own sidebar tab so they don't mix with manual work.
            predictedStore.bind(toSlideURL: url, sidecarSuffix: "predicted.geojson")
            tool = .pan
            // Predictions are slide-specific; load this slide's saved results
            // (keyed per annotation) from the companion file next to the slide.
            predictionStore.bind(toSlideURL: url)
            UserDefaults.standard.set(url.path, forKey: Self.lastSlideURLKey)
        } catch {
            errorMessage = String(describing: error)
        }
    }

    /// Close the currently open slide and return to the empty state.
    /// Annotations are flushed to their GeoJSON sidecar first, the slide
    /// handle is released, and slide-specific state (predictions, selection)
    /// is wiped. Bank + trained models are unaffected — they're cross-slide.
    private func closeSlide() {
        guard slide != nil else { return }
        store.unbind()
        predictedStore.unbind()
        predictionStore.unbind()
        tool = .pan
        slide = nil
        slideURL = nil
        UserDefaults.standard.removeObject(forKey: Self.lastSlideURLKey)
    }

    // MARK: Profile I/O

    private func saveProfilePanel() {
        let panel = NSSavePanel()
        panel.title = "Save Annotation Profile"
        panel.nameFieldStringValue = "\(profileStore.profile.name).panin-profile.json"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try profileStore.saveToFile(url)
        } catch {
            errorMessage = "Could not save profile: \(error)"
        }
    }

    private func openProfilePanel() {
        let panel = NSOpenPanel()
        panel.title = "Open Annotation Profile"
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try profileStore.loadFromFile(url)
        } catch {
            errorMessage = "Could not load profile: \(error)"
        }
    }

    // MARK: Annotation export

    private func exportAnnotationsPanel() {
        guard slide != nil else {
            errorMessage = "Open a slide first."
            return
        }
        guard !store.annotations.isEmpty else {
            errorMessage = "This slide has no annotations to export."
            return
        }

        let panel = NSOpenPanel()
        panel.title = "Choose export folder"
        panel.message = "Annotations will be saved into a sub-folder named after the slide."
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        pendingExport = ExportDirRequest(url: dir)
    }

    /// Browse to a QuPath-exported `.geojson` file and merge its annotations
    /// into the open slide. Empty-state goes straight in; non-empty triggers
    /// the Replace / Append confirmation alert. QuPath uses top-left origin
    /// while this app's canvas renders SVS data vertically mirrored, so
    /// each Y is flipped to `slideH - y` at import time.
    private func importAnnotationsPanel() {
        guard let slide else {
            errorMessage = "Open a slide first — annotations live in slide coordinates."
            return
        }
        let panel = NSOpenPanel()
        panel.title = "Import Annotations from GeoJSON"
        panel.message = "Pick a QuPath-exported GeoJSON file. Coordinates are interpreted as base-level slide pixels with QuPath's top-left origin."
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let data = try Data(contentsOf: url)
            let raw = try GeoJSON.decode(data)
            guard !raw.isEmpty else {
                errorMessage = "No annotations found in \(url.lastPathComponent)."
                return
            }
            let incoming = mirrorYForImport(raw, slideHeight: slide.dimensions.height)
            print("[annotations] mirrored Y of \(incoming.count) annotation(s) using slideH=\(Int(slide.dimensions.height))")
            if store.annotations.isEmpty {
                store.importMerge(incoming, replace: true)
                print("[annotations] imported \(incoming.count) from \(url.path)")
            } else {
                pendingAnnotationImport = AnnotationImportRequest(url: url, incoming: incoming)
            }
        } catch {
            errorMessage = "Could not read GeoJSON: \(error)"
        }
    }

    /// Flip Y of every point in every annotation: `y → slideH - y`. Brings
    /// QuPath's top-left coordinates into the app's vertically-mirrored
    /// canvas space (also used in reverse for export).
    private func mirrorYForImport(_ annotations: [Annotation],
                                  slideHeight: CGFloat) -> [Annotation] {
        annotations.map { ann in
            Annotation(
                id: ann.id,
                points: ann.points.map { CGPoint(x: $0.x, y: slideHeight - $0.y) },
                classification: ann.classification,
                color: ann.color,
                name: ann.name,
                isVisible: ann.isVisible
            )
        }
    }

    /// Export the slide's annotations to a QuPath-compatible GeoJSON file.
    /// Y is flipped at write-time so coordinates land in QuPath's top-left
    /// origin. The auto-saved sidecar (`.geojson` next to the slide) stays
    /// un-flipped to preserve compatibility with this app's internal load.
    private func exportGeoJSONPanel() {
        guard let slide else {
            errorMessage = "Open a slide first."
            return
        }
        guard !store.annotations.isEmpty else {
            errorMessage = "This slide has no annotations to export."
            return
        }
        let defaultName = slideURL?.deletingPathExtension().lastPathComponent ?? "annotations"
        let panel = NSSavePanel()
        panel.title = "Export Annotations as GeoJSON"
        panel.message = "QuPath-compatible — Y is flipped on write so the file opens upright in QuPath."
        panel.nameFieldStringValue = "\(defaultName)-qupath.geojson"
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let mirrored = mirrorYForImport(store.annotations,
                                            slideHeight: slide.dimensions.height)
            let data = try GeoJSON.encode(mirrored)
            try data.write(to: url, options: [.atomic])
            print("[annotations] exported \(mirrored.count) to \(url.path) with Y mirrored")
        } catch {
            errorMessage = "Could not export GeoJSON: \(error)"
        }
    }

    // MARK: Patch extraction

    private func requestExtractPatches() {
        guard slide != nil else {
            errorMessage = "Open a slide first."
            return
        }
        guard !store.annotations.isEmpty else {
            errorMessage = "Draw at least one annotation before extracting patches."
            return
        }
        pendingExtract = ExtractRequest()
    }

    // MARK: Bank save / load / new

    // MARK: Prediction

    private func requestPredict() {
        guard slide != nil else {
            errorMessage = "Open a slide first."
            return
        }
        guard classifierStore.classifier != nil else {
            errorMessage = "Train a model first."
            return
        }
        guard let id = store.selectedID,
              let ann = store.annotations.first(where: { $0.id == id }) else {
            errorMessage = "Select an annotation in the sidebar first."
            return
        }
        pendingPredict = PredictRequest(annotation: ann)
    }

    /// Convert the currently-displayed predictions into persistent annotations
    /// in the slide's GeoJSON sidecar. Each patch becomes a square polygon
    /// labeled with its predicted class, positioned exactly where the heatmap
    /// block is drawn (same Y-mirror as the overlay). Only predictions passing
    /// the confidence threshold and class-visibility toggles are included — i.e.
    /// what's currently on screen. Presents a confirmation before committing.
    private func requestSavePredictionsAsAnnotations() {
        guard let slide else {
            errorMessage = "Open a slide first."
            return
        }
        let preds = predictionStore.predictions
        guard !preds.isEmpty else {
            errorMessage = "Run a classification on an annotation first."
            return
        }
        // Every saved square is assigned the class of the scanned annotation,
        // not its own predicted label — the user wants them grouped under one
        // common class. Fall back to the recorded source class, then a generic
        // "Prediction" label if the source annotation is gone.
        let source = store.annotations.first(where: { $0.id == predictionStore.displayedID })
        let commonLabel = source?.classification
            ?? predictionStore.sourceAnnotationClass
            ?? "Prediction"
        let commonColor = source?.color
            ?? predictionStore.classColors[commonLabel]
            ?? .defaultColor
        let slideH = slide.dimensions.height
        let minProb = predictionStore.minProbability
        let newAnns: [Annotation] = preds.compactMap { pred in
            guard pred.maxProbability >= minProb else { return nil }
            guard predictionStore.isClassVisible(pred.predictedLabel) else { return nil }
            let size = CGFloat(pred.sizeLevel0)
            let x = CGFloat(pred.dataX)
            // Mirror Y into overlay space so the square lands on the same pixels
            // the heatmap paints (matches drawPredictions in the overlay).
            let y = slideH - CGFloat(pred.dataY) - size
            let pts = [
                CGPoint(x: x, y: y),
                CGPoint(x: x + size, y: y),
                CGPoint(x: x + size, y: y + size),
                CGPoint(x: x, y: y + size)
            ]
            return Annotation(points: pts,
                              classification: commonLabel,
                              color: commonColor)
        }
        guard !newAnns.isEmpty else {
            errorMessage = "No visible predictions to save. Check the confidence slider and the per-class toggles."
            return
        }
        pendingPredictionAnnotations = newAnns
    }

    /// Pull features from the bank, run t-SNE off the main thread, store the
    /// result in `embeddingStore`, and open the plot window when done.
    private func runTSNE(request: TSNEConfigSheet.RunRequest) {
        embeddingStore.isComputing = true
        embeddingStore.progress = 0
        embeddingStore.status = "Loading bank…"
        let context = modelContext
        let store = embeddingStore
        let openWindowAction = openWindow

        let task = Task.detached {
            do {
                let loaded = try await MainActor.run {
                    try TSNE.loadBank(
                        context: context,
                        extractorIdentity: request.extractorIdentity,
                        maxCount: request.maxCount,
                        maxWhiteFraction: request.maxWhiteFraction,
                        minNuclei: request.minNuclei
                    )
                }
                let progressClosure: @Sendable (Int, Int, String) -> Void = { current, total, msg in
                    Task { @MainActor in
                        store.progress = total > 0 ? Float(current) / Float(total) : 0
                        store.status = msg
                    }
                }
                let points = try TSNE.embed(
                    features: loaded.features,
                    dim: loaded.dim,
                    count: loaded.labels.count,
                    config: request.config,
                    progress: progressClosure
                )
                let simdPoints = points.map { SIMD2<Float>($0.0, $0.1) }
                // Compute per-class centroids in feature + plot space for the
                // overlay / future nearest-centroid classifier.
                let centroidSet = CentroidComputer.compute(
                    features: loaded.features,
                    dim: loaded.dim,
                    labels: loaded.labels,
                    plotPoints: simdPoints,
                    extractorIdentity: loaded.extractorIdentity
                )
                await MainActor.run {
                    store.current = EmbeddingStore.Embedding(
                        points: simdPoints,
                        labels: loaded.labels,
                        slideNames: loaded.slideNames,
                        extractorIdentity: loaded.extractorIdentity,
                        perplexity: request.config.perplexity,
                        iterations: request.config.iterations,
                        totalAvailable: loaded.totalAvailable,
                        createdAt: Date()
                    )
                    store.centroids = centroidSet
                    // Auto-promote: every t-SNE Classifier run also produces a
                    // ready-to-use centroid classifier in the centroid slot.
                    // The user can flip back to logistic in the sidebar.
                    promoteCentroidsToClassifier()
                    store.isComputing = false
                    store.status = "Done — \(simdPoints.count) points, \(centroidSet.centroids.count) centroids"
                    tsneTask = nil
                    openWindowAction(id: "tsne-plot")
                }
            } catch is CancellationError {
                await MainActor.run {
                    store.isComputing = false
                    store.status = "Cancelled"
                    tsneTask = nil
                    print("[tsne] cancelled by user")
                }
            } catch {
                await MainActor.run {
                    store.isComputing = false
                    store.status = "Failed: \(error)"
                    tsneTask = nil
                    errorMessage = String(describing: error)
                }
            }
        }
        tsneTask = task
    }

    /// Replace the active classifier with one derived from the current
    /// CentroidSet. The live embedding is captured into the classifier so the
    /// plot survives Save Classifier As… / Open Classifier… round trips.
    private func promoteCentroidsToClassifier() {
        guard let cset = embeddingStore.centroids else {
            errorMessage = "No centroids available. Run a t-SNE classifier first."
            return
        }
        let model = cset.toMLClassifier(embedding: embeddingStore.current)
        classifierStore.classifier = model
        classifierStore.lastFileURL = nil
        let withPlot = model.embeddingSnapshot != nil ? " (with embedded plot)" : ""
        print("[centroids] promoted to classifier: \(model.classCount) classes, " +
              "train acc \(Int(model.metrics.trainAccuracy * 100))%, " +
              "extractor \(model.resolvedExtractorIdentity.stringForm)\(withPlot)")
    }

    private func runPrediction(annotation: Annotation,
                               multiPass: PredictionPipeline.MultiPassConfig,
                               minConfidence: Float) {
        guard let slide = slide else { return }
        guard let classifier = classifierStore.classifier else { return }
        let store = predictionStore

        // Build the class-color palette: profile palette first, fall back to
        // a deterministic generator so unknown classes still get something.
        var colors: [String: AnnotationColor] = [:]
        for c in profileStore.profile.classes {
            colors[c.name] = c.color
        }
        for label in classifier.classLabels where colors[label] == nil {
            colors[label] = deterministicColor(for: label)
        }

        let annotationID = annotation.id
        let sourceClass = annotation.classification
        store.beginRun(annotationID: annotationID)

        // Geometry models classify the annotation from its handcrafted
        // descriptors rather than the patch bank — route to that pipeline.
        if classifier.isGeometry {
            let sampling = multiPass.perClass[annotation.classification]
                ?? multiPass.perClass.values.first
                ?? PredictionPipeline.PassConfig(patchSizeLevel: 224, strideLevel: 224, level: 0)
            Task.detached {
                do {
                    let report = try GeometryPipeline.predict(
                        slide: slide, annotation: annotation,
                        classifier: classifier, sampling: sampling
                    )
                    await MainActor.run {
                        let result = PredictionResult(
                            predictions: report.predictions,
                            passes: report.passes,
                            perClassCount: report.perClass,
                            classColors: colors,
                            sourceAnnotationClass: sourceClass,
                            minProbability: minConfidence,
                            hiddenClasses: [],
                            createdAt: Date()
                        )
                        store.finishRun(result, for: annotationID)
                        // Surface the actual per-class probabilities so the
                        // decision is visible without the Xcode console.
                        let probs = report.predictions.first?.probabilities ?? [:]
                        let breakdown = probs.sorted { $0.value > $1.value }
                            .map { "\($0.key): \(Int(($0.value * 100).rounded()))%" }
                            .joined(separator: "\n")
                        self.infoMessage = "Geometry prediction for a “\(sourceClass)” annotation:\n\n\(breakdown)"
                    }
                } catch {
                    await MainActor.run {
                        store.failRun()
                        self.errorMessage = "Prediction failed: \(error)"
                    }
                }
            }
            return
        }

        let progress: @Sendable (Int, Int) -> Void = { current, total in
            Task { @MainActor in
                store.progressCurrent = current
                store.progressTotal = total
            }
        }

        Task.detached {
            do {
                let report = try await PredictionPipeline.run(
                    slide: slide,
                    annotation: annotation,
                    classifier: classifier,
                    multiPass: multiPass,
                    progress: progress
                )
                await MainActor.run {
                    let result = PredictionResult(
                        predictions: report.predictions,
                        passes: report.passes,
                        perClassCount: report.perClass,
                        classColors: colors,
                        sourceAnnotationClass: sourceClass,
                        minProbability: minConfidence,
                        hiddenClasses: [],
                        createdAt: Date()
                    )
                    store.finishRun(result, for: annotationID)
                    print("[predict] total=\(report.total) skipped=\(report.skipped) perClass=\(report.perClass) passes=\(report.passes.count)")
                }
            } catch {
                await MainActor.run {
                    store.failRun()
                    self.errorMessage = "Prediction failed: \(error)"
                }
            }
        }
    }

    // MARK: VISTA (MicePan tissue segmenter) comparison

    /// Run the VISTA tissue segmenter on the selected annotation. Produces the
    /// same per-tile heatmap/overlay as the classifier path, but labels are
    /// tissue types (Neoplasia / Metaplasia / Normal / Stroma).
    private func requestPredictVISTA() {
        guard let slide = slide else {
            errorMessage = "Open a slide first."
            return
        }
        guard VISTASegmenter.isInstalled else {
            errorMessage = "VISTA models not installed.\n\nPut VISTA-neoplasia.mlpackage, VISTA-metaplasia.mlpackage and VISTA-normal.mlpackage in:\n\(VISTASegmenter.modelsDirectory.path)"
            return
        }
        guard let id = store.selectedID,
              let annotation = store.annotations.first(where: { $0.id == id }) else {
            errorMessage = "Select an annotation in the sidebar first."
            return
        }

        let segmenter: VISTASegmenter
        do { segmenter = try VISTASegmenter() }
        catch { errorMessage = "Could not load VISTA: \(error)"; return }

        let store = predictionStore
        let annotationID = annotation.id
        let sourceClass = annotation.classification
        let colors = VISTASegmenter.classColors
        store.beginRun(annotationID: annotationID)

        let progress: @Sendable (Int, Int) -> Void = { current, total in
            Task { @MainActor in
                store.progressCurrent = current
                store.progressTotal = total
            }
        }

        Task.detached {
            do {
                let report = try await VISTAPipeline.run(
                    slide: slide, annotation: annotation,
                    segmenter: segmenter, progress: progress
                )
                await MainActor.run {
                    let result = PredictionResult(
                        predictions: report.predictions,
                        passes: report.passes,
                        perClassCount: report.perClass,
                        classColors: colors,
                        sourceAnnotationClass: sourceClass,
                        minProbability: 0,
                        hiddenClasses: [],
                        createdAt: Date()
                    )
                    store.finishRun(result, for: annotationID)
                    let breakdown = report.perClass.sorted { $0.value > $1.value }
                        .map { "\($0.key): \($0.value) tiles" }
                        .joined(separator: "\n")
                    self.infoMessage = "VISTA tissue segmentation on a “\(sourceClass)” annotation (\(report.total) tiles):\n\n\(breakdown)"
                    print("[vista] total=\(report.total) skipped=\(report.skipped) perClass=\(report.perClass)")
                }
            } catch {
                await MainActor.run {
                    store.failRun()
                    self.errorMessage = "VISTA prediction failed: \(error)"
                }
            }
        }
    }

    /// Deterministic hash-based color for class labels not present in the profile.
    private func deterministicColor(for label: String) -> AnnotationColor {
        var h: UInt64 = 0xcbf29ce484222325
        for byte in label.utf8 {
            h ^= UInt64(byte)
            h &*= 0x100000001b3
        }
        let r = Int((h >> 16) & 0xFF)
        let g = Int((h >> 8) & 0xFF)
        let b = Int(h & 0xFF)
        // Push into the saturated mid-range so it's visible against tissue.
        return AnnotationColor(
            r: 80 + r / 2,
            g: 80 + g / 2,
            b: 80 + b / 2
        )
    }

    // MARK: Geometry features

    /// Compute handcrafted geometry descriptors for every annotation on the open
    /// slide and add them to the accumulating geometry bank.
    private func extractGeometry() {
        guard let slide = slide, let url = slideURL else {
            errorMessage = "Open a slide with annotations first."
            return
        }
        let annotations = store.annotations
        guard !annotations.isEmpty else {
            errorMessage = "No annotations on this slide to extract geometry from."
            return
        }
        let gs = geometryStore
        let slidePath = url.path(percentEncoded: false)
        let slideName = url.deletingPathExtension().lastPathComponent
        gs.isExtracting = true
        gs.extractCurrent = 0
        gs.extractTotal = annotations.count
        let progress: @Sendable (Int, Int) -> Void = { c, t in
            Task { @MainActor in gs.extractCurrent = c; gs.extractTotal = t }
        }
        Task.detached {
            let records = GeometryPipeline.extract(
                slide: slide, slidePath: slidePath, slideName: slideName,
                annotations: annotations, progress: progress
            )
            await MainActor.run {
                gs.add(records)
                gs.isExtracting = false
                self.infoMessage = "Geometry extracted for \(records.count) of \(annotations.count) annotation(s). Bank now holds \(gs.total) across all slides."
            }
        }
    }

    /// Train a logistic-regression classifier on the geometry bank and make it
    /// the active classifier.
    private func trainGeometryModel() {
        let records = geometryStore.records
        guard records.count >= 4 else {
            errorMessage = "Extract geometry on more annotations first (have \(records.count), need ≥4)."
            return
        }
        let gs = geometryStore
        let store = classifierStore
        gs.isTraining = true
        gs.trainingStatus = "Training…"
        Task { @MainActor in
            do {
                let classifier = try await Task.detached {
                    try GeometryTrainer.train(records: records, progress: { _, _, msg in
                        Task { @MainActor in gs.trainingStatus = msg }
                    })
                }.value
                store.classifier = classifier
                store.lastFileURL = nil
                gs.isTraining = false
                gs.trainingStatus = ""
                self.infoMessage = "Geometry model trained: \(classifier.classCount) classes, cross-validated accuracy \(Int((classifier.metrics.valAccuracy * 100).rounded()))% (\(classifier.metrics.valCount) held-out annotations). It's now the active classifier — run Classify on Annotation to use it. Check the Classifier tab for the confusion matrix."
            } catch {
                gs.isTraining = false
                gs.trainingStatus = ""
                self.errorMessage = "Geometry training failed: \(error)"
            }
        }
    }

    /// Detect-then-grade cascade: merge the current per-patch prediction's
    /// PanIN tiles into lesion regions, then grade each with the active geometry
    /// model and add the graded regions as annotations.
    /// Open the class picker so the user chooses which predicted classes count
    /// as candidate PanIN regions before merging + grading.
    private func requestProposeRegions() {
        guard slide != nil else { errorMessage = "Open a slide first."; return }
        guard let classifier = classifierStore.classifier, classifier.isGeometry else {
            errorMessage = "Make a geometry model active first (Analysis ▸ Train Geometry Model), then run this on a per-patch prediction."
            return
        }
        guard !predictionStore.predictions.isEmpty else {
            errorMessage = "Run a per-patch classification over a region first — this merges its PanIN tiles into lesions and grades them."
            return
        }
        let labels = predictionStore.predictedLabels
        guard !labels.isEmpty else { errorMessage = "No predicted classes in the current prediction."; return }
        pendingProposeRegions = ProposeRegionsRequest(labels: labels, counts: predictionStore.perClassCount)
    }

    private func proposeAndGradeRegions(candidateLabels: Set<String>) {
        guard let slide else { errorMessage = "Open a slide first."; return }
        guard let classifier = classifierStore.classifier, classifier.isGeometry else {
            errorMessage = "Make a geometry model active first (Analysis ▸ Train Geometry Model), then run this on a per-patch prediction."
            return
        }
        let preds = predictionStore.predictions
        guard !preds.isEmpty else {
            errorMessage = "Run a per-patch classification over a region first — this merges its PanIN tiles into lesions and grades them."
            return
        }
        let slideH = slide.dimensions.height
        let regions = RegionProposer.propose(
            predictions: preds,
            slideHeight: slideH,
            isCandidate: { candidateLabels.contains($0) },
            minTiles: 2
        )
        guard !regions.isEmpty else {
            errorMessage = "No PanIN-predicted tiles in the current prediction to merge into regions."
            return
        }

        // Palette for the grade labels, resolved on the main actor up front.
        var colors: [String: AnnotationColor] = [:]
        for c in profileStore.profile.classes { colors[c.name] = c.color }
        for label in classifier.classLabels where colors[label] == nil {
            colors[label] = deterministicColor(for: label)
        }

        let total = regions.count
        Task.detached {
            var results: [(poly: [CGPoint], label: String)] = []
            for region in regions {
                let probe = Annotation(points: region.polygonOverlayY,
                                       classification: "PanIN?", color: .defaultColor)
                if let (label, _) = try? GeometryPipeline.grade(
                    slide: slide, annotation: probe, classifier: classifier
                ) {
                    results.append((region.polygonOverlayY, label))
                }
            }
            let graded = results
            await MainActor.run {
                let newAnns = graded.map { g in
                    Annotation(points: g.poly, classification: g.label,
                               color: colors[g.label] ?? .defaultColor)
                }
                self.predictedStore.addBatch(newAnns)
                let skipped = total - graded.count
                self.infoMessage = "Proposed \(total) PanIN region(s); graded \(graded.count) and added them to the Predicted tab." +
                    (skipped > 0 ? " \(skipped) skipped (too little tissue to describe)." : "")
            }
        }
    }

    // MARK: Model train / save / load

    /// True when the active profile defines any "Exclude / null" class.
    private var hasNullClasses: Bool {
        profileStore.profile.classes.contains { $0.isNull }
    }

    /// Wipe every annotation on the open slide and every class in the active
    /// profile. Invoked from the confirmation dialog only.
    private func clearAllClassesAndAnnotations() {
        let annCount = store.annotations.count
        let classCount = profileStore.profile.classes.count
        store.removeAll()
        profileStore.removeAllClasses()
        // Any displayed prediction belonged to a now-deleted annotation.
        predictionStore.displayedID = nil
        infoMessage = "Cleared \(annCount) annotation\(annCount == 1 ? "" : "s") and \(classCount) class\(classCount == 1 ? "" : "es")."
    }

    /// Menu commands post notifications; this drives all of them from a single
    /// `.task` so the `body` modifier chain stays small enough to type-check.
    /// Each listener runs in its own long-lived task on the main actor.
    @MainActor
    private func runNotificationHandlers() async {
        // Discover installed Core ML feature extractors once at launch.
        ExtractorRegistry.shared.discover()

        func listen(_ name: Notification.Name, _ action: @escaping @MainActor () -> Void) {
            Task { @MainActor in
                for await _ in NotificationCenter.default.notifications(named: name) { action() }
            }
        }

        listen(.saveProfileRequested) { saveProfilePanel() }
        listen(.openProfileRequested) { openProfilePanel() }
        listen(.exportAnnotationsRequested) { exportAnnotationsPanel() }
        listen(.showAnnotationPreviewRequested) { showPreviewForSelectedAnnotation() }
        listen(.extractPatchesRequested) { requestExtractPatches() }
        listen(.saveBankRequested) { saveBankPanel() }
        listen(.openBankRequested) { openBankPanel() }
        listen(.newBankRequested) { requestNewBank() }
        listen(.revealBankRequested) { revealWorkingBank() }
        listen(.trainModelRequested) { requestTrainModel() }
        listen(.saveModelRequested) { saveModelPanel() }
        listen(.openModelRequested) { openModelPanel() }
        listen(.predictAnnotationRequested) { requestPredict() }
        listen(.predictVISTARequested) { requestPredictVISTA() }
        listen(.clearPredictionsRequested) { predictionStore.reset() }
        listen(.clearClassesAndAnnotationsRequested) { showingClearAllConfirmation = true }
        listen(.extractGeometryRequested) { extractGeometry() }
        listen(.trainGeometryModelRequested) { trainGeometryModel() }
        listen(.proposeAndGradeRegionsRequested) { requestProposeRegions() }
        listen(.revealGeometryBankRequested) {
            if let url = geometryStore.bankURL,
               FileManager.default.fileExists(atPath: url.path) {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            } else {
                errorMessage = "The geometry bank is empty — extract geometry first."
            }
        }
        listen(.showExtractorsRequested) { pendingExtractors = ExtractorsRequest() }
        listen(.closeSlideRequested) { closeSlide() }
        listen(.tsneAnalysisRequested) { pendingTSNE = TSNERequest() }
        listen(.promoteCentroidsRequested) { promoteCentroidsToClassifier() }
        listen(.showTSNEPlotRequested) { openWindow(id: "tsne-plot") }
        listen(.importAnnotationsRequested) { importAnnotationsPanel() }
        listen(.exportGeoJSONRequested) { exportGeoJSONPanel() }
        listen(.showHelpRequested) { openWindow(id: "help") }
        listen(.savePredictionsAsAnnotationsRequested) { requestSavePredictionsAsAnnotations() }
        listen(.togglePredictionHeatmapRequested) { predictionStore.isVisible.toggle() }
    }

    private func requestTrainModel() {
        if bankStore.totalPatches == 0 {
            errorMessage = "Extract patches into the bank first."
            return
        }
        // Refresh stats just before opening so the sheet shows current numbers.
        bankStore.refresh(context: modelContext)
        pendingTrainModel = TrainModelRequest()
    }

    private func runTrainModel(config: ClassifierTrainer.Config) {
        let store = classifierStore
        let context = modelContext

        Task { @MainActor in
            store.isTraining = true
            store.trainingProgress = 0
            store.trainingStatus = "Loading bank…"
            do {
                let nullClasses = Set(profileStore.profile.classes
                    .filter { $0.isNull }.map { $0.name })
                let loaded = try ClassifierTrainer.loadExamples(
                    context: context,
                    maxWhiteFraction: config.maxWhiteFraction,
                    enabledClasses: config.enabledClasses,
                    binarizeAgainstRest: config.binarizeAgainstRest,
                    extractorIdentity: config.extractorIdentity,
                    poolPerAnnotation: config.poolPerAnnotation,
                    nullClasses: nullClasses,
                    nullThreshold: config.nullThreshold
                )
                let progressClosure: @Sendable (Int, Int, String) -> Void = { current, total, msg in
                    Task { @MainActor in
                        store.trainingProgress = total > 0 ? Float(current) / Float(total) : 0
                        store.trainingStatus = msg
                    }
                }
                let classifier = try await Task.detached {
                    ClassifierTrainer.train(
                        loaded: loaded,
                        config: config,
                        progress: progressClosure
                    )
                }.value

                store.classifier = classifier
                store.isTraining = false
                store.trainingStatus = ""
                store.trainingProgress = 1
                print("[train] train=\(classifier.metrics.trainAccuracy) val=\(classifier.metrics.valAccuracy) loss=\(classifier.metrics.finalLoss)")
            } catch {
                store.isTraining = false
                store.trainingStatus = ""
                self.errorMessage = "Training failed: \(error)"
            }
        }
    }

    private func saveModelPanel() {
        guard let classifier = classifierStore.classifier else {
            errorMessage = "Train or promote a classifier before saving."
            return
        }
        let panel = NSSavePanel()
        panel.title = "Save Classifier"
        let defaultName: String = {
            switch classifier.classifierKind {
            case .logistic: return "logistic"
            case .centroid: return "centroid"
            }
        }()
        panel.nameFieldStringValue = "\(defaultName).\(ModelSerializer.suggestedExtension)"
        if let clType = UTType(filenameExtension: ModelSerializer.suggestedExtension) {
            panel.allowedContentTypes = [clType]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try ModelSerializer.save(classifier, to: url)
            classifierStore.lastFileURL = url
        } catch {
            errorMessage = "Could not save classifier: \(error)"
        }
    }

    private func openModelPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open Classifier"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        // Accept both the new `.cl` files and legacy `*.paninmodel.json`.
        var types: [UTType] = [.json]
        if let clType = UTType(filenameExtension: "cl") { types.append(clType) }
        panel.allowedContentTypes = types
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let classifier = try ModelSerializer.load(from: url)
            classifierStore.classifier = classifier
            classifierStore.lastFileURL = url
            // If the file embeds a t-SNE snapshot, restore it into the
            // shared embedding store so the plot window opens identically.
            if let snap = classifier.embeddingSnapshot {
                embeddingStore.current = snap.toEmbedding()
                embeddingStore.centroids = snap.toCentroidSet(using: classifier)
                print("[classifier] restored embedded plot: " +
                      "\(snap.pointsX.count) points, \(snap.centroidPositions.count) centroids")
            }
        } catch {
            errorMessage = "Could not load classifier: \(error)"
        }
    }

    private func revealWorkingBank() {
        guard let url = modelContext.container.configurations.first?.url else {
            errorMessage = "Could not locate the working bank store on disk."
            return
        }
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            // SwiftData lazily creates the file on first save. Reveal the
            // parent folder so the user can still find the location.
            let parent = url.deletingLastPathComponent()
            NSWorkspace.shared.activateFileViewerSelecting([parent])
        }
    }

    private func requestNewBank() {
        if bankStore.totalPatches == 0 {
            // Nothing to clear — just confirm there's no work to do.
            errorMessage = "Bank is already empty."
            return
        }
        showingNewBankConfirmation = true
    }

    private func saveBankThenClear() {
        let panel = NSSavePanel()
        panel.title = "Save Patch Bank Before Starting Fresh"
        panel.nameFieldStringValue = "patches.\(BankSerializer.suggestedExtension)"
        if let bankType = UTType(filenameExtension: BankSerializer.suggestedExtension) {
            panel.allowedContentTypes = [bankType]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let count = try BankSerializer.export(context: modelContext, to: url)
            print("[bank] saved \(count) patches to \(url.path) before clearing")
            bankStore.clearAll(context: modelContext)
        } catch {
            errorMessage = "Could not save bank: \(error). Bank was not cleared."
        }
    }

    private func saveBankPanel() {
        let panel = NSSavePanel()
        panel.title = "Save Patch Bank"
        panel.nameFieldStringValue = "patches.\(BankSerializer.suggestedExtension)"
        if let bankType = UTType(filenameExtension: BankSerializer.suggestedExtension) {
            panel.allowedContentTypes = [bankType]
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let count = try BankSerializer.export(context: modelContext, to: url)
            print("[bank] saved \(count) patches to \(url.path)")
        } catch {
            errorMessage = "Could not save bank: \(error)"
        }
    }

    private func openBankPanel() {
        let panel = NSOpenPanel()
        panel.title = "Open Patch Bank"
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        // Accept both new `.bank` files and legacy `*.paninbank.json`.
        var types: [UTType] = [.json]
        if let bankType = UTType(filenameExtension: "bank") { types.append(bankType) }
        panel.allowedContentTypes = types
        guard panel.runModal() == .OK, let url = panel.url else { return }

        if bankStore.totalPatches == 0 {
            performBankImport(url, mode: .replace)
        } else {
            pendingBankImport = BankImportRequest(url: url)
        }
    }

    private func performBankImport(_ url: URL, mode: BankSerializer.ImportMode) {
        do {
            let count = try BankSerializer.importFile(url, into: modelContext, mode: mode)
            bankStore.refresh(context: modelContext)
            print("[bank] imported \(count) patches (\(mode))")
        } catch {
            errorMessage = "Could not load bank: \(error)"
        }
    }

    private func runExtraction(annotations: [Annotation],
                               config: PatchExtractionPipeline.Config) {
        guard let slide = slide else { return }
        guard !annotations.isEmpty else {
            errorMessage = "No annotations to extract."
            return
        }
        let slidePath = slideURL?.path ?? "unknown"
        let slideName = slideURL?.deletingPathExtension().lastPathComponent ?? "slide"
        let session = extractionSession
        let container = modelContext.container
        let bank = bankStore
        let bankContext = modelContext

        session.isExtracting = true
        session.current = 0
        session.total = 0

        let progress: @Sendable (Int, Int) -> Void = { current, total in
            Task { @MainActor in
                session.current = current
                session.total = total
            }
        }

        Task.detached {
            do {
                let report = try await PatchExtractionPipeline.run(
                    slide: slide,
                    slidePath: slidePath,
                    slideName: slideName,
                    annotations: annotations,
                    config: config,
                    modelContainer: container,
                    progress: progress
                )
                await MainActor.run {
                    session.isExtracting = false
                    session.report = report
                    bank.refresh(context: bankContext)
                }
            } catch {
                await MainActor.run {
                    session.isExtracting = false
                    self.errorMessage = "Extraction failed: \(error)"
                }
            }
        }
    }

    private func runExport(to baseDir: URL,
                           annotations: [Annotation],
                           with options: AnnotationExporter.Options) {
        guard let slide = slide else { return }
        guard !annotations.isEmpty else {
            errorMessage = "No annotations to export."
            return
        }
        let slideName = slideURL?.deletingPathExtension().lastPathComponent ?? "slide"
        let session = exportSession

        session.isExporting = true
        session.current = 0
        session.total = annotations.count

        let progress: @Sendable (Int, Int) -> Void = { current, total in
            Task { @MainActor in
                session.current = current
                session.total = total
            }
        }

        Task.detached {
            do {
                let report = try await AnnotationExporter.export(
                    annotations: annotations,
                    slide: slide,
                    slideName: slideName,
                    baseDirectory: baseDir,
                    options: options,
                    progress: progress
                )
                await MainActor.run {
                    session.isExporting = false
                    session.report = report
                }
            } catch {
                await MainActor.run {
                    session.isExporting = false
                    self.errorMessage = "Export failed: \(error)"
                }
            }
        }
    }
}

private struct ExportProgressView: View {
    @Bindable var session: ExportSession

    var body: some View {
        VStack(spacing: 16) {
            ProgressView(value: session.progressFraction)
                .frame(width: 260)
            Text("Exporting \(session.current) of \(session.total)…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .padding(40)
        .frame(minWidth: 340, minHeight: 150)
        .interactiveDismissDisabled()
    }
}

private struct ExtractionProgressView: View {
    @Bindable var session: ExtractionSession

    var body: some View {
        VStack(spacing: 16) {
            ProgressView(value: session.progressFraction)
                .frame(width: 260)
            Text(session.total > 0
                 ? "Extracting patch \(session.current) of \(session.total)…"
                 : "Sampling patches…")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
        .padding(40)
        .frame(minWidth: 340, minHeight: 150)
        .interactiveDismissDisabled()
    }
}

private struct TSNEProgressView: View {
    @Bindable var store: EmbeddingStore
    let onStop: () -> Void
    @State private var stopping: Bool = false

    var body: some View {
        VStack(spacing: 16) {
            ProgressView(value: Double(store.progress))
                .progressViewStyle(.linear)
                .frame(width: 320)
            Text("t-SNE")
                .font(.headline)
            Text(store.status.isEmpty ? "Working…" : store.status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
                .frame(maxWidth: 320)
                .multilineTextAlignment(.center)
            Button(role: .destructive) {
                stopping = true
                onStop()
            } label: {
                Label(stopping ? "Stopping…" : "Stop", systemImage: "stop.circle")
            }
            .keyboardShortcut(".", modifiers: [.command])
            .disabled(stopping)
        }
        .padding(32)
        .frame(minWidth: 380, minHeight: 200)
        .interactiveDismissDisabled()
    }
}

private struct FloatingToolBar: View {
    @Binding var tool: AnnotationTool
    let predictionStore: PredictionStore
    var body: some View {
        HStack(spacing: 4) {
            ForEach(AnnotationTool.allCases) { t in
                Button {
                    tool = t
                } label: {
                    Label(t.label, systemImage: t.systemImage)
                        .labelStyle(.titleAndIcon)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(tool == t ? Color.accentColor.opacity(0.25) : Color.clear)
                        )
                        .foregroundStyle(tool == t ? Color.accentColor : Color.primary)
                }
                .buttonStyle(.plain)
            }

            Divider().frame(height: 18).padding(.horizontal, 4)

            ZoomButton(systemImage: "minus.magnifyingglass",
                       help: "Zoom out (⌘−)",
                       notification: .zoomOutRequested)
            ZoomButton(systemImage: "arrow.up.left.and.down.right.magnifyingglass",
                       help: "Fit to window (⌘0)",
                       notification: .fitToWindowRequested)
            ZoomButton(systemImage: "plus.magnifyingglass",
                       help: "Zoom in (⌘=)",
                       notification: .zoomInRequested)

            // Prediction heatmap on/off — only shown once a run has produced
            // predictions to toggle.
            if !predictionStore.predictions.isEmpty {
                Divider().frame(height: 18).padding(.horizontal, 4)
                Button {
                    predictionStore.isVisible.toggle()
                } label: {
                    Image(systemName: predictionStore.isVisible ? "square.grid.3x3.fill" : "square.grid.3x3")
                        .frame(width: 28, height: 24)
                        .foregroundStyle(predictionStore.isVisible ? Color.accentColor : Color.primary)
                }
                .buttonStyle(.plain)
                .help(predictionStore.isVisible ? "Hide prediction grid" : "Show prediction grid")
            }
        }
        .padding(4)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(Color.primary.opacity(0.1))
        )
    }
}

private struct ZoomButton: View {
    let systemImage: String
    let help: String
    let notification: Notification.Name
    var body: some View {
        Button {
            NotificationCenter.default.post(name: notification, object: nil)
        } label: {
            Image(systemName: systemImage)
                .frame(width: 28, height: 24)
        }
        .buttonStyle(.plain)
        .help(help)
    }
}

#Preview {
    ContentView()
}
