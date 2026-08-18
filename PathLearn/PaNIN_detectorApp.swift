import SwiftUI
import SwiftData

@main
struct PaNIN_detectorApp: App {
    @State private var previewState = PreviewState()
    @State private var embeddingStore = EmbeddingStore()
    @State private var profileStore = ProfileStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(previewState)
                .environment(embeddingStore)
                .environment(profileStore)
        }
        .modelContainer(for: MLPatch.self, isAutosaveEnabled: true)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Slide…") {
                    NotificationCenter.default.post(name: .openSlideRequested, object: nil)
                }
                .keyboardShortcut("o", modifiers: [.command])
                Button("Close Slide") {
                    NotificationCenter.default.post(name: .closeSlideRequested, object: nil)
                }
                .keyboardShortcut("w", modifiers: [.command])
            }

            CommandGroup(after: .newItem) {
                Divider()
                Button("Open Profile…") {
                    NotificationCenter.default.post(name: .openProfileRequested, object: nil)
                }
                Button("Save Profile As…") {
                    NotificationCenter.default.post(name: .saveProfileRequested, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
                Divider()
                Button("Export Annotations…") {
                    NotificationCenter.default.post(name: .exportAnnotationsRequested, object: nil)
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
                Button("Import Annotations from GeoJSON…") {
                    NotificationCenter.default.post(name: .importAnnotationsRequested, object: nil)
                }
                .keyboardShortcut("i", modifiers: [.command, .shift])
                Button("Export Annotations as GeoJSON…") {
                    NotificationCenter.default.post(name: .exportGeoJSONRequested, object: nil)
                }
                Button("Show Annotation Preview") {
                    NotificationCenter.default.post(name: .showAnnotationPreviewRequested, object: nil)
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
                Button("Clear All Classes & Annotations…") {
                    NotificationCenter.default.post(name: .clearClassesAndAnnotationsRequested, object: nil)
                }
                Divider()
                Button("Extract Patches…") {
                    NotificationCenter.default.post(name: .extractPatchesRequested, object: nil)
                }
                .keyboardShortcut("x", modifiers: [.command, .shift])
                Divider()
                Button("New Bank") {
                    NotificationCenter.default.post(name: .newBankRequested, object: nil)
                }
                Button("Open Bank…") {
                    NotificationCenter.default.post(name: .openBankRequested, object: nil)
                }
                Button("Save Bank As…") {
                    NotificationCenter.default.post(name: .saveBankRequested, object: nil)
                }
                Button("Reveal Working Bank in Finder") {
                    NotificationCenter.default.post(name: .revealBankRequested, object: nil)
                }
                Divider()
                Button("Extractors…") {
                    NotificationCenter.default.post(name: .showExtractorsRequested, object: nil)
                }
            }

            CommandMenu("Analysis") {
                Button("Regression Classifier…") {
                    NotificationCenter.default.post(name: .trainModelRequested, object: nil)
                }
                .keyboardShortcut("t", modifiers: [.command, .option])
                Button("t-SNE Classifier…") {
                    NotificationCenter.default.post(name: .tsneAnalysisRequested, object: nil)
                }
                .keyboardShortcut("a", modifiers: [.command, .option])
                Divider()
                Button("Extract Geometry Features") {
                    NotificationCenter.default.post(name: .extractGeometryRequested, object: nil)
                }
                Button("Train Geometry Model…") {
                    NotificationCenter.default.post(name: .trainGeometryModelRequested, object: nil)
                }
                Button("Reveal Geometry Bank in Finder") {
                    NotificationCenter.default.post(name: .revealGeometryBankRequested, object: nil)
                }
                Button("Propose & Grade PanIN Regions") {
                    NotificationCenter.default.post(name: .proposeAndGradeRegionsRequested, object: nil)
                }
                Divider()
                Button("Open Classifier…") {
                    NotificationCenter.default.post(name: .openModelRequested, object: nil)
                }
                Button("Save Classifier As…") {
                    NotificationCenter.default.post(name: .saveModelRequested, object: nil)
                }
                Divider()
                Button("Classify on Annotation…") {
                    NotificationCenter.default.post(name: .predictAnnotationRequested, object: nil)
                }
                .keyboardShortcut("r", modifiers: [.command, .option])
                Button("Classify with VISTA on Annotation") {
                    NotificationCenter.default.post(name: .predictVISTARequested, object: nil)
                }
                .keyboardShortcut("v", modifiers: [.command, .option])
                Button("Save Predictions as Annotations…") {
                    NotificationCenter.default.post(name: .savePredictionsAsAnnotationsRequested, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.command, .option])
                Button("Show / Hide Prediction Grid") {
                    NotificationCenter.default.post(name: .togglePredictionHeatmapRequested, object: nil)
                }
                .keyboardShortcut("g", modifiers: [.command, .option])
                Button("Clear Predictions") {
                    NotificationCenter.default.post(name: .clearPredictionsRequested, object: nil)
                }
            }

            CommandGroup(replacing: .help) {
                Button("PathLearn Help") {
                    NotificationCenter.default.post(name: .showHelpRequested, object: nil)
                }
                .keyboardShortcut("?", modifiers: [.command])
            }

            CommandGroup(after: .toolbar) {
                Button("Zoom In") {
                    NotificationCenter.default.post(name: .zoomInRequested, object: nil)
                }
                .keyboardShortcut("=", modifiers: [.command])

                Button("Zoom Out") {
                    NotificationCenter.default.post(name: .zoomOutRequested, object: nil)
                }
                .keyboardShortcut("-", modifiers: [.command])

                Button("Fit to Window") {
                    NotificationCenter.default.post(name: .fitToWindowRequested, object: nil)
                }
                .keyboardShortcut("0", modifiers: [.command])
            }
        }

        Window("Annotation Preview", id: "annotation-preview") {
            AnnotationPreviewWindow()
                .environment(previewState)
        }
        .defaultSize(width: 640, height: 720)
        .modelContainer(for: MLPatch.self, isAutosaveEnabled: true)

        Window("t-SNE Plot", id: "tsne-plot") {
            TSNEPlotWindow()
                .environment(embeddingStore)
                .environment(profileStore)
        }
        .defaultSize(width: 900, height: 720)

        Window("PathLearn Help", id: "help") {
            HelpView()
        }
        .defaultSize(width: 640, height: 640)
    }
}
