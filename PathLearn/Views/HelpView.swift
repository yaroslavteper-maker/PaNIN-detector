import SwiftUI

/// User-facing instructions shown from Help ▸ PathLearn Help.
/// Opened as a standalone window (id "help") from the app's menu command.
struct HelpView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header

                HelpSection(title: "Getting Started", systemImage: "play.circle") {
                    HelpStep(number: 1, text: "Open a slide with **File ▸ Open Slide…** (⌘O) or the toolbar button. Supported formats include `.svs`, `.tif`, `.ndpi`, `.mrxs`, `.scn`, and other OpenSlide types.")
                    HelpStep(number: 2, text: "Zoom with the toolbar buttons or **⌘=** / **⌘−**, and **⌘0** to fit the slide to the window. On a trackpad or Magic Mouse, hold ⌘ and scroll to zoom.")
                    HelpStep(number: 3, text: "Pan by dragging with the **Pan** tool selected in the floating toolbar at the top of the canvas.")
                }

                HelpSection(title: "Annotating Regions", systemImage: "pencil.and.outline") {
                    HelpBullet("Annotate your region of choice in the slide — any tissue area you want to label, analyze, or classify.")
                    HelpBullet("Pick the **Lasso** tool to trace a free-form region by dragging, or the **Polygon** tool to place vertices one click at a time.")
                    HelpBullet("While drawing a polygon: **Return** finishes the shape, **Esc** cancels, **Backspace** removes the last vertex, and clicking the first point (or double-clicking) closes it.")
                    HelpBullet("When you finish a shape, choose or create a **class** (label + color). Classes come from the active profile.")
                    HelpBullet("Use the **Annotations** tab in the sidebar to rename, recolor, hide/show, or delete annotations, and to select one for preview or prediction.")
                    HelpBullet("Annotations auto-save to a `.geojson` sidecar next to the slide and reload automatically the next time you open it.")
                }

                HelpSection(title: "Classes & Profiles", systemImage: "tag") {
                    HelpBullet("The **Classes** tab manages your classification profile — the set of labels and colors used for your annotations.")
                    HelpBullet("Mark a class as **Exclude / null** in the class editor to make it an exclusion reference (e.g. lumens or other regions to ignore). Its patches are never learned as a real class, and patches that resemble them can be filtered out of training and prediction.")
                    HelpBullet("Save a profile with **File ▸ Save Profile As…** (⇧⌘S) and load one with **File ▸ Open Profile…** to reuse it across slides.")
                }

                HelpSection(title: "QuPath Interoperability", systemImage: "arrow.left.arrow.right") {
                    HelpBullet("Import annotations from a QuPath GeoJSON file with **File ▸ Import Annotations from GeoJSON…** (⇧⌘I).")
                    HelpBullet("Export QuPath-compatible GeoJSON with **File ▸ Export Annotations as GeoJSON…**. The vertical (Y) axis is flipped on read/write so shapes line up correctly in QuPath.")
                    HelpBullet("Export annotation regions as image tiles (PNG/JPEG) with **File ▸ Export Annotations…** (⇧⌘E).")
                }

                HelpSection(title: "Building a Patch Bank", systemImage: "square.grid.3x3") {
                    HelpStep(number: 1, text: "With annotations drawn, choose **File ▸ Extract Patches…** (⇧⌘X).")
                    HelpStep(number: 2, text: "Set the patch size, stride, pyramid level, and maximum white (background) fraction, then pick a **feature extractor** — the built-in Vision extractor or an installed Core ML model (e.g. a pathology foundation model such as UNI or Phikon).")
                    HelpStep(number: 3, text: "Optionally enable **Require at least N nuclei** — each patch is checked for cell nuclei (hematoxylin deconvolution) and patches with too few are skipped, excluding lumen, stroma, and empty fields. Every extracted patch stores its nucleus count for later filtering.")
                    HelpStep(number: 4, text: "Extracted patches and their feature vectors accumulate in the working **bank**. Track counts per class and per extractor in the **Bank** tab.")
                    HelpBullet("Save the bank with **File ▸ Save Bank As…**, reload with **Open Bank…**, and start over with **New Bank**. **Reveal Working Bank in Finder** locates the on-disk store.")
                    HelpBullet("Manage installable Core ML extractors via **File ▸ Extractors…**.")
                }

                HelpSection(title: "Choosing a Feature Extractor", systemImage: "cpu") {
                    HelpBullet("The feature extractor turns each patch image into a numeric feature vector — it defines what the classifier can \"see.\" Pick it in the **Extract Patches** sheet; the same extractor is then required for training and prediction.")
                    HelpBullet("**Apple Vision** (built-in, 2048-dim) is always available and needs no setup. It's a general-purpose image model — a good baseline, but not trained on histology.")
                    HelpBullet("**Pathology foundation models** are encoders pre-trained on large H&E datasets and usually separate tissue classes much better: **UNI / UNI2** (Mahmood Lab), **Virchow** (Paige), **CTransPath**, and **Phikon** (Owkin). Any of them can be installed as a Core ML extractor.")
                    HelpBullet("To install one: convert the model to an `.mlpackage`, place it with its `.paninextractor.json` descriptor in the app's **Extractors** folder (open via **File ▸ Extractors…**), and rescan. Installed models appear in every extractor picker automatically.")
                    HelpBullet("Note that some models (e.g. UNI, Virchow) are gated — you must request access from their publishers before downloading the weights.")
                    HelpBullet("Every patch and classifier records the extractor identity it was built with, so mismatched feature spaces can't be mixed by accident. To compare extractors, extract the same annotations once per extractor and train one classifier per extractor.")
                }

                HelpSection(title: "Training a Classifier", systemImage: "brain") {
                    HelpBullet("**Analysis ▸ Regression Classifier…** (⌥⌘T) trains a logistic-regression classifier on the bank. Choose which classes to include and optionally train one class against the rest.")
                    HelpBullet("With an **Exclude / null** class in the profile, the train sheet offers **Exclude patches similar to null class** — patches whose feature similarity to a null patch is at/above the chosen cutoff are dropped from training, and the reference is saved on the model so predictions filter the same way.")
                    HelpBullet("**Classify whole annotation (pooled)** pools each annotation's patches into one example and makes a single per-annotation decision — better for lesion-level calls than per-patch voting.")
                    HelpBullet("**Analysis ▸ t-SNE Classifier…** (⌥⌘A) computes a t-SNE embedding of the patch features and derives a nearest-centroid classifier. The interactive plot opens automatically. Filters for white patches and minimum nuclei per patch apply before embedding.")
                    HelpBullet("**Analysis ▸ Extract Geometry Features** and **Train Geometry Model…** build an annotation-level classifier from architectural descriptors (nuclear crowding, luminal complexity, etc.) instead of patch embeddings.")
                    HelpBullet("Review training/validation accuracy and per-class metrics in the **Classifier** tab.")
                    HelpBullet("Save the active model with **Analysis ▸ Save Classifier As…** and reload it later with **Open Classifier…**. t-SNE models keep their plot embedded in the file.")
                }

                HelpSection(title: "Running Predictions", systemImage: "wand.and.stars") {
                    HelpStep(number: 1, text: "Select an annotation in the sidebar to define the region to classify.")
                    HelpStep(number: 2, text: "Choose **Analysis ▸ Classify on Annotation…** (⌥⌘R), set a minimum confidence, and run.")
                    HelpStep(number: 3, text: "Optionally enable **Require at least N nuclei per patch** — sampled patches with too few nuclei are skipped before classification, so lumen, stroma, and empty fields get no verdict and no heatmap cell.")
                    HelpStep(number: 4, text: "Predictions appear as a color-coded heatmap over the slide, with per-class visibility toggles in the Classifier panel. Clear them with **Analysis ▸ Clear Predictions**.")
                    HelpBullet("The **Decision** picker chooses between per-patch verdicts and late-fusion modes that combine all patches into one label for the annotation (mean, or “worst focus”).")
                    HelpBullet("**Analysis ▸ Save Predictions as Annotations…** (⌥⌘S) converts the displayed prediction tiles into persistent annotations.")
                    HelpBullet("**Analysis ▸ Propose & Grade Regions** merges predicted tiles into lesion regions and grades each with the active geometry model.")
                }

                HelpSection(title: "VISTA Tissue Segmentation", systemImage: "square.3.layers.3d") {
                    HelpBullet("**Analysis ▸ Classify with VISTA on Annotation** (⌥⌘V) runs the VISTA tissue segmenter — three published U-Nets that classify mouse-pancreas H&E into **Neoplasia**, **Metaplasia**, **Normal acinar**, and **Stroma** — over the selected annotation.")
                    HelpBullet("Results render as the same tile heatmap as classifier predictions, with per-tile tissue composition. Useful as an independent comparison against your own trained classifier.")
                    HelpBullet("Requires the three VISTA Core ML models installed under the app's Application Support **VISTA** folder; color normalization is applied automatically.")
                }

                HelpSection(title: "Tips", systemImage: "lightbulb") {
                    HelpBullet("A feature extractor's identity is recorded on every patch and model, so mismatched feature spaces can't be mixed accidentally — keep the same extractor across a bank and its classifier.")
                    HelpBullet("Raising the maximum white fraction excludes mostly-background patches from extraction, training, and t-SNE.")
                    HelpBullet("The nucleus filters use the same detector everywhere (extraction, classification, t-SNE) — keep the minimum roughly consistent across stages so the patches you classify resemble the ones you trained on. Patches extracted before nucleus counting existed pass the t-SNE filter until re-extracted.")
                    HelpBullet("Use **File ▸ Close Slide** (⌘W) to flush annotations and return to the empty state; the bank and trained models persist across slides.")
                }
            }
            .padding(32)
            .frame(maxWidth: 720, alignment: .leading)
        }
        .frame(minWidth: 520, minHeight: 480)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("PathLearn Help")
                .font(.largeTitle.bold())
            Text("A macOS tool for viewing whole-slide pathology images, annotating regions of interest, and training machine-learning classifiers on them.")
                .font(.body)
                .foregroundStyle(.secondary)
        }
    }
}

/// A titled group of help content with a leading SF Symbol.
private struct HelpSection<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(title, systemImage: systemImage)
                .font(.title2.bold())
            VStack(alignment: .leading, spacing: 8) {
                content
            }
        }
    }
}

/// A numbered step in an ordered how-to sequence.
private struct HelpStep: View {
    let number: Int
    let text: LocalizedStringKey

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("\(number)")
                .font(.callout.monospacedDigit().bold())
                .foregroundStyle(.white)
                .frame(width: 22, height: 22)
                .background(Circle().fill(Color.accentColor))
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// A single unordered help point.
private struct HelpBullet: View {
    let text: LocalizedStringKey

    init(_ text: LocalizedStringKey) {
        self.text = text
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "circle.fill")
                .font(.system(size: 5))
                .foregroundStyle(.secondary)
                .padding(.top, 7)
            Text(text)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

#Preview {
    HelpView()
}
