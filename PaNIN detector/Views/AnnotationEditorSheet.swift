import SwiftUI
import AppKit

struct AnnotationEditorSheet: View {
    let annotation: Annotation
    let existingLabels: [String]
    @Bindable var profileStore: ProfileStore
    let onCancel: () -> Void
    let onSave: (Annotation) -> Void

    @State private var name: String
    @State private var classification: String
    @State private var swiftUIColor: Color
    @State private var isVisible: Bool
    @State private var saveToProfile: Bool = true
    @State private var showingNewClassSheet: Bool = false

    init(annotation: Annotation,
         existingLabels: [String],
         profileStore: ProfileStore,
         onCancel: @escaping () -> Void,
         onSave: @escaping (Annotation) -> Void) {
        self.annotation = annotation
        self.existingLabels = existingLabels
        self.profileStore = profileStore
        self.onCancel = onCancel
        self.onSave = onSave
        _name = State(initialValue: annotation.name ?? "")
        _classification = State(initialValue: annotation.classification)
        _swiftUIColor = State(initialValue: Color(
            red: Double(annotation.color.r)/255,
            green: Double(annotation.color.g)/255,
            blue: Double(annotation.color.b)/255
        ))
        _isVisible = State(initialValue: annotation.isVisible)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Edit annotation").font(.headline)

            HStack(spacing: 8) {
                Text("Name:")
                    .frame(width: 64, alignment: .trailing)
                TextField("(optional)", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            HStack(spacing: 8) {
                Text("Class:")
                    .frame(width: 64, alignment: .trailing)
                TextField("Classification", text: $classification)
                    .textFieldStyle(.roundedBorder)
                ColorPicker("", selection: $swiftUIColor, supportsOpacity: false)
                    .labelsHidden()
                    .frame(width: 44)
                Button {
                    showingNewClassSheet = true
                } label: {
                    Image(systemName: "plus.circle")
                }
                .buttonStyle(.borderless)
                .help("Create a new class with color, add to the profile palette")
            }

            if !profileStore.profile.classes.isEmpty {
                HStack(alignment: .top) {
                    Text("Profile:")
                        .frame(width: 64, alignment: .trailing)
                        .foregroundStyle(.secondary)
                        .font(.caption)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(profileStore.profile.classes) { c in
                                Button {
                                    classification = c.name
                                    swiftUIColor = c.swiftUIColor
                                } label: {
                                    HStack(spacing: 6) {
                                        Circle().fill(c.swiftUIColor).frame(width: 10, height: 10)
                                        Text(c.name)
                                    }
                                }
                                .buttonStyle(.bordered)
                            }
                        }
                    }
                }
            }

            if !trimmedClass.isEmpty, !classIsInProfile {
                HStack {
                    Spacer().frame(width: 64)
                    Toggle(
                        "Also save “\(trimmedClass)” to the profile palette",
                        isOn: $saveToProfile
                    )
                    .toggleStyle(.checkbox)
                    .font(.caption)
                    Spacer()
                }
            }

            Toggle("Visible on canvas", isOn: $isVisible)
                .padding(.leading, 64)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmedClass.isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 480)
        .sheet(isPresented: $showingNewClassSheet) {
            let seed = Classification(
                name: trimmedClass,
                color: currentColor
            )
            ClassEditorSheet(
                classification: seed,
                isNew: true,
                onCancel: { showingNewClassSheet = false },
                onSave: { newClass in
                    profileStore.addClass(name: newClass.name, color: newClass.color)
                    classification = newClass.name
                    swiftUIColor = Color(
                        red: Double(newClass.color.r)/255,
                        green: Double(newClass.color.g)/255,
                        blue: Double(newClass.color.b)/255
                    )
                    showingNewClassSheet = false
                }
            )
        }
    }

    private var trimmedClass: String {
        classification.trimmingCharacters(in: .whitespaces)
    }

    private var classIsInProfile: Bool {
        profileStore.profile.classes.contains(where: { $0.name == trimmedClass })
    }

    private var currentColor: AnnotationColor {
        let ns = NSColor(swiftUIColor).usingColorSpace(.sRGB) ?? .red
        return AnnotationColor(
            r: Int(round(ns.redComponent * 255)),
            g: Int(round(ns.greenComponent * 255)),
            b: Int(round(ns.blueComponent * 255))
        )
    }

    private func commit() {
        let trimmed = trimmedClass
        let color = currentColor
        if saveToProfile, !classIsInProfile, !trimmed.isEmpty {
            profileStore.addClass(name: trimmed, color: color)
        }
        var updated = annotation
        updated.name = name.trimmingCharacters(in: .whitespaces).isEmpty ? nil : name
        updated.classification = trimmed
        updated.color = color
        updated.isVisible = isVisible
        onSave(updated)
    }
}
