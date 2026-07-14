import SwiftUI
import AppKit

struct LabelPickerSheet: View {
    @State private var label: String
    @State private var swiftUIColor: Color
    @State private var saveToProfile: Bool = true
    @State private var showingNewClassSheet: Bool = false

    let existingLabels: [String]
    @Bindable var profileStore: ProfileStore
    let onCancel: () -> Void
    let onCommit: (String, AnnotationColor) -> Void

    init(initialLabel: String,
         initialColor: AnnotationColor,
         existingLabels: [String],
         profileStore: ProfileStore,
         onCancel: @escaping () -> Void,
         onCommit: @escaping (String, AnnotationColor) -> Void) {
        _label = State(initialValue: initialLabel)
        _swiftUIColor = State(initialValue: Color(
            red: Double(initialColor.r) / 255,
            green: Double(initialColor.g) / 255,
            blue: Double(initialColor.b) / 255
        ))
        self.existingLabels = existingLabels
        self.profileStore = profileStore
        self.onCancel = onCancel
        self.onCommit = onCommit
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Label annotation").font(.headline)

            HStack(spacing: 8) {
                TextField("Classification (e.g., PaNIN-1)", text: $label)
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
                Text("Profile classes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(profileStore.profile.classes) { c in
                            classChip(c)
                        }
                    }
                }
            }

            let otherLabels = existingLabels.filter { lbl in
                !profileStore.profile.classes.contains(where: { $0.name == lbl })
            }
            if !otherLabels.isEmpty {
                Text("Other recent")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(otherLabels, id: \.self) { l in
                            Button(l) { label = l }
                                .buttonStyle(.bordered)
                        }
                    }
                }
            }

            // Inline "save to profile" checkbox — only appears when the typed
            // class is non-empty AND not already in the profile.
            if !trimmedLabel.isEmpty, !labelIsInProfile {
                Toggle(
                    "Also save “\(trimmedLabel)” to the profile palette",
                    isOn: $saveToProfile
                )
                .toggleStyle(.checkbox)
                .font(.caption)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmedLabel.isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 440)
        .sheet(isPresented: $showingNewClassSheet) {
            let seed = Classification(
                name: trimmedLabel,
                color: currentColor
            )
            ClassEditorSheet(
                classification: seed,
                isNew: true,
                onCancel: { showingNewClassSheet = false },
                onSave: { newClass in
                    profileStore.addClass(name: newClass.name, color: newClass.color)
                    // Adopt the new class into the label sheet so the user can
                    // immediately save the annotation with it.
                    label = newClass.name
                    swiftUIColor = Color(
                        red: Double(newClass.color.r) / 255,
                        green: Double(newClass.color.g) / 255,
                        blue: Double(newClass.color.b) / 255
                    )
                    showingNewClassSheet = false
                }
            )
        }
    }

    private func classChip(_ c: Classification) -> some View {
        Button {
            label = c.name
            swiftUIColor = c.swiftUIColor
        } label: {
            HStack(spacing: 6) {
                Circle().fill(c.swiftUIColor).frame(width: 10, height: 10)
                Text(c.name)
            }
        }
        .buttonStyle(.bordered)
    }

    private var trimmedLabel: String {
        label.trimmingCharacters(in: .whitespaces)
    }

    private var labelIsInProfile: Bool {
        profileStore.profile.classes.contains(where: { $0.name == trimmedLabel })
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
        let trimmed = trimmedLabel
        let color = currentColor
        if saveToProfile, !labelIsInProfile, !trimmed.isEmpty {
            profileStore.addClass(name: trimmed, color: color)
        }
        onCommit(trimmed, color)
    }
}
