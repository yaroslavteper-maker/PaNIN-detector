import SwiftUI
import AppKit

struct ClassEditorSheet: View {
    let initialClass: Classification
    let isNew: Bool
    let onCancel: () -> Void
    let onSave: (Classification) -> Void

    @State private var name: String
    @State private var color: Color

    init(classification: Classification,
         isNew: Bool = false,
         onCancel: @escaping () -> Void,
         onSave: @escaping (Classification) -> Void) {
        self.initialClass = classification
        self.isNew = isNew
        self.onCancel = onCancel
        self.onSave = onSave
        _name = State(initialValue: classification.name)
        _color = State(initialValue: Color(
            red: Double(classification.color.r)/255,
            green: Double(classification.color.g)/255,
            blue: Double(classification.color.b)/255
        ))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(isNew ? "New class" : "Edit class")
                .font(.headline)

            HStack(spacing: 8) {
                TextField("Name (e.g., PaNIN-1)", text: $name)
                    .textFieldStyle(.roundedBorder)
                ColorPicker("", selection: $color, supportsOpacity: false)
                    .labelsHidden()
                    .frame(width: 44)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { commit() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 360)
    }

    private func commit() {
        let ns = NSColor(color).usingColorSpace(.sRGB) ?? .red
        let ac = AnnotationColor(
            r: Int(round(ns.redComponent * 255)),
            g: Int(round(ns.greenComponent * 255)),
            b: Int(round(ns.blueComponent * 255))
        )
        var updated = initialClass
        updated.name = name.trimmingCharacters(in: .whitespaces)
        updated.color = ac
        onSave(updated)
    }
}
