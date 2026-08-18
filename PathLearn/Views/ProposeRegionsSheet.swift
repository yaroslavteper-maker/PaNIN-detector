import SwiftUI

/// Lets the user pick which predicted classes from the current per-patch
/// prediction count as candidate PanIN regions. The chosen classes' tiles are
/// merged into lesion polygons and graded by the active geometry model.
struct ProposeRegionsSheet: View {
    let labels: [String]
    let counts: [String: Int]
    let onCancel: () -> Void
    let onRun: (Set<String>) -> Void

    @State private var selected: Set<String>

    init(labels: [String],
         counts: [String: Int],
         onCancel: @escaping () -> Void,
         onRun: @escaping (Set<String>) -> Void) {
        self.labels = labels
        self.counts = counts
        self.onCancel = onCancel
        self.onRun = onRun
        // Default to anything that looks like a PanIN class.
        _selected = State(initialValue: Set(labels.filter { $0.lowercased().contains("panin") }))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Propose & grade PanIN regions").font(.headline)

            Text("Choose which predicted classes to treat as PanIN. Their tiles are merged into lesion regions and each is graded by the active geometry model.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            HStack {
                Text("Predicted classes")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                Button(allSelected ? "Deselect all" : "Select all") {
                    selected = allSelected ? [] : Set(labels)
                }
                .buttonStyle(.borderless)
                .font(.caption2)
            }

            ScrollView {
                VStack(spacing: 2) {
                    ForEach(labels, id: \.self) { label in
                        HStack(spacing: 6) {
                            Toggle("", isOn: Binding(
                                get: { selected.contains(label) },
                                set: { on in
                                    if on { selected.insert(label) } else { selected.remove(label) }
                                }
                            ))
                            .toggleStyle(.checkbox)
                            .labelsHidden()
                            Text(label)
                                .foregroundStyle(selected.contains(label) ? .primary : .secondary)
                            Spacer()
                            Text("\(counts[label] ?? 0)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        .font(.callout)
                        .padding(.vertical, 1)
                    }
                }
            }
            .frame(maxHeight: 240)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                    .keyboardShortcut(.cancelAction)
                Button("Propose & Grade") { onRun(selected) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(selected.isEmpty)
            }
        }
        .padding(20)
        .frame(minWidth: 420)
    }

    private var allSelected: Bool { selected.count == labels.count && !labels.isEmpty }
}
