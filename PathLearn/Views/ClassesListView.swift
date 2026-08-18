import SwiftUI

struct ClassesListView: View {
    @Bindable var profileStore: ProfileStore
    @Bindable var annotationStore: AnnotationStore

    @State private var editing: Classification?
    @State private var showingNew = false
    @State private var pendingDelete: Classification?

    var body: some View {
        VStack(spacing: 0) {
            if profileStore.profile.classes.isEmpty {
                emptyState
            } else {
                list
            }
            Divider()
            footer
        }
        .sheet(item: $editing) { c in
            ClassEditorSheet(
                classification: c,
                isNew: false,
                onCancel: { editing = nil },
                onSave: { updated in
                    profileStore.updateClass(updated)
                    editing = nil
                }
            )
        }
        .sheet(isPresented: $showingNew) {
            ClassEditorSheet(
                classification: Classification(name: "", color: .defaultColor),
                isNew: true,
                onCancel: { showingNew = false },
                onSave: { newClass in
                    profileStore.addClass(newClass)
                    showingNew = false
                }
            )
        }
        .confirmationDialog(
            "Delete class “\(pendingDelete?.name ?? "")”?",
            isPresented: deleteConfirmationBinding,
            titleVisibility: .visible,
            presenting: pendingDelete
        ) { c in
            Button("Delete Class", role: .destructive) {
                profileStore.removeClass(id: c.id)
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: { c in
            let n = count(for: c.name)
            if n > 0 {
                Text("\(n) annotation\(n == 1 ? "" : "s") use this class. They keep their label and color but the class is removed from the profile.")
            } else {
                Text("This removes the class from the profile.")
            }
        }
    }

    private var deleteConfirmationBinding: Binding<Bool> {
        Binding(get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } })
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "tag")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("No classes yet")
                .foregroundStyle(.secondary)
            Text("Tap + to add a class.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var list: some View {
        List {
            ForEach(profileStore.profile.classes) { c in
                HStack(spacing: 8) {
                    Circle()
                        .fill(c.swiftUIColor)
                        .frame(width: 12, height: 12)
                    Text(c.name)
                        .lineLimit(1)
                    Spacer()
                    Text("\(count(for: c.name))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
                .contentShape(Rectangle())
                .contextMenu {
                    Button("Edit…") { editing = c }
                    Button("Apply to selected annotation") {
                        applyToSelectedAnnotation(c)
                    }
                    .disabled(annotationStore.selectedID == nil)
                    Divider()
                    Button("Delete", role: .destructive) {
                        pendingDelete = c
                    }
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) {
                        pendingDelete = c
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
            .onDelete { offsets in
                if let i = offsets.first {
                    pendingDelete = profileStore.profile.classes[i]
                }
            }
        }
        .listStyle(.inset)
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Button {
                showingNew = true
            } label: {
                Image(systemName: "plus.circle")
                    .imageScale(.medium)
            }
            .buttonStyle(.plain)
            .help("Add class")

            Divider().frame(height: 14)

            Image(systemName: "doc.text")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(profileStore.profile.name)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Spacer()

            if let url = profileStore.profileURL {
                Image(systemName: "externaldrive")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help("Saved at: \(url.path)")
            } else {
                Image(systemName: "circle.dashed")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .help("Profile not saved to disk yet. File ▸ Save Profile As…")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func count(for className: String) -> Int {
        annotationStore.annotations.filter { $0.classification == className }.count
    }

    private func applyToSelectedAnnotation(_ c: Classification) {
        guard let id = annotationStore.selectedID else { return }
        annotationStore.setClassification(id, name: c.name, color: c.color)
    }
}

extension Classification {
    var swiftUIColor: Color {
        Color(red: Double(color.r)/255, green: Double(color.g)/255, blue: Double(color.b)/255)
    }
}
