import SwiftUI

/// Sidebar panel for the handcrafted-geometry feature bank: per-class and
/// per-slide counts, live extract/train progress, and actions to extract on the
/// current slide, train a geometry model, reveal the JSON bank, or clear it.
struct GeometryBankPanel: View {
    @Bindable var store: GeometryFeatureStore

    @State private var showingClearConfirmation = false
    @State private var pendingClassDelete: String?

    var body: some View {
        VStack(spacing: 0) {
            if store.isExtracting {
                progressBar(label: "Extracting geometry…",
                            current: store.extractCurrent, total: store.extractTotal)
                Divider()
            } else if store.isTraining {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(store.trainingStatus.isEmpty ? "Training…" : store.trainingStatus)
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 10).padding(.vertical, 6)
                Divider()
            }

            if store.staleCount > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text("\(store.staleCount) record(s) use an old descriptor version. Clear the bank and re-extract before training.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 10).padding(.vertical, 6)
                Divider()
            }

            if store.total == 0 {
                emptyState
            } else {
                list
            }
            Divider()
            actions
            Divider()
            footer
        }
        .confirmationDialog(
            "Clear the geometry bank?",
            isPresented: $showingClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear all \(store.total) records", role: .destructive) { store.clear() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This deletes every geometry descriptor. Annotations themselves are not touched.")
        }
        .confirmationDialog(
            classDeletePrompt,
            isPresented: Binding(
                get: { pendingClassDelete != nil },
                set: { if !$0 { pendingClassDelete = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let label = pendingClassDelete {
                Button("Delete \(store.perClass[label] ?? 0) records", role: .destructive) {
                    store.removeClass(label)
                    pendingClassDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingClassDelete = nil }
            }
        } message: {
            Text("Removes this class's geometry records. Annotations themselves are not touched.")
        }
    }

    private var classDeletePrompt: String {
        if let name = pendingClassDelete { return "Delete all “\(name)” geometry records?" }
        return "Delete records?"
    }

    private func progressBar(label: String, current: Int, total: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(label) \(current)/\(total)")
                .font(.caption).foregroundStyle(.secondary)
            ProgressView(value: total > 0 ? Double(current) / Double(total) : 0)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "square.on.square.dashed")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("Geometry bank is empty")
                .foregroundStyle(.secondary)
            Text("Open an annotated slide, then Extract Geometry below. The bank accumulates across slides.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var list: some View {
        List {
            Section("By class") {
                ForEach(sortedClasses, id: \.key) { entry in
                    HStack {
                        Text(entry.key)
                        Spacer()
                        Text("\(entry.value)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(entry.value < 2 ? .orange : .secondary)
                            .help(entry.value < 2 ? "Needs ≥2 annotations to train this class." : "")
                    }
                    .contentShape(Rectangle())
                    .contextMenu {
                        Button("Delete \(entry.value) “\(entry.key)” records", role: .destructive) {
                            pendingClassDelete = entry.key
                        }
                    }
                }
            }
            if store.perSlide.count >= 1 {
                Section("By slide") {
                    ForEach(sortedSlides, id: \.key) { entry in
                        HStack {
                            Text(entry.key).lineLimit(1)
                            Spacer()
                            Text("\(entry.value)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .listStyle(.inset)
    }

    private var actions: some View {
        HStack(spacing: 8) {
            Button {
                NotificationCenter.default.post(name: .extractGeometryRequested, object: nil)
            } label: {
                Label("Extract", systemImage: "square.on.square.dashed")
            }
            .help("Compute geometry for every annotation on the open slide")

            Button {
                NotificationCenter.default.post(name: .trainGeometryModelRequested, object: nil)
            } label: {
                Label("Train", systemImage: "brain")
            }
            .disabled(store.total < 4 || store.isExtracting || store.isTraining)
            .help("Train a geometry classifier from the bank")
            Spacer()
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Image(systemName: "square.on.square.dashed")
                .foregroundStyle(.secondary)
            Text("\(store.total) record\(store.total == 1 ? "" : "s")")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                if let url = store.bankURL {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                }
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.plain)
            .disabled(store.bankURL == nil)
            .help("Reveal geometry bank in Finder")

            Button {
                store.reload()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.plain)
            .help("Reload from disk")

            Button(role: .destructive) {
                showingClearConfirmation = true
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .disabled(store.total == 0)
            .help("Clear geometry bank")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var sortedClasses: [(key: String, value: Int)] {
        store.perClass.sorted { $0.key < $1.key }
    }

    private var sortedSlides: [(key: String, value: Int)] {
        store.perSlide.sorted { $0.key < $1.key }
    }
}
