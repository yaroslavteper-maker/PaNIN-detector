import SwiftUI
import SwiftData

struct MLBankPanel: View {
    @Bindable var bankStore: MLBankStore
    let modelContext: ModelContext

    @State private var showingClearConfirmation = false
    @State private var pendingClassDelete: String?
    @State private var pendingSlideDelete: String?

    var body: some View {
        VStack(spacing: 0) {
            if bankStore.totalPatches == 0 {
                emptyState
            } else {
                list
            }
            Divider()
            footer
        }
        .onAppear {
            bankStore.refresh(context: modelContext)
        }
        .confirmationDialog(
            "Clear the feature bank?",
            isPresented: $showingClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear all \(bankStore.totalPatches) patches", role: .destructive) {
                bankStore.clearAll(context: modelContext)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This deletes every feature vector. Annotations themselves are not touched.")
        }
        .confirmationDialog(
            classDeletePrompt,
            isPresented: Binding(
                get: { pendingClassDelete != nil },
                set: { if !$0 { pendingClassDelete = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let className = pendingClassDelete {
                Button("Delete \(bankStore.perClass[className] ?? 0) patches", role: .destructive) {
                    bankStore.removePatches(forClass: className, context: modelContext)
                    pendingClassDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingClassDelete = nil }
            }
        }
        .confirmationDialog(
            slideDeletePrompt,
            isPresented: Binding(
                get: { pendingSlideDelete != nil },
                set: { if !$0 { pendingSlideDelete = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let slideName = pendingSlideDelete {
                Button("Delete \(bankStore.perSlide[slideName] ?? 0) patches", role: .destructive) {
                    bankStore.removePatches(forSlide: slideName, context: modelContext)
                    pendingSlideDelete = nil
                }
                Button("Cancel", role: .cancel) { pendingSlideDelete = nil }
            }
        }
    }

    private var classDeletePrompt: String {
        if let name = pendingClassDelete {
            return "Delete all patches labelled “\(name)”?"
        }
        return "Delete patches?"
    }

    private var slideDeletePrompt: String {
        if let name = pendingSlideDelete {
            return "Delete all patches from “\(name)”?"
        }
        return "Delete patches?"
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "cube.box")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("Bank is empty")
                .foregroundStyle(.secondary)
            Text("Extract patches via ML ▸ Extract Patches… (⇧⌘X)")
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
                            .foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                    .contextMenu {
                        Button("Delete \(entry.value) patches", role: .destructive) {
                            pendingClassDelete = entry.key
                        }
                    }
                }
            }
            if !bankStore.perExtractor.isEmpty {
                Section("By extractor") {
                    ForEach(sortedExtractors, id: \.key) { entry in
                        HStack {
                            Text(entry.key).lineLimit(1)
                                .font(.caption.monospaced())
                            Spacer()
                            Text("\(entry.value)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            if bankStore.perSlide.count >= 1 {
                Section("By slide") {
                    ForEach(sortedSlides, id: \.key) { entry in
                        HStack {
                            Text(entry.key).lineLimit(1)
                            Spacer()
                            Text("\(entry.value)")
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                        .contextMenu {
                            Button("Delete \(entry.value) patches", role: .destructive) {
                                pendingSlideDelete = entry.key
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.inset)
    }

    private var footer: some View {
        HStack(spacing: 6) {
            Image(systemName: "cube.box")
                .foregroundStyle(.secondary)
            Text("\(bankStore.totalPatches) patches")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                NotificationCenter.default.post(name: .showExtractorsRequested, object: nil)
            } label: {
                Image(systemName: "cpu")
            }
            .buttonStyle(.plain)
            .help("Manage feature extractors…")

            Button {
                bankStore.refresh(context: modelContext)
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.plain)
            .help("Refresh stats")

            Button(role: .destructive) {
                showingClearConfirmation = true
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.plain)
            .disabled(bankStore.totalPatches == 0)
            .help("Clear bank")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private var sortedClasses: [(key: String, value: Int)] {
        bankStore.perClass.sorted { $0.key < $1.key }
    }

    private var sortedSlides: [(key: String, value: Int)] {
        bankStore.perSlide.sorted { $0.key < $1.key }
    }

    private var sortedExtractors: [(key: String, value: Int)] {
        bankStore.perExtractor.sorted { $0.key < $1.key }
    }
}
