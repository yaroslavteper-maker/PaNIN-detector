import Foundation
import Observation

@Observable
final class ProfileStore {
    var profile: ClassificationProfile
    private(set) var profileURL: URL?

    private static let defaultsKey = "PaNINDetector.profile"
    private static let lastURLKey  = "PaNINDetector.lastProfileURL"

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.defaultsKey),
           let decoded = try? JSONDecoder().decode(ClassificationProfile.self, from: data) {
            self.profile = decoded
        } else {
            self.profile = .default
        }
        if let path = UserDefaults.standard.string(forKey: Self.lastURLKey),
           FileManager.default.fileExists(atPath: path) {
            self.profileURL = URL(fileURLWithPath: path)
        }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(profile) {
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        }
    }

    // MARK: Mutations

    func addClass(name: String, color: AnnotationColor) {
        profile.classes.append(Classification(name: name, color: color))
        persist()
    }

    func updateClass(_ c: Classification) {
        guard let i = profile.classes.firstIndex(where: { $0.id == c.id }) else { return }
        profile.classes[i] = c
        persist()
    }

    func removeClass(id: UUID) {
        profile.classes.removeAll { $0.id == id }
        persist()
    }

    func setProfileName(_ name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        profile.name = trimmed.isEmpty ? "Untitled" : trimmed
        persist()
    }

    // MARK: File I/O

    func loadFromFile(_ url: URL) throws {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode(ClassificationProfile.self, from: data)
        profile = decoded
        profileURL = url
        UserDefaults.standard.set(url.path, forKey: Self.lastURLKey)
        persist()
    }

    func saveToFile(_ url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(profile)
        try data.write(to: url, options: [.atomic])
        profileURL = url
        UserDefaults.standard.set(url.path, forKey: Self.lastURLKey)
    }
}
