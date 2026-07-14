import XCTest
import CoreGraphics
@testable import PathologyFeatures

final class PathologyFeaturesTests: XCTestCase {

    func testVisionIdentity() {
        let extractor = VisionFeatureExtractor(revision: 1)
        XCTAssertEqual(extractor.identity.kind, "vision")
        XCTAssertEqual(extractor.identity.name, "vision")
        XCTAssertEqual(extractor.identity.revision, 1)
        XCTAssertEqual(extractor.identity.stringForm, "vision:vision:r1")
    }

    func testVisionExtractsFromTinyImage() async throws {
        // Make a tiny 32×32 RGBA bitmap to feed Vision.
        let w = 32, h = 32
        let bytesPerRow = w * 4
        var pixels = [UInt8](repeating: 200, count: h * bytesPerRow)
        // Add some non-uniform structure so Vision has something to embed.
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                pixels[i + 0] = UInt8((x * 8) % 256) // R
                pixels[i + 1] = UInt8((y * 8) % 256) // G
                pixels[i + 2] = 128                  // B
                pixels[i + 3] = 255                  // A
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &pixels,
            width: w, height: h,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let cgImage = ctx.makeImage()!
        let extractor = VisionFeatureExtractor()
        let result = try await extractor.extract(cgImage)
        XCTAssertGreaterThan(result.dim, 0)
        XCTAssertEqual(result.dim, extractor.featureDim)
        XCTAssertGreaterThan(result.data.count, 0)
        XCTAssertTrue(result.elementType == .float32 || result.elementType == .float16)
    }

    func testDescriptorRoundTrip() throws {
        let descriptor = ExtractorDescriptor(
            identity: ExtractorIdentity(kind: "coreml", name: "uni-v1", revision: 1),
            kind: .coreml,
            inputWidth: 224,
            inputHeight: 224,
            featureDim: 1024,
            pixelNormalization: .imageNet,
            modelFilename: "uni.mlpackage"
        )
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("descriptor-test-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try descriptor.save(to: tmp)
        let loaded = try ExtractorDescriptor.load(from: tmp)
        XCTAssertEqual(loaded.identity, descriptor.identity)
        XCTAssertEqual(loaded.inputWidth, 224)
        XCTAssertEqual(loaded.featureDim, 1024)
        XCTAssertEqual(loaded.modelFilename, "uni.mlpackage")
        XCTAssertEqual(loaded.pixelNormalization.meanRGB.count, 3)
    }
}
