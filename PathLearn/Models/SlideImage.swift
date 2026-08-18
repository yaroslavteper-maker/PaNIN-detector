import Foundation
import CoreGraphics
import COpenSlide

/// Thin Swift wrapper around an OpenSlide handle.
/// Marked `nonisolated` and `@unchecked Sendable` because OpenSlide's read
/// functions are thread-safe on a single handle, and CATiledLayer drives tile
/// drawing from a background queue.
nonisolated final class SlideImage: @unchecked Sendable {
    enum SlideError: Error, CustomStringConvertible {
        case unsupportedFormat
        case openSlideError(String)
        var description: String {
            switch self {
            case .unsupportedFormat: return "Not a recognised whole-slide image."
            case .openSlideError(let m): return "OpenSlide error: \(m)"
            }
        }
    }

    let url: URL
    let dimensions: CGSize          // base level
    let levelCount: Int32
    let levelDownsamples: [Double]
    let levelDimensions: [CGSize]
    let properties: [String: String]

    private let handle: OpaquePointer

    init(url: URL) throws {
        let path = url.path(percentEncoded: false)
        guard let h = openslide_open(path) else {
            throw SlideError.unsupportedFormat
        }
        if let err = openslide_get_error(h) {
            let msg = String(cString: err)
            openslide_close(h)
            throw SlideError.openSlideError(msg)
        }
        self.handle = h
        self.url = url

        let n = openslide_get_level_count(h)
        self.levelCount = n

        var w0: Int64 = 0, h0: Int64 = 0
        openslide_get_level0_dimensions(h, &w0, &h0)
        self.dimensions = CGSize(width: Double(w0), height: Double(h0))

        var downs: [Double] = []
        var dims: [CGSize] = []
        for i in 0..<n {
            downs.append(openslide_get_level_downsample(h, i))
            var lw: Int64 = 0, lh: Int64 = 0
            openslide_get_level_dimensions(h, i, &lw, &lh)
            dims.append(CGSize(width: Double(lw), height: Double(lh)))
        }
        self.levelDownsamples = downs
        self.levelDimensions = dims

        var props: [String: String] = [:]
        if let names = openslide_get_property_names(h) {
            var i = 0
            while let cstr = names[i] {
                let key = String(cString: cstr)
                if let v = openslide_get_property_value(h, cstr) {
                    props[key] = String(cString: v)
                }
                i += 1
            }
        }
        self.properties = props
    }

    deinit { openslide_close(handle) }

    func bestLevel(forDownsample d: Double) -> Int32 {
        let i = openslide_get_best_level_for_downsample(handle, d)
        return max(0, min(i, levelCount - 1))
    }

    /// `x`, `y` are in **level-0** coordinates. `width`, `height` are in
    /// the **target level's** coordinates. Returned image is sRGB BGRA premultiplied.
    func readRegion(x: Int64, y: Int64, level: Int32, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0 else { return nil }
        let count = width * height
        let bytes = count * 4
        let buf = UnsafeMutablePointer<UInt32>.allocate(capacity: count)
        openslide_read_region(handle, buf, x, y, level, Int64(width), Int64(height))
        if let err = openslide_get_error(handle) {
            let msg = String(cString: err)
            print("openslide_read_region error:", msg)
            buf.deallocate()
            return nil
        }
        let release: CGDataProviderReleaseDataCallback = { _, dataPtr, _ in
            UnsafeMutableRawPointer(mutating: dataPtr).deallocate()
        }
        guard let provider = CGDataProvider(dataInfo: nil, data: buf, size: bytes, releaseData: release) else {
            buf.deallocate()
            return nil
        }
        let bmi: CGBitmapInfo = [
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            .byteOrder32Little
        ]
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bmi,
            provider: provider, decode: nil,
            shouldInterpolate: true, intent: .defaultIntent
        )
    }
}
