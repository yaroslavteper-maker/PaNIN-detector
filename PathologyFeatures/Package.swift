// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "PathologyFeatures",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "PathologyFeatures",
            targets: ["PathologyFeatures"]
        ),
    ],
    targets: [
        .target(
            name: "PathologyFeatures",
            path: "Sources/PathologyFeatures"
        ),
        .testTarget(
            name: "PathologyFeaturesTests",
            dependencies: ["PathologyFeatures"],
            path: "Tests/PathologyFeaturesTests"
        ),
    ]
)
