"""
Convert the 3 rebuilt+weight-loaded VISTA UNets to Core ML .mlpackage files.

The model has an internal Lambda(x/255), so the Core ML input is a plain RGB
image with pixel values 0-255 (scale=1, bias=0). Output is a 512x512x1 float
probability map.
"""
import os, argparse
import numpy as np
import coremltools as ct
from vista_model import load_unet, MODELS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreml")


def convert_one(cls, h5, out_dir):
    keras_model = load_unet(h5)
    mlmodel = ct.convert(
        keras_model,
        source="tensorflow",
        convert_to="mlprogram",
        # Plain tensor input (NHWC, raw 0-255 floats). The model keeps its
        # internal /255 Lambda. Feeding a MultiArray from Swift avoids Core ML's
        # image-input color management, which silently gamma-shifted the pixels
        # and produced ~20x-too-low activations (all-stroma bug).
        inputs=[ct.TensorType(name="input_1", shape=(1, 512, 512, 3))],
        minimum_deployment_target=ct.target.macOS13,
    )
    spec = mlmodel.get_spec()
    out_name = spec.description.output[0].name
    ct.utils.rename_feature(spec, "input_1", "image")
    ct.utils.rename_feature(spec, out_name, "mask")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    mlmodel.short_description = f"VISTA/MicePan UNet - {cls} probability map (512x512, sigmoid)"
    mlmodel.author = "gelatinfrogs/MicePan-Segmentation (Nat Sci Rep 2020)"
    path = os.path.join(out_dir, f"VISTA-{cls}.mlpackage")
    mlmodel.save(path)
    print(f"  saved {path}")
    return keras_model, mlmodel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="convert a single class (metaplasia/neoplasia/normal)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = {args.only: MODELS[args.only]} if args.only else MODELS
    for cls, (h5, _thr) in items.items():
        print(f"Converting {cls} ...")
        convert_one(cls, h5, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
