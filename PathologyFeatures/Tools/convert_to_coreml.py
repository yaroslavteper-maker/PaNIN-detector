#!/usr/bin/env python3
"""
Convert a pretrained pathology image encoder into a Core ML .mlpackage plus
a matching ExtractorDescriptor sidecar JSON for PathologyFeatures.

Usage:
    python3 Tools/convert_to_coreml.py --model uni --output /path/to/output_dir
    python3 Tools/convert_to_coreml.py --model phikon --output /path/to/output_dir

Requirements:
    pip install torch coremltools transformers huggingface_hub timm

Gating:
    - UNI / Virchow are gated. Request access on HuggingFace, then run
      `huggingface-cli login` once.
    - Phikon / Phikon-v2 are non-gated as of writing — no login needed.

This is a reference script — not run by the Swift package build. The
output is two files:
    output_dir/uni.mlpackage
    output_dir/uni-v1.paninextractor.json
"""

import argparse
import json
from pathlib import Path

import torch
import coremltools as ct

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _download_weights(repo_id):
    """Try common HF weight filenames in priority order so the loader works
    whether the repo ships `model.safetensors` (newer) or `pytorch_model.bin`
    (older). Returns the loaded state_dict."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    candidates = ["model.safetensors", "pytorch_model.bin"]
    last_exc = None
    for name in candidates:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=name)
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(path)
            return torch.load(path, map_location="cpu")
        except (EntryNotFoundError, LocalEntryNotFoundError) as e:
            last_exc = e
            continue
    raise FileNotFoundError(
        f"Could not find weight file in {repo_id}. Tried {candidates}. "
        f"Check the model card for the actual filename. Last error: {last_exc}"
    )

def load_uni():
    """UNI from MahmoodLab. https://huggingface.co/MahmoodLab/UNI"""
    import timm
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=False,
    )
    state = _download_weights("MahmoodLab/UNI")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(name="uni-v1", input_size=224, feature_dim=1024)

def load_uni2():
    """UNI2-h from MahmoodLab. https://huggingface.co/MahmoodLab/UNI2-h
    DINOv2-style ViT-Huge with register tokens + SwiGLU MLP.
    If load_state_dict raises shape/key errors, paste the exact
    timm.create_model(...) kwargs from the UNI2-h model card's
    "Direct use" section over the block below."""
    import timm
    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        img_size=224,
        patch_size=14,
        init_values=1e-5,
        embed_dim=1536,
        depth=24,
        num_heads=24,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=False,   # we always feed fixed 224×224 — dynamic_img_size
                                  # triggers pos_embed `int()` ops that current
                                  # coremltools 7.x can't lower to MIL.
    )
    state = _download_weights("MahmoodLab/UNI2-h")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(name="uni2-h", input_size=224, feature_dim=1536)

def load_virchow():
    """Virchow from Paige.AI. https://huggingface.co/paige-ai/Virchow"""
    import timm
    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
    )
    state = _download_weights("paige-ai/Virchow")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, dict(name="virchow-v1", input_size=224, feature_dim=2560)

class _HFViTClsTokenWrapper(torch.nn.Module):
    """Wrap a HuggingFace ViT-style model to expose the [CLS] feature only.
    Tracing-friendly: takes a single positional pixel-values tensor and
    returns the [CLS] token vector from last_hidden_state."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        out = self.base(pixel_values=x)
        return out.last_hidden_state[:, 0]

def load_phikon():
    """Phikon (v1) from Owkin. https://huggingface.co/owkin/phikon
    ViT-Base/16, 768-dim [CLS] features. Non-gated as of writing."""
    from transformers import AutoModel
    base = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = _HFViTClsTokenWrapper(base)
    model.eval()
    return model, dict(name="phikon-v1", input_size=224, feature_dim=768)

def load_phikon_v2():
    """Phikon-v2 from Owkin. https://huggingface.co/owkin/phikon-v2
    ViT-Large/16; feature_dim is auto-detected at trace time so the value
    below is just a hint."""
    from transformers import AutoModel
    base = AutoModel.from_pretrained("owkin/phikon-v2", add_pooling_layer=False)
    model = _HFViTClsTokenWrapper(base)
    model.eval()
    return model, dict(name="phikon-v2", input_size=224, feature_dim=1024)

MODELS = {
    "uni": load_uni,
    "uni2": load_uni2,
    "virchow": load_virchow,
    "phikon": load_phikon,
    "phikon-v2": load_phikon_v2,
}

# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert(model_name: str, output_dir: Path):
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model '{model_name}'. Choices: {list(MODELS)}")

    print(f"==> Loading {model_name}…")
    model, meta = MODELS[model_name]()
    input_size = meta["input_size"]
    feature_dim = meta["feature_dim"]

    # Trace
    print("==> Tracing…")
    example_input = torch.randn(1, 3, input_size, input_size)
    traced = torch.jit.trace(model, example_input)
    with torch.no_grad():
        output = traced(example_input)
    actual_dim = output.shape[-1]
    if actual_dim != feature_dim:
        print(f"   (note: actual feature_dim = {actual_dim}, overriding meta)")
        feature_dim = actual_dim

    # Convert
    print("==> Converting with coremltools…")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example_input.shape)],
        outputs=[ct.TensorType(name="features")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlpackage_path = output_dir / f"{model_name}.mlpackage"
    mlmodel.save(str(mlpackage_path))
    print(f"   wrote {mlpackage_path}")

    # Descriptor sidecar
    descriptor = {
        "identity": {"kind": "coreml", "name": meta["name"], "revision": 1},
        "kind": "coreml",
        "inputWidth": input_size,
        "inputHeight": input_size,
        "featureDim": feature_dim,
        "pixelNormalization": {
            "meanRGB": [0.485, 0.456, 0.406],
            "stdRGB":  [0.229, 0.224, 0.225],
            "scale":    1.0 / 255.0,
        },
        "modelFilename": f"{model_name}.mlpackage",
    }
    descriptor_path = output_dir / f"{meta['name']}.paninextractor.json"
    descriptor_path.write_text(json.dumps(descriptor, indent=2, sort_keys=True))
    print(f"   wrote {descriptor_path}")
    print("Done. Load the descriptor JSON from PathologyFeatures:")
    print(f"   ExtractorRegistry.load(descriptor: URL(fileURLWithPath: \"{descriptor_path}\"))")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS),
                        help="Which pretrained encoder to convert.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Directory to write the .mlpackage and descriptor JSON.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    convert(args.model, args.output)

if __name__ == "__main__":
    main()
