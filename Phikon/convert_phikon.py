"""
Convert Owkin's Phikon (iBOT ViT-B/16 pathology encoder) to a Core ML
.mlpackage that outputs the 768-d CLS embedding.

Input is a plain tensor [1,3,224,224] of ALREADY-normalized pixels (NCHW) —
PathLearn's CoreMLFeatureExtractor applies (pixel/255 - mean)/std itself using
the descriptor, and feeds an MLMultiArray. A tensor input (not an image input)
avoids Core ML's image color management.
"""
import numpy as np
import torch
import coremltools as ct
from transformers import AutoModel

MODEL_ID = "owkin/phikon"


class PhikonEmbed(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).last_hidden_state[:, 0]  # CLS token -> [B, 768]


def main():
    model = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    wrapper = PhikonEmbed(model).eval()

    example = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref = wrapper(example).numpy()
    traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
        minimum_deployment_target=ct.target.macOS13,
    )
    spec = mlmodel.get_spec()
    out_name = spec.description.output[0].name
    ct.utils.rename_feature(spec, out_name, "features")
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)
    mlmodel.short_description = "Phikon (Owkin) iBOT ViT-B/16 pathology encoder — 768-d CLS embedding"
    mlmodel.save("phikon.mlpackage")
    print("saved phikon.mlpackage")

    # Verify Core ML matches PyTorch on the same input.
    cm = mlmodel.predict({"input": example.numpy()})
    cmv = np.array(list(cm.values())[0]).reshape(-1)
    refv = ref.reshape(-1)
    diff = np.abs(cmv - refv)
    print("dim:", refv.shape[0])
    print("max abs diff:", float(diff.max()), "mean abs diff:", float(diff.mean()))
    print("cos sim:", float(np.dot(cmv, refv) / (np.linalg.norm(cmv) * np.linalg.norm(refv))))


if __name__ == "__main__":
    main()
