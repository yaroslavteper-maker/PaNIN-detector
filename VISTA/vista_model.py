"""
Rebuilds the MicePan / "VISTA" UNet architecture and loads the published .h5
weights by name.

Why rebuild instead of load_model(): the released .h5 files embed a Keras
`Lambda` layer whose function is stored as Python-3.7 marshaled bytecode, which
cannot be unmarshaled under Python 3.11 (EOFError in marshal.loads). Since only
the Conv2D / Conv2DTranspose layers carry weights, we reconstruct the exact
topology with matching layer names and use load_weights(by_name=True). The
Lambda (input /255 rescale) is reproduced explicitly.

Architecture confirmed by dumping model_config from the .h5:
  Input 512x512x3 -> Lambda(x/255)
  enc: (8,8) (16,16) (32,32) (64,64) each followed by 2x2 maxpool
  bottleneck: (128,128)
  dec: convT+concat -> (64,64) (32,32) (16,16) (8,8)
  out: Conv2D(1, 1x1, sigmoid)
All convs are 3x3, padding='same', relu (output excepted).
"""
import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate,
)
from tensorflow.keras.models import Model


def build_unet():
    inp = Input((512, 512, 3), name="input_1")
    s = Lambda(lambda x: x / 255.0, name="lambda_1")(inp)

    c1 = Conv2D(8, 3, activation="relu", padding="same", name="conv2d_1")(s)
    c1 = Conv2D(8, 3, activation="relu", padding="same", name="conv2d_2")(c1)
    p1 = MaxPooling2D(2, name="max_pooling2d_1")(Dropout(0.2)(c1))

    c2 = Conv2D(16, 3, activation="relu", padding="same", name="conv2d_3")(p1)
    c2 = Conv2D(16, 3, activation="relu", padding="same", name="conv2d_4")(c2)
    p2 = MaxPooling2D(2, name="max_pooling2d_2")(Dropout(0.1)(c2))

    c3 = Conv2D(32, 3, activation="relu", padding="same", name="conv2d_5")(p2)
    c3 = Conv2D(32, 3, activation="relu", padding="same", name="conv2d_6")(c3)
    p3 = MaxPooling2D(2, name="max_pooling2d_3")(c3)

    c4 = Conv2D(64, 3, activation="relu", padding="same", name="conv2d_7")(p3)
    c4 = Conv2D(64, 3, activation="relu", padding="same", name="conv2d_8")(c4)
    p4 = MaxPooling2D(2, name="max_pooling2d_4")(c4)

    c5 = Conv2D(128, 3, activation="relu", padding="same", name="conv2d_9")(p4)
    c5 = Conv2D(128, 3, activation="relu", padding="same", name="conv2d_10")(c5)

    u6 = Conv2DTranspose(64, 2, strides=2, padding="same", name="conv2d_transpose_1")(c5)
    u6 = concatenate([u6, c4], name="concatenate_1")
    c6 = Conv2D(64, 3, activation="relu", padding="same", name="conv2d_11")(u6)
    c6 = Conv2D(64, 3, activation="relu", padding="same", name="conv2d_12")(c6)

    u7 = Conv2DTranspose(32, 2, strides=2, padding="same", name="conv2d_transpose_2")(c6)
    u7 = concatenate([u7, c3], name="concatenate_2")
    c7 = Conv2D(32, 3, activation="relu", padding="same", name="conv2d_13")(u7)
    c7 = Conv2D(32, 3, activation="relu", padding="same", name="conv2d_14")(c7)

    u8 = Conv2DTranspose(16, 2, strides=2, padding="same", name="conv2d_transpose_3")(c7)
    u8 = concatenate([u8, c2], name="concatenate_3")
    c8 = Conv2D(16, 3, activation="relu", padding="same", name="conv2d_15")(u8)
    c8 = Conv2D(16, 3, activation="relu", padding="same", name="conv2d_16")(c8)

    u9 = Conv2DTranspose(8, 2, strides=2, padding="same", name="conv2d_transpose_4")(c8)
    u9 = concatenate([u9, c1], name="concatenate_4")
    c9 = Conv2D(8, 3, activation="relu", padding="same", name="conv2d_17")(u9)
    c9 = Conv2D(8, 3, activation="relu", padding="same", name="conv2d_18")(c9)

    out = Conv2D(1, 1, activation="sigmoid", name="conv2d_19")(c9)
    return Model(inp, out)


# The published weights live in the cloned repo alongside this file.
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MicePan-Segmentation")


def load_unet(h5_path):
    """Build the UNet and load published weights by name. Returns the model.

    Bare filenames are resolved against the cloned repo's WEIGHTS_DIR.
    """
    if not os.path.isabs(h5_path) and not os.path.exists(h5_path):
        h5_path = os.path.join(WEIGHTS_DIR, h5_path)
    model = build_unet()
    model.load_weights(h5_path, by_name=True)
    return model


# Maps the three released weight files to their tissue class + published threshold.
MODELS = {
    "metaplasia": ("MicePan-ADM-512-2Tone-T2-10Normal-2Ductal-Ep50-B32-L7E4.h5", 0.5),
    "neoplasia":  ("MicePan-Ductal-512-2Tone-T2-5ADM-5Normal-Ep50-B32-L7E4.h5", 0.7),
    "normal":     ("MicePan-Normal-512-2Tone-T2-10ADM-2Ductal-Ep50-B32-L7E4.h5", 0.3),
}
