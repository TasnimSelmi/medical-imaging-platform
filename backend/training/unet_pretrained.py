"""U-Net with an ImageNet-pretrained MobileNetV2 encoder (transfer learning).

Encoder: MobileNetV2 (keras.applications, weights="imagenet"); its BatchNorm layers
  stay non-trainable so their statistics stay frozen, including during MC Dropout.
Skips: block_1/3/6/13 expand_relu (128, 64, 32, 16 px) + block_16_project (8 px)
Decoder: 5 blocks Conv2DTranspose + concatenation skip + 2x Conv2D ReLU + Dropout
Output: Conv2D(1,1) sigmoid
Input: RGB in [0, 1], rescaled internally to [-1, 1] as MobileNetV2 expects.
"""

from tensorflow import keras
from tensorflow.keras import layers

SKIP_LAYERS = [
    "block_1_expand_relu",   # 128x128
    "block_3_expand_relu",   # 64x64
    "block_6_expand_relu",   # 32x32
    "block_13_expand_relu",  # 16x16
    "block_16_project",      # 8x8 (bottleneck)
]


def build_pretrained_unet(input_shape=(256, 256, 3)) -> keras.Model:
    backbone = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    encoder = keras.Model(backbone.input, [backbone.get_layer(n).output for n in SKIP_LAYERS], name="encoder")

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    s1, s2, s3, s4, x = encoder(x, training=False)

    for skip, filters, rate in ((s4, 256, 0.3), (s3, 128, 0.2), (s2, 64, 0.2), (s1, 32, 0.1)):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, filters)
        x = layers.Dropout(rate)(x)

    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(x)
    x = _conv_block(x, 16)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="unet_mobilenetv2")
    set_encoder_trainable(model, False)
    return model


def set_encoder_trainable(model: keras.Model, trainable: bool) -> None:
    encoder = model.get_layer("encoder")
    encoder.trainable = trainable
    # A non-trainable BatchNorm always runs in inference mode, so its ImageNet
    # statistics survive fine-tuning and MC Dropout calls with training=True.
    for layer in encoder.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def _conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation="relu",
                      kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), activation="relu",
                      kernel_initializer="he_normal", padding="same")(x)
    return x
