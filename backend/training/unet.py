"""U-Net architecture matching TP4 specifications.

Encoder: 5 blocks with filters 16->32->64->128->256
  Conv2D(3,3) ReLU he_normal padding=same, Dropout, MaxPooling(2,2)
Decoder: 4 symmetric blocks
  Conv2DTranspose, concatenation skip, Dropout
Output: Conv2D(1,1) sigmoid
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_unet(input_shape=(256, 256, 1)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    # Encoder
    c1 = _conv_block(inputs, 16)
    c1 = layers.Dropout(0.1)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = _conv_block(p1, 32)
    c2 = layers.Dropout(0.1)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = _conv_block(p2, 64)
    c3 = layers.Dropout(0.2)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = _conv_block(p3, 128)
    c4 = layers.Dropout(0.2)(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = _conv_block(p4, 256)
    c5 = layers.Dropout(0.3)(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = _conv_block(u6, 128)
    c6 = layers.Dropout(0.2)(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = _conv_block(u7, 64)
    c7 = layers.Dropout(0.2)(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = _conv_block(u8, 32)
    c8 = layers.Dropout(0.1)(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = _conv_block(u9, 16)
    c9 = layers.Dropout(0.1)(c9)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return keras.Model(inputs, outputs, name="unet")


def _conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), activation="relu",
                       kernel_initializer="he_normal", padding="same")(x)
    x = layers.Conv2D(filters, (3, 3), activation="relu",
                       kernel_initializer="he_normal", padding="same")(x)
    return x
