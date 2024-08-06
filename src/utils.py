from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

OUT_CHANNEL = 1
IMAGE_DIR = "../images/"
LOGS_FIT_DIR = "logs/fit"

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result


# Build the generator
def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 508])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        OUT_CHANNEL,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 508], name="input_image")
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def plot_graph(array_x, array_y, convolved_np, title):
    plt.subplot(1, 4, 1)
    plt.imshow(np.sum(array_x[:, 1:256, 1:256], axis=0))

    plt.subplot(1, 4, 2)
    plt.imshow(np.sum(convolved_np[:, 1:256, 1:256], axis=0))

    plt.subplot(1, 4, 3)
    plt.imshow(convolved_np[0, 1:256, 1:256])

    plt.subplot(1, 4, 4)
    plt.imshow(array_y[1:256, 1:256])

    # Ensure image won't block main thread
    # plt.show()
    plt.savefig(IMAGE_DIR + title + ".png")


def generate_images(model, test_input, tar, image_title):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [tar[0], prediction[0]]
    title = ["Ground Truth", "Predicted Image"]

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")

    # Ensure image won't block main thread
    # plt.show()
    plt.savefig(IMAGE_DIR + image_title + ".png")
