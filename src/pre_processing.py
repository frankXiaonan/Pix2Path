import numpy as np
import spatialdata_io as sd_io
import tensorflow as tf
from scipy.ndimage import convolve
from spatialdata import bounding_box_query

from utils import plot_graph

MIN_COORDINATE = [3200, 3200]
MAX_COORDINATE = [4800, 4800]
DATA_PATH = "../data/Xenium_V1_FFPE_TgCRND8_17_9_months_outs"
PATCH_SIZE = 256
BATCH_SIZE = 8
TRAIN_STEP_X = [0, 2, 4]
TEST_STEP_X = [1, 3, 5]
TRAIN_STEP_Y = [0, 2, 4]
TEST_STEP_Y = [1, 3, 5]


def pre_process_spt(so_data, scale=4.70588235, minvx=1200, minvy=1200):
    genes = so_data["transcripts"]["feature_name"].unique()
    num_genes = len(genes)
    gene_to_index = {gene: i for i, gene in enumerate(genes)}

    # Define the dimensions of the image
    width = int(so_data["transcripts"]["x"].max().compute() * scale) + 10
    height = int(so_data["transcripts"]["y"].max().compute() * scale) + 10

    # create an empty array to hold the data
    array = np.zeros((num_genes, height - minvy, width - minvx), dtype=np.float32)

    # Fill the array with expression value
    for index, row in so_data["transcripts"].iterrows():
        x, y, gene = (
            int(row["x"] * scale) - minvx,
            int(row["y"] * scale) - minvy,
            row["feature_name"],
        )
        gene_index = gene_to_index[gene]
        array[gene_index, y, x] += 1
    return array


# check the shape of the resulting array
def generate_spt_array():
    so_data = sd_io.xenium(DATA_PATH)
    sel_so_data = so_data.subset(["morphology_focus", "transcripts"])
    crop_so = bounding_box_query(
        sel_so_data,
        min_coordinate=MIN_COORDINATE,
        max_coordinate=MAX_COORDINATE,
        axes=("x", "y"),
        target_coordinate_system="global",
    )

    array_x = pre_process_spt(crop_so, minvx=MIN_COORDINATE[0], minvy=MIN_COORDINATE[1])
    print("Array shape:", array_x.shape)
    print(np.sum(array_x))

    array_y = crop_so.images["morphology_focus"]["scale0"]["image"]
    array_y = np.squeeze(array_y, axis=0)
    normalizedY = array_y / 2685 - 1

    return array_x, normalizedY


def preparePatches(x_data, y_data, patch_size):
    x_train = np.zeros(
        (
            len(TRAIN_STEP_X) * len(TRAIN_STEP_Y),
            x_data.shape[0],
            patch_size,
            patch_size,
        ),
        dtype=np.float32,
    )
    y_train = np.zeros(
        (len(TRAIN_STEP_X) * len(TRAIN_STEP_Y), patch_size, patch_size),
        dtype=np.float32,
    )

    x_test = np.zeros(
        (len(TEST_STEP_X) * len(TEST_STEP_Y), x_data.shape[0], patch_size, patch_size),
        dtype=np.float32,
    )
    y_test = np.zeros(
        (len(TEST_STEP_X) * len(TEST_STEP_Y), patch_size, patch_size), dtype=np.float32
    )

    index = 0
    for i in TRAIN_STEP_X:
        for j in TRAIN_STEP_Y:
            x_train[index] = x_data[
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            y_train[index] = y_data[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            index += 1

    index = 0
    for i in TEST_STEP_X:
        for j in TEST_STEP_Y:
            x_test[index] = x_data[
                :,
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            y_test[index] = y_data[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            index += 1

    return x_train, y_train, x_test, y_test


def prepare_datasets(x_train, y_train, x_test, y_test, batch_size=8):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset


def generate_datasets():
    # Generate array x/y, and normalize
    array_x, array_y = generate_spt_array()

    # Perform 2D convolution for each slice in the first dimension
    kernel = np.ones((5, 5), dtype=np.float32)
    convolved_np = np.array(
        [
            convolve(array_x[i], kernel, mode="constant", cval=0.0)
            for i in range(array_x.shape[0])
        ]
    )
    plot_graph(array_x, array_y, convolved_np, "generate_datasets")

    # Prepare patches and tranpose data
    x_train, y_train, x_test, y_test = preparePatches(array_x, array_y, PATCH_SIZE)
    x_train = np.transpose(x_train, (0, 2, 3, 1))
    x_test = np.transpose(x_test, (0, 2, 3, 1))
    y_train = y_train[..., np.newaxis]  # Add a channel dimension to y_train
    y_test = y_test[..., np.newaxis]

    # Generate datasets
    train_dataset, test_dataset = prepare_datasets(
        x_train, y_train, x_test, y_test, BATCH_SIZE
    )
    return train_dataset, test_dataset
