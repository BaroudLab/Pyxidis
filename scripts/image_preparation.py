import numpy as np
import tifffile
import glob
import pandas
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
import sys

import numpy as np
import tifffile
import os
from tqdm.notebook import tqdm

from skimage.segmentation import find_boundaries
from skimage.measure import label

import math


def save_cut_images(
    multi_matrix,
    SAVE_PATH,
    dirname="saucisson_",
    savename="image_piece_",
    number_images_per_batch=50,
):
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    for piece, i in tqdm(zip(multi_matrix, np.arange(len(multi_matrix)))):

        k = i // number_images_per_batch
        new_dir = os.path.join(SAVE_PATH, dirname + f"{k:02d}")

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        tifffile.imsave(os.path.join(new_dir, savename + f"{i:04}.tif"), piece)

    return


def reconstruct_image(SAVE_PATH, original_image, codex, image_name, max_image_size=300, image_is_2D = False):

    """
    Rebuilds the complete fetal liver image from the segmented sub-images. They
    are assembled, the individual cells are relabeled and the labeled image
    is concatenated with the original fluorescence image. Finally we save the
    result.
    """

    max_label = 0
    multi_matrix_seg = []

    for dirname in sorted(os.listdir(SAVE_PATH)):

        print(dirname)

        for fname in tqdm(sorted(glob.glob(os.path.join(SAVE_PATH, dirname, "*.npy")))):

            # load image
            new_matrix = np.load(fname, allow_pickle=True).item()["masks"]
            new_matrix[new_matrix > 0] += max_label

            # separate cells
            boundary_label_mask = find_boundaries(new_matrix, connectivity=2)
            new_matrix[boundary_label_mask == 1] = 0

            # append image
            del boundary_label_mask
            max_label = np.max(new_matrix)
            multi_matrix_seg.append(new_matrix.astype("int32"))

    # reconstructs image from small masks
    reconstructed_image = image_recompose(
        multi_matrix_seg,
        original_image[..., 0],
        max_image_size=max_image_size,
        codex=codex,
        image_is_2D=image_is_2D
    )

    # make binary mask
    mask_image = (reconstructed_image > 0).astype(int)

    tifffile.imsave(os.path.join(SAVE_PATH, image_name + "_mask_image.tif"), mask_image)

    label_image = label(mask_image).astype("int32")
    tifffile.imsave(
        os.path.join(SAVE_PATH, image_name + "_label_image.tif"), label_image
    )
    del mask_image

    img_test_seg = np.concatenate(
        (original_image, label_image[..., np.newaxis].astype("int32")), axis=-1
    )
    del original_image
    del label_image

    tifffile.imsave(os.path.join(SAVE_PATH, image_name + "_labeled.tif"), img_test_seg)

    return

def cell_type_maker(property_frame,
                    category_channels,
                    marker_list,
                    pm_list):
    
    for (cat, marker) in zip(category_channels, marker_list):

        property_frame['cell_properties_' + marker] = [marker + pm_list[cat_val] for cat_val in property_frame[cat].values]

        property_frame['cell_properties'] = property_frame['cell_properties'].values + [marker + pm_list[cat_val] for cat_val in property_frame[cat].values]

    property_frame['cell_properties'] = property_frame[['cell_properties_'+ marker for marker in marker_list]].agg(', '.join, axis=1)
    return property_frame


def image_recompose(
    multi_segmented_matrix,
    original_image,
    max_image_size,
    codex,
    cut_in_z_direction=False,
    image_is_2D=False,
):

    reconstructed_image = np.zeros(np.shape(original_image))

    if not cut_in_z_direction:

        for plane, n in zip(multi_segmented_matrix, codex):

            n_x, n_y = n

            if image_is_2D:
                reconstructed_image[
                    n_x * max_image_size : (n_x + 1) * max_image_size,
                    n_y * max_image_size : (n_y + 1) * max_image_size,
                ] = plane
            else:
                reconstructed_image[
                    :,
                    n_x * max_image_size : (n_x + 1) * max_image_size,
                    n_y * max_image_size : (n_y + 1) * max_image_size,
                ] = plane

    else:

        for plane, n in zip(multi_segmented_matrix, codex):

            n_z, n_x, n_y = n

            reconstructed_image[
                n_z * max_image_size : (n_z + 1) * max_image_size,
                n_x * max_image_size : (n_x + 1) * max_image_size,
                n_y * max_image_size : (n_y + 1) * max_image_size,
            ] = plane

    return reconstructed_image


def image_cutter(
    original_image, max_image_size=300, cut_in_z_direction=False, image_is_2D=False
):

    multi_matrix = []
    codex = []

    if image_is_2D:
        nx, ny = original_image.shape
    else:
        nz, nx, ny = original_image.shape

    if not cut_in_z_direction:

        for n_x in range(math.ceil(nx / max_image_size)):

            for n_y in range(math.ceil(ny / max_image_size)):

                if image_is_2D:
                    multi_matrix.append(
                        original_image[
                            n_x * max_image_size : (n_x + 1) * max_image_size,
                            n_y * max_image_size : (n_y + 1) * max_image_size,
                        ]
                    )

                else:
                    multi_matrix.append(
                        original_image[
                            :,
                            n_x * max_image_size : (n_x + 1) * max_image_size,
                            n_y * max_image_size : (n_y + 1) * max_image_size,
                        ]
                    )

                codex.append((n_x, n_y))

        return multi_matrix, codex

    else:

        for n_z in range(math.ceil(original_image.shape[0] / max_image_size)):

            for n_x in range(math.ceil(original_image.shape[1] / max_image_size)):

                for n_y in range(math.ceil(original_image.shape[2] / max_image_size)):

                    multi_matrix.append(
                        original_image[
                            n_z * max_image_size : (n_z + 1) * max_image_size,
                            n_x * max_image_size : (n_x + 1) * max_image_size,
                            n_y * max_image_size : (n_y + 1) * max_image_size,
                        ]
                    )
                    codex.append((n_z, n_x, n_y))

    return multi_matrix, codex


def test_color_threshold(
    test_img,
    color_channel,
    nuclei_channel,
    property_frame,
    color_column,
    color_threshold,
    vmin=0,
    vmax=5000,
    s=40,
):

    property_frame["type"] = (property_frame[color_column] > color_threshold).astype(
        int
    )

    fig, ax = plt.subplots(1, 2, figsize=(18, 12))

    ax[0].imshow(test_img[..., nuclei_channel], cmap="gray")
    ax[0].scatter(
        property_frame.y, property_frame.x, c=property_frame.type, cmap="RdYlBu_r", s=s
    )
    ax[0].axis("off")

    ax[1].imshow(test_img[..., color_channel], vmin=vmin, vmax=vmax, cmap="gray")
    ax[1].scatter(
        property_frame.y, property_frame.x, c=property_frame.type, cmap="RdYlBu_r", s=s
    )
    ax[1].axis("off")


def cell_selection(
    full_image: np.ndarray,
    property_frame: pandas.DataFrame,
    nuclei_channel: int,
    selection_channel: int,
    dx: int = 80,
    savedir: str = ".",
):

    """
    From the full_image, loop through each cell in nuclei_channel. For each cell, find its position, and crop the image in the selection_channel into a
    square of size dx. Save the cropped image in the savedir. Each saved image is named according to the cell number.
    """

    ndim = full_image.ndim
    property_frame.index = property_frame.label.values

    # Create a folder for the cells
    os.makedirs(savedir + "/" + str(selection_channel), exist_ok=True)

    if ndim == 3:

        # Loop over the cells
        for cell in tqdm(property_frame.index):

            # Crop the image
            x = property_frame.loc[cell].x.astype("int64")
            y = property_frame.loc[cell].y.astype("int64")

            cropped_image = full_image[
                x - dx // 2 : x + dx // 2, y - dx // 2 : y + dx // 2, selection_channel
            ]

            nuclei_image = full_image[
                x - dx // 2 : x + dx // 2, y - dx // 2 : y + dx // 2, nuclei_channel
            ]

            nuclei_image = (nuclei_image == cell).astype("int32")

            save_image = [cropped_image, nuclei_image]
            save_image = np.array(save_image, dtype=np.uint32)

            if (2, dx, dx) == np.shape(save_image):

                #save_image = map_to_uint8(save_image)
                save_image = np.array([map_to_uint8(save_image[0]), map_to_uint8(save_image[1])])

                tifffile.imsave(
                    os.path.join(savedir, str(selection_channel), f"{cell:06}.tif"),
                    save_image,
                )

    if ndim == 4:

        # Loop over the cells
        for cell in tqdm(property_frame.index):

            # Crop the image
            z = property_frame.loc[cell].z.astype("int64")
            x = property_frame.loc[cell].x.astype("int64")
            y = property_frame.loc[cell].y.astype("int64")

            cropped_image = full_image[
                z,
                x - dx // 2 : x + dx // 2,
                y - dx // 2 : y + dx // 2,
                selection_channel,
            ]

            nuclei_image = full_image[
                z, x - dx // 2 : x + dx // 2, y - dx // 2 : y + dx // 2, nuclei_channel
            ]

            nuclei_image = nuclei_image == cell

            save_image = [cropped_image, nuclei_image]
            save_image = np.array(save_image, dtype=np.uint32)

            if (2, dx, dx) == np.shape(save_image):

                #save_image = map_to_uint8(save_image)
                save_image = np.array([map_to_uint8(save_image[0]), map_to_uint8(save_image[1])])

                tifffile.imsave(
                    os.path.join(savedir, str(selection_channel), f"{cell:06}.tif"),
                    save_image,
                )

    return


def map_to_uint8(img):
    """
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    """

    lower_bound = np.min(img)
    upper_bound = np.max(img)

    lut = np.concatenate(
        [
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2 ** 16 - upper_bound, dtype=np.uint16) * 255,
        ]
    )
    return lut[img].astype(np.uint8)
