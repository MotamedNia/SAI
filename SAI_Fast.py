# SAI_v3.py This version of SAI program accelerate the aggregation process

# Hossein Motamednia
# 98-6-3

import cv2
import numpy as np
import os

images_dir = "masked_byGT"
maskes_dir = "GT_maskes"

# endregion
center_loc = (11, 3)
mul1 = 7.5
mul2 = 6.5


def write_shifted_images(current_img, current_mask, index, rows, cols, row_mov, col_mov):
    image = np.zeros(current_img.shape, np.uint8)
    mask = np.zeros(current_img.shape, np.uint8)
    if row_mov < 0 and col_mov < 0:
        image[-row_mov:rows, -col_mov:cols] += current_img[0:rows + row_mov,
                                               0:cols + col_mov]
        mask[-row_mov:rows, -col_mov:cols] += current_mask[0:rows + row_mov,
                                              0:cols + col_mov]

    elif row_mov < 0:
        image[-row_mov:rows, 0:cols - col_mov] += current_img[0:rows + row_mov,
                                                  col_mov:cols]
        mask[-row_mov:rows, 0:cols - col_mov] += current_mask[0:rows + row_mov,
                                                 col_mov:cols]
    elif col_mov < 0:
        image[0:rows - row_mov, -col_mov:cols] += current_img[row_mov:rows,
                                                  0:cols + col_mov]
        mask[0:rows - row_mov, -col_mov:cols] += current_mask[row_mov:rows,
                                                 0:cols + col_mov]
    else:
        image[0:rows - row_mov, 0:cols - col_mov] += current_img[row_mov:rows,
                                                     col_mov:cols]
        mask[0:rows - row_mov, 0:cols - col_mov] += current_mask[row_mov:rows,
                                                    col_mov:cols]

    image_path = os.path.join("shift/images", str(index) + ".png")
    cv2.imwrite(image_path, image)
    mask_path = os.path.join("shift/masks", str(index) + ".png")
    cv2.imwrite(mask_path, mask)


class SAI:
    def __init__(self, ref_image):



        self.image = np.zeros(ref_image.shape, np.uint16)
        self.mask = np.zeros(ref_image.shape, np.uint16)



    def compute_SAI(self, index_images, index_maskes):
        for indx in range(0, len(index_images)):
            # Read image
            current_img = index_images[indx]

            # Read mask
            current_mask = index_maskes[indx]
            current_mask = np.array(current_mask / 255, np.uint16)

            self.image += current_img
            self.mask += current_mask
            # write_shifted_images(current_img, current_mask, i, rows, cols, row_mov, col_mov)

        # Create final image
        float_final_image = np.divide(self.image, self.mask)
        final_image = np.array(float_final_image, np.uint8)
        #
        return final_image

    def loc_spec(self, i, center_loc):
        y = int((i - 1) / 21)
        x = i - y * 21
        y = y + 1

        if y % 2 == 0:
            x_diff = center_loc[0] - x
        else:
            x_diff = x - center_loc[0]

        y_diff = center_loc[1] - y

        return x_diff, y_diff
# endregion
