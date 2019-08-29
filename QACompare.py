# Simple quality assessment Kernel specialized for 4 image synthetic aperture imaging
# Compare image quality using both full and no reference methods

# Hossein Motamednia
# 3-6-98 initial version

from imutils.perspective import four_point_transform
from skimage.measure import compare_ssim, compare_psnr
import numpy as np
import os
import cv2


class QACompare:
    def __init__(self):
        pass

    @staticmethod
    def compare_fullReference(image, refImage, regions_corners):
        psnr = 0
        ssim = 0
        for corners in regions_corners:
            img_patch = four_point_transform(image, corners)
            refImg_patch = four_point_transform(refImage, corners)

            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)
            refImg_patch = cv2.cvtColor(refImg_patch, cv2.COLOR_RGB2GRAY)

            psnr += compare_psnr(refImg_patch, img_patch)
            ssim = compare_ssim(refImg_patch, img_patch)

        psnr /= len(regions_corners)
        ssim /= len(regions_corners)

        return psnr, ssim

    @staticmethod
    def get_mask_covering(indexes, mask_dir, regions_corners):

        tmp_path = os.path.join(mask_dir, "53.png")
        tmp_img = cv2.imread(tmp_path)

        mask_covering = np.zeros(tmp_img.shape, np.uint8)

        for index in indexes:
            mask_path = os.path.join(mask_dir, str(index) + ".png")
            mask_img = cv2.imread(mask_path)

            mask_covering = np.bitwise_or(mask_covering,mask_img)

        mask_covering = cv2.cvtColor(mask_covering, cv2.COLOR_RGB2GRAY)

        img_nonZero_pixels = cv2.countNonZero(mask_covering)
        cd_nonZero_pixels = 0
        for corners in regions_corners:
            mask_patch = four_point_transform(mask_covering, corners)
            cv2.threshold(mask_patch, 127, 255, cv2.THRESH_BINARY,mask_patch)

            cd_nonZero_pixels += cv2.countNonZero(mask_patch)


        return img_nonZero_pixels, cd_nonZero_pixels
