# At this version SAI_v3.py is used

# Hossein Motamednia
# 98-6-3 initial version
#

from SAI_Fast import SAI
import numpy as np
# import gnumpy as gpu
import itertools
import cv2
from QACompare import QACompare
from multiprocessing import Pool
from multiprocessing import cpu_count
import os


region_corners = []
cd1_corners = np.array([[238, 268], [355, 268], [355, 378], [237, 374]], dtype="float32")
cd2_corners = np.array([[453, 273], [571, 270], [575, 381], [456, 377]], dtype="float32")

region_corners.append(cd1_corners)
region_corners.append(cd2_corners)

ref_image = cv2.imread("ref.png")

shifted_images = []
shifted_masks = []
for i in range(1,106):
    img_path = os.path.join("shift/images", str(i)+".png")
    mask_path = os.path.join("shift/masks", str(i) + ".png")
    shifted_images.append(cv2.imread(img_path))
    shifted_masks.append(cv2.imread(mask_path))

def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def QAS(indexes):
    print(indexes)
    index_images = []
    index_masks = []
    for i in indexes:
        index_images.append(shifted_images[i-1])
        index_masks.append(shifted_masks[i-1])
    sai = SAI(ref_image)
    # sai = SAI("warped", (11, 3), 7.5, 6.5,indexes, location_specialization_method=loc_spec)
    sai_image = sai.compute_SAI(index_images, index_masks)

    psnr, ssim = QACompare.compare_fullReference(sai_image, ref_image, region_corners)

    nonZeroPix = QACompare.get_mask_covering(indexes, "shift/masks", region_corners)

    return indexes, (psnr, ssim), nonZeroPix


s = []
for i in range(1, 106):
    s.append(i)
n = 2

indexes_list = findsubsets(s, n)

print(len(indexes_list))

pool = Pool(processes=14)
results = pool.map(QAS, indexes_list)

print(results)

results = np.array(results)

print(results)
np.save("results.npy", results)

# res = np.load("results.npy",allow_pickle=True)
# print("####################")
# print(res)

