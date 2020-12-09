import cv2
import numpy as np
import utils

mask = cv2.imread("PennFudanPed/PedMasks/FudanPed00001_mask.png")
# print(np.max(mask.flatten()))
mask_scaled = np.floor(mask / np.max(mask) * 255).astype(np.uint8)
# mask_scaled = mask
cv2.imshow("IMG",mask_scaled)
# cv2.waitKey(0)

img = cv2.imread("PennFudanPed/PNGImages/FudanPed00001.png")
# utils.display_seg_mask(img,mask_scaled)
utils.display_seg_mask(img[::4,::4,:],mask_scaled[::4,::4,:])