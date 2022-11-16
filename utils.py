import cv2 as cv
import numpy as np


def box(img, mask):
    maskimg = cv.bitwise_and(src1=img, src2=img, mask=mask)
    sub_img = np.where(maskimg > 0)
    final = maskimg[min(sub_img[0]):max(sub_img[0]), min(sub_img[1]):max(sub_img[1]), :]
    return final
