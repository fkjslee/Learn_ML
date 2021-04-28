import cv2
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


# test_list_1 = np.array([[0, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 0, 0]]).astype(np.bool)
# test_list_2 = np.array([1, 1, 1, 1, 1, 1, 0])
# krle = b'eNoLCAgIMAEABJkBdQ=='
# sz = [2048, 2048]
# rle = binary_mask_to_rle(test_list_1)
# rle = {'counts': krle, 'size': sz}
#
# # compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
# # u = mask.decode(compressed_rle)
#
# # print(rle['counts'])
# rle = encode(test_list_1)
# ls = decode(rle)
#
#
# base64_str = zlib.compress(rle, zlib.Z_BEST_COMPRESSION)
# print(base64_str)
def scale(img, width, height):
    img = cv2.warpAffine(img, np.float32([[width / img.shape[0], 0, 0], [0, height / img.shape[1], 0]]),
                         (width, height))
    return img


mask = np.uint8([[1, 0, 0], [1, 0, 0]]).astype(np.bool)
mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
mask_to_encode = mask_to_encode.astype(np.uint8)
mask_to_encode = np.asfortranarray(mask_to_encode)

# RLE encode mask --
encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

# compress and base64 encoding --
binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
base64_str = base64.b64encode(binary_str)
t1 = base64.b64decode(base64_str)
t1 = base64.b64decode(
    b'eNp9lHnPnDYQxr/SGMO2So+katQmHB7uw8Cyy3Ivx/f/rzPLm+StKtUW+MfjmcGDxDNmQWxcRboLX0R4g88fvvxkX2zTvdiXrx/++vXPj3/UolUrBFeV2Hku4ksxqsbWuZgsParWbnPRWfGoenvKRWMdg5rtLReltQ1qsYNCFNY0qM1OC5FZ7aB2u2YqBuU7PdPeE/ma4speJU6nRWLdOtU4dUna0areOSox/uKZnuFJZSipBAo0+A6C7gYCnndDGWh4FOFJz3Qlk0sZxAaKH3kAb9nAWd8GP/97/Ff5/yHw/fJO/yaLsyQt9MSoTgnh/fVW4cTv1cTbE3fAk3vkyZ28unn7Jq+uqWPqWZ3zzD3zi8/lqMwuFxVGk/ptzYTGblJ3KDOS1kndIM5EjdFMtKbiisWsPo6paPBKcE/Fp5bWmoWZIOGQcOHghNIqpjahUuOiWkgTKp8+qfwYE+lVDX/fnOkS5+IqHrvylQ7EJvZdHV4QiEMkB9HsE1VMD6aWqfaFbwxMOdN2qM2LfBEagY+rtx8QGNrHxRsOiIzOx8GrDsiM3cfWCw/QRhVg7T12uBljgIVX7NAbe4CpF+4wGXGIiffcYDZ0iLE3bPA0riGGXrvBarRM1w02o2fSTDNTyrsHk8+UReh700q7twgPt2FamBKmJMbdXZ5wGFem2xN8OcX4dLMnhPKIcXOPhShJiCamKqG4doFA3pmurHVMmmniuGyBSK4JVYkWSGSQ4uTuM2QyS3Fwpxm0rFO8u48ZGnlP8erWMzzkmKJ28xkGuaRYuAnTkWLmBjPM0s+IjokoyjB1V6aUtYmpZO0xwSJvGSZuw1rPuznTkymcYJJxTnHzSFTnmLvNSO8Yc3pvMkIv/QJLdx3oLEWBtdsxNUw101DQSbMBWvlkige4y73AxvVZCzVW7t5DJyPNVXqql2nqY+5hlIWmE4w9naXSGLuPHjZ50+i7tx4O2Wk8nLKHwJw0rk7eQ2w+Nc5O3ENu7hoHJ+yhMP2SyO9Bm2GJnbN3UJlRiQ9n66A2kxJvztrBzUxLvDpLB62Zl1g6cwe9WZRYOGMHo1mVmDh9B0+zKTFw7h0cZl/iYlcdpNZC9eyM6llbiY0ddXC3/ApLO6AqVlBhzjRaYYWJ7XewMIWsbUz7V6Lg4j/g9fMLMr/zR+cJpyG+LnEa38sgxWkebIvqhyme1mGcZvrSaJfMVrxZCc+XpXLmy4R5VzqWc/ny8+ffP5UL/f7/AF3q3Co=')
t2 = zlib.decompress(t1)
rleObj = {'counts': t2, 'size': [2, 3]}
rleObj = {'counts': t2, 'size': [2048, 2048]}
rleObj = [rleObj]
t3 = coco_mask.decode(rleObj)
print(type(t3))
print(t3.shape)
t3 = t3.reshape(t3.shape[0], t3.shape[1])
print(t3.dtype)

for i in range(t3.shape[0]):
    for j in range(t3.shape[1]):
        if t3[i, j] != 0:
            t3[i, j] = 250

cv2.imshow("img", scale(t3, 500, 500))
cv2.waitKey(0)
