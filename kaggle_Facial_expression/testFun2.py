import cv2
import numpy as np

img = cv2.imread("1.jpg")
points = np.array([[1, 2], [2, 3]])
x, y, w, h = cv2.boundingRect(points)
print(points)
print(x, y, w, h)
cv2.circle(img, (x + int(w / 2), y + int(h / 2)), int((h) / 3), (0, 0, 255), 3)
cv2.imshow("img", img)
cv2.waitKey(0)
