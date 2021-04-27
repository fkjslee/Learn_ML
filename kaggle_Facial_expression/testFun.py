import numpy as np
import time
import cv2


def process_frame(frame):
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.GaussianBlur(processed_frame, (11, 11), 0)
    _, processed_frame = cv2.threshold(processed_frame, 75, 255, cv2.THRESH_BINARY_INV)

    return processed_frame


def draw_pupil(processed_frame, output_frame):
    """
    Identifies the largest contour (by area) on the processed_frame, and draws its center and fitted ellipse,
    (which is the estimation of the pupil centroid and pupil outline respectively) on the output_frame
    :param processed_frame: Processed frame used to identify the largest contour by area
    :param output_frame: Frame on which to draw the estimated pupil outline and centroid,
    identified using the largest contour
    :return: None
    """
    # after thresholding, the largest contour by area should be the pupil, so we draw that.
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if len(contours) > 0:
        # retrieve the largest contour
        contour = contours[0]

        # apply the convex hull operation to smoothen the contour
        contour = cv2.convexHull(contour)

        # retrieve the center of the contour using the moments function,
        # and draw a dot to signify the centroid of the pupil
        moment = cv2.moments(contour)
        if moment['m00'] != 0:
            centroid = (int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00']))
            cv2.circle(output_frame, centroid, 3, (0, 255, 0), -1)

        # draw an ellipse to outline the pupil
        try:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(output_frame, box=ellipse, color=(0, 255, 0))
        except:
            pass


if __name__ == "__main__":
    img = np.float32([0] * (48 * 48 * 3)).reshape(48, 48, 3)
    cv2.imshow("img", img)
    cv2.waitKey(0)
