import numpy as np
import cv2

def imshowbig(title, image):
    cv2.imshow(title, cv2.pyrUp(image))

def find_largest(contours):
    largest = contours[0]
    largest_area = cv2.contourArea(largest)
    for contour in contours:
        # moment = cv2.moments(contour)
        # area = moment['m00']
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest = contour
            largest_area = area
    return largest

def extract_tongue_from(motion, THRESHOLD):
    viz_motion = motion/motion.max()
    mask = (viz_motion > THRESHOLD).astype('uint8')

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest(contours)
    hull = cv2.convexHull(largest_contour)

    mask_tongue = np.zeros_like(mask)
    cv2.drawContours(mask_tongue, [hull], 0, (1), -1)

    kernel = np.ones((3, 3), dtype = 'uint8')
    mask_tongue_dilated = cv2.dilate(mask_tongue, kernel, iterations = 1)

    imshowbig('mask', mask*255)
    imshowbig('mask_tongue', mask_tongue_dilated*255)
    cv2.waitKey(0)

    return mask_tongue_dilated.astype('bool')
