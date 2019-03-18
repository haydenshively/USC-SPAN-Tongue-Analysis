import cv2
import numpy as np

class History:
    DEFAULT_MEMORY_LENGTH = 20

    def __init__(self, length = DEFAULT_MEMORY_LENGTH):
        self.length = length
        self.time = 0
        self.data = None

    def __iadd__(self, frame):
        if self.time is 0: self.data = np.zeros((self.length, frame.shape[0], frame.shape[1]), dtype = frame.dtype)
        self.data[self.time%self.length] = frame
        self.time += 1
        return self

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

INPUT_PATH = 'datasets/worms/A.mov'
OUTPUT_PATH = 'results/worms/A.avi'
MEMORY_LENGTH = 10
DIFFERENCE_THRESH = 5

film = cv2.VideoCapture(INPUT_PATH)
history = History(length = MEMORY_LENGTH)
motion = None

while film.isOpened():
    print(history.time)
    available, frame = film.read()
    if not available: break

    grey = frame[:,:,0]
    if history.time is 0: motion = np.zeros_like(grey).astype('float32')
    history += grey

    temporal_mean = history.data.mean(axis = 0)
    difference = np.absolute(grey - temporal_mean)
    difference[difference < DIFFERENCE_THRESH] = 0

    motion = motion + difference


viz_motion = motion/motion.max()
cv2.imshow('vismot', (viz_motion*255).astype('uint8'))
mask = (viz_motion > .5).astype('uint8')

kernel = np.ones((3, 3), dtype = 'uint8')
mask = cv2.dilate(mask, kernel, iterations = 2)

im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = find_largest(contours)
hull = cv2.convexHull(largest_contour)

mask_tongue = np.zeros_like(mask)
cv2.drawContours(mask_tongue, [hull], 0, (1), -1)

kernel = np.ones((3, 3), dtype = 'uint8')
mask_tongue_dilated = cv2.dilate(mask_tongue, kernel, iterations = 1)

cv2.imshow('mask', mask*255)
cv2.imshow('mask_tongue', mask_tongue_dilated*255)

cv2.waitKey(0)

mask_final = mask_tongue_dilated.astype('bool')
film = cv2.VideoCapture(INPUT_PATH)

"""
Temporary video saving code
"""
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, film.get(cv2.CAP_PROP_FPS), (mask_final.shape[1]*4, mask_final.shape[0]*2))

while film.isOpened():
    available, frame = film.read()
    if not available: break

    grey = frame[:,:,0].astype('uint8')

    mod = grey.copy()
    mod[~mask_final] = 0#grey[~mask_final]//2
    cv2.imshow('mod', mod)
    # mod[mod < 60] = 0
    # mod[mod >= 60] = 255

    # gradient = cv2.morphologyEx(mod, cv2.MORPH_GRADIENT, kernel)
    # mod_dilated = cv2.dilate(mod, kernel, iterations = 1)

    im2, contours, hierarchy = cv2.findContours(mod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE
    largest_contour = find_largest(contours)

    tongue = np.zeros_like(grey)
    cv2.drawContours(tongue, [largest_contour], 0, (255), -1)
    combined = grey.copy()
    cv2.drawContours(combined, [largest_contour], 0, (255), -1)

    side_by_side = np.hstack((grey, combined))

    cv2.imshow('result', side_by_side)
    out.write(cv2.cvtColor(mod, cv2.COLOR_GRAY2BGR))
    ch = cv2.waitKey(1)
    if ch == 27:
        break



cv2.destroyAllWindows()
film.release()
out.release()
