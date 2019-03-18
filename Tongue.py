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

def find_motion(film, history):
    motion = None

    while film.isOpened():
        available, frame = film.read()
        if not available: break

        gray = frame[:,:,0]
        if history.time is 0: motion = np.zeros_like(gray).astype('float32')
        history += gray

        temporal_mean = history.data.mean(axis = 0)
        difference = np.absolute(gray - temporal_mean)
        difference[difference < DIFFERENCE_THRESH] = 0

        motion = motion + difference

    return motion

def find_tongue_region(motion):
    viz_motion = motion/motion.max()
    mask = (viz_motion > .32).astype('uint8')

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

global OUTPUT_PATH, fps, mask_final

INPUT_PATH = 'datasets/archive/lac02122017_20_08_52_withaudio.avi'
OUTPUT_PATH = 'results/archive/lac02122017_20_08_52_withaudio.avi'
MEMORY_LENGTH = 20
DIFFERENCE_THRESH = 10# 15

film = cv2.VideoCapture(INPUT_PATH)
fps = film.get(cv2.CAP_PROP_FPS)
history = History(length = MEMORY_LENGTH)

motion = find_motion(film, history)
mask_final = find_tongue_region(motion)

def saves_video(func_that_returns_gray_image):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    global OUTPUT_PATH, fps, mask_final
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (mask_final.shape[1]*4, mask_final.shape[0]*2))#film.get(cv2.CAP_PROP_FPS), )

    def inner(*args, **kwargs):
        gray = func_that_returns_gray_image(*args, **kwargs)
        out.write(cv2.cvtColor(cv2.pyrUp(gray), cv2.COLOR_GRAY2BGR))
        return gray

    return inner

@saves_video
def find_tongue(tongue_region, gray):
    mod = gray.copy()
    mod[~mask_final] = 0#gray[~mask_final]//2
    mod[mod < 60] = 0
    mod[mod >= 60] = 255

    im2, contours, hierarchy = cv2.findContours(mod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE
    largest_contour = find_largest(contours)

    tongue = np.zeros_like(gray)
    cv2.drawContours(tongue, [largest_contour], 0, (255), -1)
    combined = gray.copy()
    cv2.drawContours(combined, [largest_contour], 0, (255), -1)

    side_by_side = np.hstack((gray, combined))
    return side_by_side



film.release()
film = cv2.VideoCapture(INPUT_PATH)
# counter = 0
# with open('saved_points.txt', 'w') as file_handler:
#     while film.isOpened():
#
#
#         file_handler.write('Begin points for frame {}\n'.format(counter))
#         for point in largest_contour:
#             file_handler.write('{} {}\n'.format(point[0][0], point[0][1]))
#         file_handler.write('End points for frame {}\n'.format(counter))
#         counter += 1

while film.isOpened():
    available, frame = film.read()
    if not available: break

    gray = frame[:,:,0].astype('uint8')
    tongue = find_tongue(mask_final, gray)

    cv2.imshow('result', cv2.pyrUp(tongue))
    ch = cv2.waitKey(1)
    if ch == 27:
        break



cv2.destroyAllWindows()
film.release()
out.release()
