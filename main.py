import cv2
import numpy as np

from util import *
from history import History

INPUT_PATH = 'datasets/people/lac02122017_20_08_52_withaudio.avi'
OUTPUT_PATH = 'results/people/lac02122017_20_08_52_withaudio'# NOTE: path should not include extension for output
MEMORY_LENGTH = 25
DIFFERENCE_THRESH = 25
MOTION_THRESHOLD = 0.23
WHITE_THRESHOLD = 60

from film import Film
film = Film(INPUT_PATH)
film.out_path = OUTPUT_PATH
film.out_scale = (2, 1)

@film.processor
def find_motion(frame, prev_ret_val, iter, MEMORY_LENGTH, DIFFERENCE_THRESH):
    if iter is 0: return [History(MEMORY_LENGTH), None]
    else:
        history, motion = prev_ret_val

        gray = frame[:,:,0]
        if history.time is 0: motion = np.zeros_like(gray).astype('float32')
        history += gray

        temporal_mean = history.data.mean(axis = 0)
        difference = np.absolute(gray - temporal_mean)
        difference[difference < DIFFERENCE_THRESH] = 0

        motion = motion + difference
        return [history, motion]

@film.processor
@film.output
def find_tongue(frame, prev_ret_val, iter, TONGUE_MASK, WHITE_THRESHOLD):
    if iter is 0: return (None, [])
    else:
        gray = frame[:,:,0]

        mod = gray.copy()
        mod[~TONGUE_MASK] = 0
        mod[mod < WHITE_THRESHOLD] = 0
        mod[mod >= WHITE_THRESHOLD] = 255

        contours, hierarchy = cv2.findContours(mod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = np.asarray(find_largest(contours))[:, 0]
        prev_ret_val[1].append(largest_contour)

        combined = gray.copy()
        cv2.drawContours(combined, [largest_contour], 0, (255), -1)

        # display
        side_by_side = np.hstack((gray, combined))
        imshowbig('result', side_by_side)

        return (side_by_side, prev_ret_val[1])


_, motion = find_motion(MEMORY_LENGTH = MEMORY_LENGTH, DIFFERENCE_THRESH = DIFFERENCE_THRESH)
imshowbig('motion', (motion*255/motion.max()).astype('uint8'))
print('Choose new constants for motion masking, or type continue to move on.')
input_mem_length = input('MEMORY LENGTH: ')
input_dif_thresh = input('DIFFERENCE THRESH: ')
while ('c' not in input_mem_length) and ('c' not in input_dif_thresh):
    _, motion = find_motion(MEMORY_LENGTH = int(eval(input_mem_length)), DIFFERENCE_THRESH = eval(input_dif_thresh))
    imshowbig('motion', (motion*255/motion.max()).astype('uint8'))
    input_mem_length = input('MEMORY LENGTH: ')
    input_dif_thresh = input('DIFFERENCE THRESH: ')

tongue_mask = extract_tongue_from(motion, THRESHOLD = MOTION_THRESHOLD)
print('Choose new constant for tongue extraction, or type continue to move on.')
input_thresh = input('MOTION THRESHOLD: ')
while 'c' not in input_thresh:
    tongue_mask = extract_tongue_from(motion, THRESHOLD = eval(input_thresh))
    input_thresh = input('MOTION THRESHOLD: ')

_, contour_history = find_tongue(TONGUE_MASK = tongue_mask, WHITE_THRESHOLD  = WHITE_THRESHOLD)


with open(OUTPUT_PATH + '.txt', 'w') as file_handler:
    for i, contour in enumerate(contour_history):
        file_handler.write('Begin points for frame {}\n'.format(i))
        for j in range(contour.shape[0]):
            file_handler.write('{} {}\n'.format(contour[j, 0], contour[j, 1]))
        file_handler.write('End points for frame {}\n'.format(i))


cv2.destroyAllWindows()
film.release_all()
