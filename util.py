import numpy as np
import cv2


def imshowbig(title, image):
    cv2.imshow(title, cv2.pyrUp(image))
    cv2.waitKey(1)


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


def match_centroid(contours, target):
    best = None
    best_offset = None
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0: continue
        centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])
        offset = (centroid[0] - target[0]) ** 2 + (centroid[1] - target[1]) ** 2
        if (best is None) or (offset < best_offset):
            best = contour
            best_offset = offset
    return best


def remove_lips(mask, contour):
    hull = cv2.convexHull(contour, returnPoints=False)

    defects = cv2.convexityDefects(contour, hull)
    defects_argmax = defects[:, :, 3].argmax()  # remove tongue-arch defect from list
    defects[defects_argmax, :, 3] = 0
    defects_argmax = defects[:, :, 3].argmax()  # now the biggest is bridge between tongue and lips

    s, e, f, d = defects[defects_argmax, 0]
    bridge_x, bridge_y = tuple(contour[f][0])
    mask[:, :bridge_x] = 0
    # cv2.line(mask, (bridge_x - 4, 0), (bridge_x - 4, mask.shape[0] - 1), 0, 8)

    # for i in range(defects.shape[0]):
    #     s,e,f,d = defects[i, 0]
    #     print(d/256.0)
    #     if d/256.0 > 5:
    #         start = tuple(largest_contour[s][0])
    #         end = tuple(largest_contour[e][0])
    #         far = tuple(largest_contour[f][0])
    #         cv2.line(mask,start,end,127,1)
    #         cv2.circle(mask,far,2,127,-1)


def remove_chin(mask, contour):
    hull = cv2.convexHull(contour)

    left_most = tuple(hull[hull[:, :, 0].argmin()][0])

    xy_sums = hull[:, :, 0] + hull[:, :, 1]
    bottom_right_id = xy_sums.argmax()
    bottom_right = tuple(hull[bottom_right_id][0])

    cv2.line(mask, left_most, bottom_right, 0, 4)

    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])

    # hull = cv2.convexHull(contour, returnPoints = False)
    #
    # defects = cv2.convexityDefects(contour, hull)

    # defects_argmax = defects[:,:,3].argmax()# remove tongue-arch defect
    # defects[defects_argmax,:,3] = 0
    # defects_argmax = defects[:,:,3].argmax()# now the biggest is bridge between tongue and lips

    # s, e, f, d = defects[defects_argmax, 0]
    # bridge_x, bridge_y = tuple(contour[f][0])
    # mask[:, :bridge_x] = 0
    # cv2.line(mask, (bridge_x - 4, 0), (bridge_x - 4, mask.shape[0] - 1), 0, 8)

    # print('start frame')
    # for i in range(defects.shape[0]):
    #     s,e,f,d = defects[i, 0]
    #
    #     print(d/256.0)
    #     if (d/256.0 > 5):
    #         start = tuple(contour[s][0])
    #         end = tuple(contour[e][0])
    #         far = tuple(contour[f][0])
    #         cv2.line(mask,start,end,127,1)
    #         cv2.circle(mask,far,2,127,-1)
    # print('end frame')
    # imshowbig('ibuwv', mask)


def extract_tongue_from(motion, THRESHOLD):
    mask = (motion / motion.max() > THRESHOLD).astype('uint8')

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = find_largest(contours)
    M = cv2.moments(largest_contour)
    centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])
    remove_lips(mask, largest_contour)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    remove_chin(mask, match_centroid(contours, centroid))

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(match_centroid(contours, centroid))

    mask_tongue = np.zeros_like(mask)
    cv2.drawContours(mask_tongue, [hull], 0, (1), -1)

    kernel = np.ones((3, 3), dtype='uint8')
    mask_tongue_dilated = cv2.dilate(mask_tongue, kernel, iterations=1)

    vis = (motion / motion.max() * 255).astype('uint8')
    cv2.drawContours((vis), [hull], 0, (255), 1)
    imshowbig('mask', vis)

    return mask_tongue_dilated.astype('bool')
