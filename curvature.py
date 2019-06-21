import numpy as np
from scipy import interpolate

INPUT_PATH = 'lac02182018_19_04_51_withaudio.txt'

with open(INPUT_PATH, 'r') as file:
    x = []
    y = []
    x_min = 9999
    x_max = 0
    y_min = 9999
    y_max = 0
    for line in file:
        if "Begin" in line:
            x.append([])
            y.append([])
        elif "End" in line:
            x[-1] = np.asarray(x[-1])
            y[-1] = np.asarray(y[-1])

            y[-1] = y[-1].max() - y[-1]

            bottom_right = (x[-1] - y[-1]).argmax()
            x[-1] = np.roll(x[-1], -bottom_right + 1)
            y[-1] = np.roll(y[-1], -bottom_right + 1)

            if x[-1].min() < x_min: x_min = x[-1].min()
            if x[-1].max() > x_max: x_max = x[-1].max()
            if y[-1].min() < y_min: y_min = y[-1].min()
            if y[-1].max() > y_max: y_max = y[-1].max()
        else:
            (px, py) = (eval(p) for p in line.split())
            x[-1].append(px)
            y[-1].append(py)


    u = np.linspace(0.0, 1.0, num = 100, endpoint = True)
    results = []
    first_ders = []
    second_ders = []
    curvatures = []
    for arrx, arry in zip(x, y):
        spline, t = interpolate.splprep([arrx, arry], s = 10)
        results.append(interpolate.splev(u, spline))
        first_ders.append(interpolate.splev(u, spline, der = 1))
        second_ders.append(interpolate.splev(u, spline, der = 2))

        first_der_x = first_ders[-1][0]
        first_der_y = first_ders[-1][1]
        first_der = np.vstack((first_der_x, first_der_y))
        second_der_x = second_ders[-1][0]
        second_der_y = second_ders[-1][1]
        second_der = np.vstack((second_der_x, second_der_y))

        numerator = np.cross(first_der, second_der, axis = 0)
        denominator = np.power(np.linalg.norm(first_der, axis = 0), 3)
        curvatures.append(numerator/denominator)

    curvatures = np.vstack(curvatures)
    print(curvatures.shape)
    print(curvatures.min())
    print(curvatures.max())

    import cv2
    c_view = curvatures - curvatures.min()
    c_view = (c_view.T/c_view.max()*255).astype('uint8')

    yaxiskey_h = np.zeros_like(c_view) + (u/u.max()*127).astype('uint8')[:,np.newaxis]
    yaxiskey_h = yaxiskey_h[:,:40]
    yaxiskey_s = np.full_like(yaxiskey_h, 255)
    yaxiskey = np.dstack((yaxiskey_h, yaxiskey_s, yaxiskey_s))
    yaxiskey = cv2.cvtColor(yaxiskey, cv2.COLOR_HSV2BGR)

    yaxiskey = cv2.resize(yaxiskey, dsize = (0, 0), fx = 1, fy = 10.0)
    image = cv2.resize(c_view, dsize = (0, 0), fx = 1, fy = 10.0)

    cv2.imshow('Curvature over Time', cv2.pyrDown(image))
    cv2.imshow('Key', cv2.pyrDown(yaxiskey))
    cv2.waitKey(0)
    cv2.imwrite('Curvatures over Time.jpg', image)
    cv2.imwrite('Key.jpg', yaxiskey)
    cv2.destroyAllWindows()

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm

    for i, result in enumerate(results):
        plt.clf()
        plt.cla()

        result_x = result[0]
        result_y = result[1]

        # The following code paints the line according to the vector
        # valued function parameter, U, such that it matches with the Color-Coded
        # Key that was created above
        points = np.array([result_x, result_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis = 1)
        norm = plt.Normalize(u.min(), u.max())
        lc = LineCollection(segments, cmap = 'rainbow', norm = norm)
        lc.set_array(u)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)

        # The following code paints the line according to curvature
        # points = np.array([result_x, result_y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis = 1)
        # norm = plt.Normalize(curvatures[i].min(), curvatures[i].max())
        # lc = LineCollection(segments, cmap = 'cool', norm = norm)
        # lc.set_array(curvatures[i])
        # lc.set_linewidth(5)
        # line = plt.gca().add_collection(lc)


        #plt.plot(x[i], y[i], 'o', result_x, result_y, '-')
        plt.axis([x_min - 5, x_max + 5, y_min, y_max + 2])
        #plt.savefig('ArcLengthAnimation/img{}.png'.format(i), bbox_inches='tight')
        plt.pause(0.001)



            #x[-1].append()
    # file_handler.write('MEMORY_LENGTH {}\n'.format(MEMORY_LENGTH))
    # file_handler.write('DIFFERENCE_THRESH {}\n'.format(DIFFERENCE_THRESH))
    # file_handler.write('MOTION_THRESHOLD {}\n'.format(MOTION_THRESHOLD))
    # file_handler.write('WHITE_THRESHOLD {}\n'.format(WHITE_THRESH
