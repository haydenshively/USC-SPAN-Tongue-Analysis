import cv2

class Film(cv2.VideoCapture):

    def __init__(self, path):
        import sys
        version = sys.version_info[0]
        if version == 2: super(Film, self).__init__(path)
        else: super().__init__(path)
        del sys

        import platform
        self.PLATFORM = platform.system()
        del platform

        self.PATH = path
        self.FPS = int(self.get(cv2.CAP_PROP_FPS))
        self.WIDTH = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_path = 'output'
        self.out_scale = (1, 1)

        if self.PLATFORM is 'Windows':
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.out_path_extension = '.avi'
        else:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_path_extension = '.mov'
            
        self.out = None#cv2.VideoWriter(self.out_path + self.out_path_extension, self.fourcc, self.FPS, (self.WIDTH*self.out_scale[0], self.HEIGHT*self.out_scale[1]))

    def reopen(self):
        self.open(self.PATH)

    def loop(self, function, *args, **kwargs):
        self.reopen()
        ret_val = function(None, prev_ret_val = None, iter = 0, *args, **kwargs)
        iter = 1

        while self.isOpened():
            available, frame = self.read()
            if not available: break

            ret_val = function(frame, prev_ret_val = ret_val, iter = iter, *args, **kwargs)
            iter += 1

        return ret_val

    # decorator
    def processor(self, function):
        def looped_function(*args, **kwargs):
            return self.loop(function, *args, **kwargs)
        return looped_function

    # decorator
    def output(self, image_producer):
        self.out = cv2.VideoWriter(self.out_path + self.out_path_extension, self.fourcc, self.FPS, (self.WIDTH*self.out_scale[0], self.HEIGHT*self.out_scale[1]))
        def decorated_function(*args, **kwargs):
            # get whatever value(s) the image_producer returns
            ret_val = image_producer(*args, **kwargs)
            if ret_val is not None:
                # if it returns a tuple, assume the first item is the image
                # otherwise, assume the return value itself is the image
                if isinstance(ret_val, tuple) or isinstance(ret_val, list): image = ret_val[0]
                else: image = ret_val

                if image is not None:
                    # if the image doesn't have a third dimension already, convert to BGR colorspace
                    if len(image.shape) is 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    # write image to file
                    self.out.write(image)
            # allow return value to pass through decorator
            return ret_val

        return decorated_function

    def release_all(self):
        self.out.release()
        self.release()
