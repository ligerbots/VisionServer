#!/usr/bin/python3

# Wrap the CSCore camera
# Allows for easier handling of different models

import cscore
import logging
from numpy import rot90
import os.path
from threading import Thread
from time import time, sleep
from camerautil import load_calibration_file


class Camera:
    '''Wrapper for camera related functionality.
    Makes handling different camera models easier
    Includes a threaded reader, so you can grab a frame without waiting, if needed'''

    def __init__(self, camera_server, name, device, height=240, fps=30, width=320, rotation=0, threaded=False):
        '''Create a USB camera and configure it.
        Note: rotation is an angle: 0, 90, 180, -90

        You can use threading for reading the image to make sure it get each one.
        Otherwise, when processing is long, we might skip every other one, and get only 15fps.'''

        self.width = int(width)
        self.height = int(height)
        self.rot90_count = (rotation // 90) % 4  # integer division

        self.camera = cscore.UsbCamera(name, device)
        camera_server.startAutomaticCapture(camera=self.camera)
        # keep the camera open for faster switching
        self.camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

        self.camera.setResolution(self.width, self.height)
        self.camera.setFPS(int(fps))

        # set the camera for no auto focus, focus at infinity
        # NOTE: order does matter
        self.set_property('focus_auto', 0)
        self.set_property('focus_absolute', 0)

        mode = self.camera.getVideoMode()
        logging.info(f"Camera '{name}': pixel format = {mode.pixelFormat}, {mode.width}x{mode.height}, {mode.fps}FPS")
        if threaded:
            logging.info(f"Camera '{name}': using threaded reader")

        # Variables for the threaded read loop
        # TODO: this seems to freeze occasionally. 
        self.sink = camera_server.getVideo(camera=self.camera)
        logging.info('sink was created')

        self.calibration_matrix = None
        self.distortion_matrix = None

        self.threaded = threaded
        self.frametime = None
        self.temp_frame = None
        self.camera_frame = None
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0

        return

    def get_name(self):
        return self.camera.getName()

    def set_exposure(self, value):
        '''Set the camera exposure. 0 means auto exposure'''

        logging.info(f"Setting camera exposure to '{value}'")
        if value == 0:
            self.camera.setExposureAuto()
            # Logitech does not like having exposure_auto_priority on when the light is poor
            #  slows down the frame rate
            # camera.getProperty('exposure_auto_priority').set(1)
        else:
            self.camera.setExposureManual(int(value))
            # camera.getProperty('exposure_auto_priority').set(0)
        return

    def set_property(self, name, value):
        '''Set a camera property, such as auto_focus'''

        logging.info(f"Setting camera property '{name}' to '{value}'")
        try:
            try:
                propVal = int(value)
            except ValueError:
                self.camera.getProperty(name).setString(value)
            else:
                self.camera.getProperty(name).set(propVal)
        except Exception as e:
            logging.warn("Unable to set property '{}': {}".format(name, e))

        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        if self.threaded:
            t = Thread(target=self.update, args=())
            t.daemon = True
            t.start()
        return

    def update(self):
        '''Threaded read loop'''

        # fps_startt = time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self._read_one_frame()

            # if self.frame_number % 150 == 0:
            #     endt = time()
            #     dt = endt - fps_startt
            #     logging.info("threadedcamera '{0}': 150 frames in {1:.3f} seconds = {2:.2f} FPS".format(self.get_name(), dt, 150.0 / dt))
            #     fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        if self.threaded:
            while self.last_read == self.frame_number:
                sleep(0.001)
            self.last_read = self.frame_number
        else:
            self._read_one_frame()

        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        if not self.threaded:
            raise Exception("Called get_frame on a non-threaded reader")

        return self.frametime, self.camera_frame

    def _read_one_frame(self):
        ft, self.temp_frame = self.sink.grabFrame(self.temp_frame)

        if self.rot90_count and ft > 0:
            # Numpy is *much* faster than the OpenCV routine
            self.temp_frame = rot90(self.temp_frame, self.rot90_count)

        # Don't update camera_frame and frame_number until done all processing
        # this prevents the client from fetching it before it is ready
        self.camera_frame = self.temp_frame
        self.frametime = ft
        self.frame_number += 1
        return

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # sleep(0.3)         # time for thread to stop (proper way???)
        return


class LogitechC930e(Camera):
    def __init__(self, camera_server, name, device, height=240, fps=30, width=None, rotation=0, threaded=False):
        if not width:
            width = 424 if height == 240 else 848

        super().__init__(camera_server, name, device, height=height, fps=fps, width=width, rotation=rotation, threaded=threaded)

        # Logitech does not like having exposure_auto_priority on when the light is poor
        #  slows down the frame rate
        self.set_property('exposure_auto_priority', 0)

        return

    def load_calibration(self, calibration_dir):
        '''Find the correct calibration file and load it'''

        filename = f'c930e_{self.width}x{self.height}_calib.json'
        fullname = os.path.join(calibration_dir, filename)
        self.calibration_matrix, self.distortion_matrix = load_calibration_file(fullname, self.rot90_count*90)
        return


class PSEye(Camera):
    def __init__(self, camera_server, name, device, height=240, fps=30, width=None, rotation=0, threaded=False):
        if not width:
            width = 320 if height == 240 else 640

        super().__init__(camera_server, name, device, height=height, fps=fps, width=width, rotation=rotation, threaded=threaded)

        # PS Eye camera needs to have its pixelformat set
        # Not tested yet. Does this need to happen earlier?
        self.camera.setPixelFormat(cscore.VideoMode.PixelFormat.kYUYV)

        return

    def load_calibration(self, calibration_dir):
        '''Find the correct calibration file and load it'''

        # nothing yet
        return
