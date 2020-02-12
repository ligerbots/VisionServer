#!/usr/bin/python3

# Wrap the CSCore camera
# Allows for easier handling of different models

import cscore
import logging
from numpy import array, rot90
import json
import os.path
from threading import Thread
from time import time


class LigerCamera:
    '''Wrapper for camera related functionality.
    Makes handling different camera models easier
    Includes a threaded reader, so you can grab a frame without waiting, if needed'''

    def __init__(self, camera_server, name, device, image_height=240, fps=30, image_width=320, rotation=0):
        '''Create a USB camera and configure it.
        Note: rotation is an angle: 0, 90, 180, -90'''

        self.rotation = rotation

        # camera calibration info, if loaded
        self.calibration_matrix = None
        self.distortion_matrix = None

        self.camera = cscore.UsbCamera(name, device)
        camera_server.startAutomaticCapture(camera=self.camera)
        # keep the camera open for faster switching
        self.camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

        self.camera.setResolution(int(image_width), int(image_height))
        self.camera.setFPS(int(fps))

        # set the camera for no auto focus, focus at infinity
        # NOTE: order does matter
        self.set_property('focus_auto', 0)
        self.set_property('focus_absolute', 0)

        mode = self.camera.getVideoMode()
        logging.info("camera '%s' pixel format = %s, %dx%d, %dFPS", name,
                     mode.pixelFormat, mode.width, mode.height, mode.fps)

        # Variables for the threaded read loop
        self.sink = camera_server.getVideo(camera=self.camera)

        self.frametime = None
        self.camera_frame = None
        self.stopped = False
        self.frame_number = 0
        self.last_read = 0

        return

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

    def load_calibration(self, calib_file):
        '''Load calibration information from the specified file'''

        logging.info(f'Loading calibration from {calib_file}')

        with open(calib_file) as f:
            json_data = json.load(f)
            self.calibration_matrix = array(json_data["camera_matrix"])
            self.distortion_matrix = array(json_data["distortion"])
        return

    def start(self):
        '''Start the thread to read frames from the video stream'''

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        '''Threaded read loop'''

        fps_startt = time()

        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            self.frametime, self.camera_frame = self.sink.grabFrame(self.camera_frame)
            self.frame_number += 1

            if self.rotation:
                # Numpy is *much* faster than the OpenCV routine
                num_rot = (self.rotation // 90) % 4   # integer division
                self.camera_frame = rot90(self.camera_frame, num_rot)

            if self.frame_number % 150 == 0:
                endt = time()
                dt = endt - fps_startt
                logging.info("threadedcamera: 150 frames in {0:.3f} seconds = {1:.2f} FPS".format(dt, 150.0 / dt))
                fps_startt = endt
        return

    def next_frame(self):
        '''Wait for a new frame'''

        while self.last_read == self.frame_number:
            time.sleep(0.001)
        self.last_read = self.frame_number
        return self.frametime, self.camera_frame

    def get_frame(self):
        '''Return the frame most recently read, no waiting. This may be a repeat of the previous image.'''

        return self.frametime, self.camera_frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.3)         # time for thread to stop (proper way???)
        return


class LogitechC930e(LigerCamera):
    def __init__(self, camera_server, name, device, image_height=240, fps=30, image_width=None, rotation=None):
        if not image_width:
            image_width = 424 if image_height == 240 else 848

        super().__init__(self, camera_server, name, device, image_height=image_height, fps=fps, image_width=image_width, rotation=rotation)

        # Logitech does not like having exposure_auto_priority on when the light is poor
        #  slows down the frame rate
        self.set_property('exposure_auto_priority', 0)

        return

    def find_calibration(self, calibration_dir):
        '''Find the correct calibration file and load it'''

        mode = self.camera.getVideoMode()
        filename = f'c930e_{mode.width}x{mode.height}_calib.json'
        fullname = os.path.join(calibration_dir, filename)
        super().load_calibration(fullname)
        return


class PSEye(LigerCamera):
    def __init__(self, camera_server, name, device, image_height=240, fps=30, image_width=None, rotation=None):
        if not image_width:
            image_width = 320 if image_height == 240 else 640

        super().__init__(self, camera_server, name, device, image_height=image_height, fps=fps, image_width=image_width, rotation=rotation)

        # PS Eye camera needs to have its pixelformat set
        # Not tested yet. Does this need to happen earlier?
        self.camera.setPixelFormat(cscore.VideoMode.PixelFormat.kYUYV)

        return
