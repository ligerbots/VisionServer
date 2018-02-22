#!/usr/bin/env python3

# Base camera server
# Create a subclass with this years image processing to actually use

import time
import cv2
import numpy
import logging

import cscore
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty
from networktables import NetworkTables

from switchtarget2018 import SwitchTarget2018
from cubefinder2018 import CubeFinder2018


class VisionServer2018(object):
    '''Vision server for 2018 Power Up'''

    INITIAL_MODE = 'switch'
    DRIVER_MODE = 3.0

    # NetworkTable parameters

    output_fps_limit = ntproperty('/SmartDashboard/vision/output_fps_limit', 16,
                                  doc='FPS limit of frames sent to MJPEG server')

    # fix the TCP port for the main video, so it does not change with multiple cameras
    output_port = ntproperty('/SmartDashboard/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup.
    tuning = ntproperty('/SmartDashboard/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')

    image_width = ntproperty('/SmartDashboard/vision/width', 640, writeDefault=False, doc='Image width')
    image_height = ntproperty('/SmartDashboard/vision/height', 480, writeDefault=False, doc='Image height')
    camera_fps = ntproperty('/SmartDashboard/vision/fps', 30, writeDefault=False, doc='FPS from camera')

    # Cube finding parameters

    # Color threshold values, in HSV space
    cube_hue_low_limit = ntproperty('/SmartDashboard/vision/cube/hue_low_limit', 25,
                                    doc='Hue low limit for thresholding (cube mode)')
    cube_hue_high_limit = ntproperty('/SmartDashboard/vision/cube/hue_high_limit', 75,
                                     doc='Hue high limit for thresholding (cube mode)')

    cube_saturation_low_limit = ntproperty('/SmartDashboard/vision/cube/saturation_low_limit', 95,
                                           doc='Saturation low limit for thresholding (cube mode)')
    cube_saturation_high_limit = ntproperty('/SmartDashboard/vision/cube/saturation_high_limit', 255,
                                            doc='Saturation high limit for thresholding (cube mode)')

    cube_value_low_limit = ntproperty('/SmartDashboard/vision/cube/value_low_limit', 95,
                                      doc='Value low limit for thresholding (cube mode)')
    cube_value_high_limit = ntproperty('/SmartDashboard/vision/cube/value_high_limit', 255,
                                       doc='Value high limit for thresholding (cube mode)')

    cube_exposure = ntproperty('/SmartDashboard/vision/cube/exposure', 0, doc='Camera exposure for cube (0=auto)')

    # Switch target parameters

    switch_hue_low_limit = ntproperty('/SmartDashboard/vision/switch/hue_low_limit', 70,
                                      doc='Hue low limit for thresholding (switch mode)')
    switch_hue_high_limit = ntproperty('/SmartDashboard/vision/switch/hue_high_limit', 100,
                                       doc='Hue high limit for thresholding (switch mode)')

    switch_saturation_low_limit = ntproperty('/SmartDashboard/vision/switch/saturation_low_limit', 100,
                                             doc='Saturation low limit for thresholding (switch mode)')
    switch_saturation_high_limit = ntproperty('/SmartDashboard/vision/switch/saturation_high_limit', 255,
                                              doc='Saturation high limit for thresholding (switch mode)')

    switch_value_low_limit = ntproperty('/SmartDashboard/vision/switch/value_low_limit', 130,
                                        doc='Value low limit for thresholding (switch mode)')
    switch_value_high_limit = ntproperty('/SmartDashboard/vision/switch/value_high_limit', 255,
                                         doc='Value high limit for thresholding (switch mode)')

    switch_exposure = ntproperty('/SmartDashboard/vision/switch/exposure', 6, doc='Camera exposure for switch (0=auto)')

    camera_height = ntproperty('/SmartDashboard/vision/camera_height', 24.0, doc='Camera height (inches)')

    # Paul Rensing 1/21/2018: not sure we need these as tunable NetworkTable parameters,
    #  so comment out for now.
    #
    # # distance between the two target bars, in units of the width of a bar
    # switch_target_separation = ntproperty('/SmartDashboard/vision/switch/target_separation', 3.0,
    #                                       doc='switch target horizontal separation, in widths')

    # # max distance in pixels that a contour can from the guessed location
    # switch_max_target_dist = ntproperty('/SmartDashboard/vision/switch/max_target_dist', 50,
    #                                     doc='switch max distance between contour and expected location (pixels)')

    # # pixel area of the bounding rectangle - just used to remove stupidly small regions
    # switch_contour_min_area = ntproperty('/SmartDashboard/vision/switch/contour_min_area', 100,
    #                                      doc='switch min area of an interesting contour (sq. pixels)')

    # switch_approx_polydp_error = ntproperty('/SmartDashboard/vision/switch/approx_polydp_error', 0.06,
    #                                         doc='switch approxPolyDP error')

    image_writer_state = ntproperty('/SmartDashboard/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # This ought to be a Choosable, but the Python implementation is lame. Use a string for now.
    # This is the NT variable, which can be set from the Driver's station
    nt_active_mode = ntproperty('/SmartDashboard/vision/active_mode', INITIAL_MODE, doc='Active mode')

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time
    # Value is (Found, tvec, rvec) as a flat array. All values are floating point (required by NT).
    target_info = ntproperty('/SmartDashboard/vision/target_info', 6 * [0.0, ], doc='Packed array of target info: found, tvec, rvec')

    def __init__(self, calib_file):
        # for processing stored files and no camera
        self.file_mode = False

        # self.camera_device_vision = 1
        # self.camera_device_driver = 2  # TODO: correct value?

        # Pick the cameras by USB/device path. That way, they are always the same
        self.camera_device_vision = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_E4B9053E-video-index0'
        self.camera_device_driver = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_D95C053E-video-index0'

        # time of each frame. Sent to the RoboRio as a heartbeat
        self.image_time = 0

        self.camera_server = cscore.CameraServer.getInstance()
        self.camera_server.enableLogging()

        self.camera_feeds = {}
        self.current_sink = None
        self.main_camera = None
        self.add_cameras()

        self.create_output_stream()

        self.switch_finder = SwitchTarget2018(calib_file)
        self.cube_finder = CubeFinder2018(calib_file)

        self.update_parameters()

        # active mode. To be compared to nt_active_mode to see if it has changed
        self.active_mode = None
        self.curr_processor = None

        # Start in cube mode, then then switch to INITIAL_MODE after camera is fully initialized
        self.switch_mode('cube')

        # TODO: set all the parameters from NT

        # rate limit parameters
        self.previous_output_time = time.time()
        self.camera_frame = None
        self.output_frame = None

        # if asked, save image every 1/2 second
        # images are saved under the directory 'saved_images' in the current directory
        #  (ie current directory when the server is started)
        self.image_writer = ImageWriter(location_root='./saved_images',
                                        capture_period=0.5, image_format='jpg')

        return

    # --------------------------------------------------------------------------------
    # Methods generally customized each year

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.switch_finder.set_color_thresholds(self.switch_hue_low_limit, self.switch_hue_high_limit,
                                                self.switch_saturation_low_limit, self.switch_saturation_high_limit,
                                                self.switch_value_low_limit, self.switch_value_high_limit)
        self.cube_finder.set_color_thresholds(self.cube_hue_low_limit, self.cube_hue_high_limit,
                                              self.cube_saturation_low_limit, self.cube_saturation_high_limit,
                                              self.cube_value_low_limit, self.cube_value_high_limit)

        self.cube_finder.camera_height = self.camera_height
        return

    def add_cameras(self):
        '''add a single camera at /dev/videoN, N=camera_device'''

        self.add_camera('main', self.camera_device_vision, True)
        self.add_camera('driver', self.camera_device_driver, False)
        return

    def preallocate_arrays(self):
        '''Preallocate the intermediate result image arrays'''

        # NOTE: shape is (height, width, #bytes)
        self.camera_frame = numpy.zeros(shape=(self.image_height, self.image_width, 3),
                                        dtype=numpy.uint8)
        # self.output_frame = self.camera_frame
        return

    def switch_mode(self, new_mode):
        logging.info("Switching mode to '%s'" % new_mode)

        if new_mode == 'cube':
            if self.active_camera != 'main':
                self.switch_camera('main')
            self.curr_processor = self.cube_finder
            VisionServer2018.set_exposure(self.main_camera, self.cube_exposure)

        elif new_mode == 'switch':
            if self.active_camera != 'main':
                self.switch_camera('main')
            self.curr_processor = self.switch_finder
            VisionServer2018.set_exposure(self.main_camera, self.switch_exposure)

        elif new_mode == 'intake':
            if self.active_camera != 'main':
                self.switch_camera('main')
            self.curr_processor = None
            VisionServer2018.set_exposure(self.main_camera, 0)
        
        elif new_mode in ('driver', 'drive'):
            if self.active_camera != 'driver':
                self.switch_camera('driver')
            self.curr_processor = None

        else:
            logging.error("Unknown mode '%s'" % new_mode)
            return

        self.active_mode = new_mode
        self.nt_active_mode = self.active_mode  # make sure they are in sync
        return

    def process_image(self):
        '''Run the processor on the image to find the target'''

        # rvec, tvec return as None if no target found
        if self.curr_processor is not None:
            result = self.curr_processor.process_image(self.camera_frame)
        else:
            result = [1.0, VisionServer2018.DRIVER_MODE, 0.0, 0.0, 0.0]  # TODO: or maybe just have the result fail?

        # Send the results as one big array in order to guarantee that the results
        #  all arrive at the RoboRio at the same time
        # Value is (Timestamp, Found, Mode, distance, angle1, angle2) as a flat array.
        #  All values are floating point (required by NT).
        res = [self.image_time, ]
        if not result:
            res.extend(5*[0.0, ])
        else:
            # Found
            res.extend(result)

        self.target_info = res

        # Try to force an update of NT to the RoboRio. Docs say this may be rate-limited,
        #  so it might not happen every call.
        NetworkTables.flush()

        return

    def prepare_output_image(self):
        '''Prepare an image to send to the drivers station'''
        self.output_frame = self.camera_frame.copy()
        if self.curr_processor is not None:
            self.curr_processor.prepare_output_image(self.output_frame)
        elif self.active_mode == 'driver':
            # stored as enum: ROTATE_90_CLOCKWISE = 0, ROTATE_180 = 1, ROTATE_90_COUNTERCLOCKWISE = 2
            self.output_frame = cv2.rotate(self.camera_frame, cv2.ROTATE_90_CLOCKWISE)

        if self.image_writer_state:
            cv2.circle(self.output_frame, (20, 20), 5, (0, 0, 255), thickness=10, lineType=8, shift=0)

        return

    # ----------------------------------------------------------------------------
    # Methods which hopefully don't need to be updated

    def create_output_stream(self):
        '''Create the main image MJPEG server'''

        # Output server
        # Need to do this the hard way to set the TCP port
        self.output_stream = cscore.CvSource('camera', cscore.VideoMode.PixelFormat.kMJPEG,
                                             self.image_width, self.image_height,
                                             min(self.camera_fps, self.output_fps_limit))
        self.camera_server.addCamera(self.output_stream)
        server = self.camera_server.addServer(name='camera', port=self.output_port)
        server.setSource(self.output_stream)

        return

    @staticmethod
    def set_exposure(camera, value):
        logging.info("Setting camera exposure to '%d'" % value)
        if value == 0:
            camera.setExposureAuto()
        else:
            camera.setExposureManual(int(value))
        return

    def add_camera(self, name, device, active=True):
        '''Add a single camera and set it to active/disabled as indicated.
        Cameras are referenced by their name, so pick something unique'''

        camera = cscore.UsbCamera(name, device)
        self.camera_server.startAutomaticCapture(camera=camera)
        camera.setResolution(self.image_width, self.image_height)
        camera.setFPS(self.camera_fps)

        if name == 'main':
            self.main_camera = camera

        sink = self.camera_server.getVideo(camera=camera)
        self.camera_feeds[name] = sink
        if active:
            self.current_sink = sink
            self.active_camera = name
        else:
            # if not active, disable it to save CPU
            sink.setEnabled(False)

        return

    def switch_camera(self, name):
        '''Switch the active camera, and disable the previously active one'''

        new_cam = self.camera_feeds.get(name, None)
        if new_cam is not None:
            # disable the old camera feed
            self.current_sink.setEnabled(False)
            # enable the new one. Do this 2nd in case it is the same as the old one.
            new_cam.setEnabled(True)
            self.current_sink = new_cam
            self.active_camera = name
        else:
            logging.warning('Unknown camera %s' % name)

        return

    def run(self):
        '''Main loop. Read camera, process the image, send to the MJPEG server'''

        frame_num = 0
        errors = 0
        while True:
            try:
                # Check whether DS has asked for a different camera
                ntmode = self.nt_active_mode  # temp, for efficiency
                if ntmode != self.active_mode:
                    self.switch_mode(ntmode)

                if self.camera_frame is None:
                    self.preallocate_arrays()

                if self.tuning:
                    self.update_parameters()

                # Tell the CvSink to grab a frame from the camera and put it
                # in the source image.  If there is an error notify the output.
                frametime = 0
                try:
                    frametime, self.camera_frame = self.current_sink.grabFrame(self.camera_frame)
                except:
                    if errors < 10:
                        errors += 1
                    else:   #if greater than 10 iterations without any stream switch cameras
                        logging.warning(self.active_camera + " camera is no longer streaming. Switching cameras...")
                        if self.active_camera == 'main':
                            self.switch_camera('driver')
                        else:
                            self.switch_camera('main')
                        errors = 0
                        
                
                if frametime == 0:
                    # Send the output the error.
                    self.output_stream.notifyError(self.current_sink.getError())
                    # skip the rest of the current iteration
                    # TODO: do we need to indicate error to RoboRio?
                    continue

                frame_num += 1
                if frame_num == 30:
                    # This is a bit stupid, but you need to poke the camera *after* the first
                    #  bunch of frames has been collected.
                    self.switch_mode(VisionServer2018.INITIAL_MODE)

                # frametime = time.time() * 1e7  (ie in 1/10 microseconds)
                # convert frametime to seconds to use as the heartbeat sent to the RoboRio
                self.image_time = 1e-7 * frametime

                if self.image_writer_state:
                    self.image_writer.setImage(self.camera_frame)

                try:
                    self.process_image()
                except Exception as e:
                    logging.error('Caught processing exception: %s', e)
                    res = [self.image_time, ]
                    res.extend(5*[0.0, ])
                    self.target_info = res

                # Done. Output the marked up image, if needed
                now = time.time()
                deltat = now - self.previous_output_time
                min_deltat = 1.0 / self.output_fps_limit
                if deltat >= min_deltat:
                    self.prepare_output_image()

                    self.output_stream.putFrame(self.output_frame)
                    self.previous_output_time = now

            except Exception as e:
                # major exception. Try to keep going
                logging.error('Caught general exception: %s', e)

        return

    def run_files(self, file_list):
        '''Run routine to loop through a set of files and process each.
        Waits a couple seconds between each, and loops forever'''

        self.file_mode = True
        file_index = 0
        while True:
            if self.camera_frame is None:
                self.preallocate_arrays()

            image_file = file_list[file_index]
            print('Processing', image_file)
            file_frame = cv2.imread(image_file)
            numpy.copyto(self.camera_frame, file_frame)

            self.process_image()

            self.prepare_output_image()

            self.output_stream.putFrame(self.output_frame)
            # probably don't want to use sleep. Want something thread-compatible
            # for _ in range(4):
            time.sleep(0.5)

            file_index = (file_index + 1) % len(file_list)
        return

# -----------------------------------------------------------------------------


# syntax checkers don't like global variables, so use a simple function
def main():
    '''Main routine'''

    import argparse
    parser = argparse.ArgumentParser(description='2018 Vision Server')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('--calib', required=True, help='Calibration file for camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.test:
        # FOR TESTING, set this box as the server
        NetworkTables.enableVerboseLogging()
        NetworkTables.initialize()
    else:
        NetworkTables.initialize(server='10.28.77.2')

    server = VisionServer2018(args.calib)

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return


# Main routine
if __name__ == '__main__':
    main()
