#!/usr/bin/env python3

'''Defines a class for which each year's subclass vision server inherits from'''

import time
import cv2
import numpy
import logging

import cscore
from cscore.imagewriter import ImageWriter
from networktables.util import ntproperty
from networktables import NetworkTables

from threadedcamera import ThreadedCamera


class VisionServer:
    '''Base class for the VisionServer'''

    # NetworkTable parameters

    # frame rate is pretty variable, so set this a fair bit higher than what you really want
    # using a large number for no limit
    output_fps_limit = ntproperty('/SmartDashboard/vision/output_fps_limit', 1000,
                                  doc='FPS limit of frames sent to MJPEG server')

    # fix the TCP port for the main video, so it does not change with multiple cameras
    output_port = ntproperty('/SmartDashboard/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup.
    tuning = ntproperty('/SmartDashboard/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')

    # Logitech c930 are wide-screen cameras, so 320x180 has the biggest FOV
    image_width = ntproperty('/SmartDashboard/vision/width', 424, writeDefault=False, doc='Image width')
    image_height = ntproperty('/SmartDashboard/vision/height', 240, writeDefault=False, doc='Image height')
    camera_fps = ntproperty('/SmartDashboard/vision/fps', 30, writeDefault=False, doc='FPS from camera')

    image_writer_state = ntproperty('/SmartDashboard/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # This ought to be a Choosable, but the Python implementation is lame. Use a string for now.
    # This is the NT variable, which can be set from the Driver's station
    # Use "0" for the initial value; needs to be set by the subclass.
    nt_active_mode = ntproperty('/SmartDashboard/vision/active_mode', 'default', doc='Active mode')

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time.
    # Value is (time, success, finder_id, distance, angle1, angle2) as a flat array.
    # All values are floating point (required by NT).
    target_info = ntproperty('/SmartDashboard/vision/target_info', 6 * [0.0, ],
                             doc='Packed array of target info: time, success, finder_id, distance, angle1, angle2')

    def __init__(self, initial_mode, test_mode=False):
        self.test_mode = test_mode
        # for processing stored files and no camera
        self.file_mode = False

        # time of each frame. Sent to the RoboRio as a heartbeat
        self.image_time = 0

        self.camera_server = cscore.CameraServer.getInstance()
        self.camera_server.enableLogging()

        self.video_readers = {}
        self.current_reader = None
        self.cameras = {}
        self.active_camera = None

        self.create_output_stream()

        # Dictionary of finders. The key is the string "name" of the finder.
        self.target_finders = {}

        # Initial mode for start of match.
        # VisionServer switches to this mode after a second, to get the cameras initialized
        self.initial_mode = initial_mode
        self.nt_active_mode = self.initial_mode

        # active mode. To be compared to nt_active_mode to see if it has changed
        self.active_mode = None

        self.curr_finder = None

        # rate limit parameters
        self.previous_output_time = time.time()
        self.camera_frame = None
        self.output_frame = None

        # Last error message (from cscore)
        self.error_msg = None

        # if asked, save image every 1/2 second
        # images are saved under the directory 'saved_images' in the current directory
        #  (ie current directory when the server is started)
        self.image_writer = ImageWriter(location_root='./saved_images',
                                        capture_period=0.5, image_format='jpg')

        return

    # --------------------------------------------------------------------------------
    # Methods generally customized each year

    """Methods you should/must include"""

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable
        raise NotImplementedError

    # --------------------------------------------------------------------------------
    # Methods which hopefully don't need to be updated

    def preallocate_arrays(self):
        '''Preallocate the intermediate result image arrays'''

        # NOTE: shape is (height, width, #bytes)
        # self.camera_frame = numpy.zeros(shape=(int(self.image_height), int(self.image_width), 3),
        #                                 dtype=numpy.uint8)

        return

    def create_output_stream(self):
        '''Create the main image MJPEG server'''

        # Output server
        # Need to do this the hard way to set the TCP port
        self.output_stream = cscore.CvSource('camera', cscore.VideoMode.PixelFormat.kMJPEG,
                                             int(self.image_width), int(self.image_height),
                                             int(min(self.camera_fps, self.output_fps_limit)))
        self.camera_server.addCamera(self.output_stream)
        server = self.camera_server.addServer(name='camera', port=int(self.output_port))
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

    @staticmethod
    def set_camera_property(camera, name, value):
        '''Set a camera property, such as auto_focus'''

        logging.info("Setting camera property '{}' to '{}'".format(name, value))
        try:
            try:
                propVal = int(value)
            except ValueError:
                camera.getProperty(name).setString(value)
            else:
                camera.getProperty(name).set(propVal)
        except Exception as e:
            logging.warn("Unable to set property '{}': {}".format(name, e))

        return

    def add_camera(self, name, device, active=True):
        '''Add a single camera and set it to active/disabled as indicated.
        Cameras are referenced by their name, so pick something unique'''

        camera = cscore.UsbCamera(name, device)
        # remember the camera object, in case we need to adjust it (eg the exposure)
        self.cameras[name] = camera

        self.camera_server.startAutomaticCapture(camera=camera)

        # PS Eye camera needs to have its pixelformat set
        # camera.setPixelFormat(cscore.VideoMode.PixelFormat.kYUYV)

        camera.setResolution(int(self.image_width), int(self.image_height))
        camera.setFPS(int(self.camera_fps))

        # keep the camera open for faster switching
        camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

        # set the camera for no auto focus, focus at infinity
        # TODO: different cameras have different properties
        # NOTE: order does matter
        VisionServer.set_camera_property(camera, 'focus_auto', 0)
        VisionServer.set_camera_property(camera, 'focus_absolute', 0)

        mode = camera.getVideoMode()
        logging.info("camera '%s' pixel format = %s, %dx%d, %dFPS", name,
                     mode.pixelFormat, mode.width, mode.height, mode.fps)

        reader = ThreadedCamera(self.camera_server.getVideo(camera=camera)).start()
        self.video_readers[name] = reader
        if active:
            self.current_reader = reader
            self.active_camera = name

        return

    def switch_camera(self, name):
        '''Switch the active camera, and disable the previously active one'''

        new_reader = self.video_readers.get(name, None)
        if new_reader is not None:
            self.current_reader = new_reader
            self.active_camera = name
        else:
            logging.error('Unknown camera %s' % name)

        return

    def add_target_finder(self, finder):
        logging.info("Adding target finder '{}' id {}".format(finder.name, finder.finder_id))
        self.target_finders[finder.name] = finder
        return

    def switch_mode(self, new_mode):
        '''Switch processing mode. new_mode is the string name'''

        try:
            logging.info("Switching mode to '%s'" % new_mode)
            finder = self.target_finders.get(new_mode, None)
            if finder is not None:
                if self.active_camera != finder.camera:
                    self.switch_camera(finder.camera)

                self.curr_finder = finder
                self.set_exposure(self.cameras[finder.camera], finder.exposure)
                self.active_mode = new_mode
            else:
                logging.error("Unknown mode '%s'" % new_mode)

            self.nt_active_mode = self.active_mode  # make sure they are in sync
        except Exception as e:
            logging.error('Exception when switching mode: %s', e)

        return

    def process_image(self):
        # rvec, tvec return as None if no target found
        try:
            result = self.curr_finder.process_image(self.camera_frame)
        except Exception as e:
            logging.error("Exception caught in process_image(): %s", e)
            result = (0.0, 0.0, 0.0, 0.0, 0.0)

        return result

    def prepare_output_image(self):
        '''Create the image to send to the Driver station.
        Finder is expected to *copy* the input image, as needed'''

        try:
            if self.curr_finder is None:
                self.output_frame = self.camera_frame.copy()
            else:
                base_frame = self.camera_frame

                # Finders are allowed to designate a different image to stream to the DS
                cam = self.curr_finder.stream_camera
                if cam is not None:
                    rdr = self.video_readers.get(cam, None)
                    if rdr is not None:
                        _, base_frame = rdr.get_frame()  # does not wait

                self.output_frame = self.curr_finder.prepare_output_image(base_frame)

            image_shape = self.output_frame.shape
            if image_shape[0] < 400:  # test on height
                dotrad = 3
                fontscale = 0.4
                fontthick = 1
            else:
                dotrad = 5
                fontscale = 0.75
                fontthick = 2

            # If saving images, add a little red "Recording" dot in upper left
            if self.image_writer_state:
                cv2.circle(self.output_frame, (20, 20), dotrad, (0, 0, 255), thickness=2*dotrad, lineType=8, shift=0)

            # If tuning mode is on, add text to the upper left corner saying "Tuning On"
            if self.tuning:
                cv2.putText(self.output_frame, "TUNING ON", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (255, 0, 0), thickness=fontthick)

            # If test mode (ie running the NT server), give a warning
            if self.test_mode:
                cv2.putText(self.output_frame, "TEST MODE", (5, image_shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 255), thickness=fontthick)

        except Exception as e:
            logging.error("Exception caught in prepare_output_image(): %s", e)

        return

    def run(self):
        '''Main loop. Read camera, process the image, send to the MJPEG server'''

        frame_num = 0
        errors = 0

        fps_count = 0
        fps_startt = time.time()
        while True:
            try:
                # Check whether DS has asked for a different camera
                ntmode = self.nt_active_mode  # temp, for efficiency
                if ntmode != self.active_mode:
                    self.switch_mode(ntmode)

                # if self.camera_frame is None:
                #     self.preallocate_arrays()

                if self.tuning:
                    self.update_parameters()

                # Tell the CvReader to grab a frame from the camera and put it
                # in the source image.  Frametime==0 on error
                frametime, self.camera_frame = self.current_reader.next_frame()
                frame_num += 1

                if frametime == 0:
                    # ERROR!!
                    self.error_msg = self.current_reader.sink.getError()

                    if errors < 10:
                        errors += 1
                    else:   # if 10 or more iterations without any stream switch cameras
                        logging.warning(self.active_camera + " camera is no longer streaming. Switching cameras...")
                        self.switch_mode(self.mode_after_error())
                        errors = 0

                    target_res = [time.time(), ]
                    target_res.extend(5*[0.0, ])
                else:
                    self.error_msg = None
                    errors = 0

                    if self.image_writer_state:
                        self.image_writer.setImage(self.camera_frame)

                    # frametime = time.time() * 1e8  (ie in 1/100 microseconds)
                    # convert frametime to seconds to use as the heartbeat sent to the RoboRio
                    target_res = [1e-8 * frametime, ]

                    proc_result = self.process_image()
                    target_res.extend(proc_result)

                # Send the results as one big array in order to guarantee that the results
                #  all arrive at the RoboRio at the same time
                # Value is (Timestamp, Found, Mode, distance, angle1, angle2) as a flat array.
                #  All values are floating point (required by NT).
                self.target_info = target_res

                # Try to force an update of NT to the RoboRio. Docs say this may be rate-limited,
                #  so it might not happen every call.
                # NetworkTables.flush()

                # Done. Output the marked up image, if needed
                # Note this can also be done via the URL, but this is more efficient
                now = time.time()
                deltat = now - self.previous_output_time
                min_deltat = 1.0 / self.output_fps_limit
                if deltat >= min_deltat:
                    self.prepare_output_image()
                    self.output_stream.putFrame(self.output_frame)
                    self.previous_output_time = now

                if frame_num == 30:
                    # This is a bit stupid, but you need to poke the camera *after* the first
                    #  bunch of frames has been collected.
                    self.switch_mode(self.initial_mode)

                fps_count += 1
                if fps_count == 150:
                    endt = time.time()
                    dt = endt - fps_startt
                    logging.info("{0} frames in {1:.3f} seconds = {2:.2f} FPS".format(fps_count, dt, fps_count/dt))
                    fps_count = 0
                    fps_startt = endt

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


def wait_on_nt_connect(max_delay=10):
    cnt = 0
    while True:
        if NetworkTables.isConnected():
            logging.info('Connect to NetworkTables after %d seconds', cnt)
            return

        if cnt >= max_delay:
            break

        if cnt > 0 and cnt % 5 == 0:
            logging.warning("Still waiting to connect to NT (%d sec)", cnt)
        time.sleep(1)
        cnt += 1

    logging.warning("Failed to connect to NetworkTables after %d seconds. Continuing", cnt)
    return


# syntax checkers don't like global variables, so use a simple function
def main(server_type):
    '''Main routine'''

    import argparse
    parser = argparse.ArgumentParser(description='2018 Vision Server')
    parser.add_argument('--calib', required=True, help='Calibration file for camera')
    parser.add_argument('--test', action='store_true', help='Run in local test mode')
    parser.add_argument('--delay', type=int, default=0, help='Max delay trying to connect to NT server (seconds)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('--files', action='store_true', help='Process input files instead of camera')
    parser.add_argument('input_files', nargs='*', help='input files')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    if args.test:
        # FOR TESTING, set this box as the server
        NetworkTables.enableVerboseLogging()
        NetworkTables.startServer()
    else:
        if args.verbose:
            # Turn up the noise from NetworkTables. VERY noisy!
            # DO NOT do this during competition, unless you are really sure
            NetworkTables.enableVerboseLogging()
        # NetworkTables.startClient('10.28.77.2')
        # Try startClientTeam() method; it auto tries a whole bunch of standard addresses
        NetworkTables.startClientTeam(2877)
        if args.delay > 0:
            wait_on_nt_connect(args.delay)

    server = server_type(calib_file=args.calib, test_mode=args.test)

    if args.files:
        if not args.input_files:
            parser.usage()

        server.run_files(args.input_files)
    else:
        server.run()
    return
