#!/usr/bin/env python3

'''Vision server for 2022 Rapid React'''

from networktables.util import ntproperty

from visionserver import VisionServer, main
import cameras
from genericfinder import GenericFinder
# from hubfinder2022 import HubFinder2022
from fastfinder2022 import FastFinder2022 as HubFinder2022


class VisionServer2022(VisionServer):

    # Retro-reflective target finding parameters

    # Color threshold values, in HSV space
    hub_hue_low_limit = ntproperty('/SmartDashboard/vision/hub/hue_low_limit', 50,
                                   doc='Hue low limit for thresholding (rrtarget mode)')
    hub_hue_high_limit = ntproperty('/SmartDashboard/vision/hub/hue_high_limit', 80,
                                    doc='Hue high limit for thresholding (rrtarget mode)')

    hub_saturation_low_limit = ntproperty('/SmartDashboard/vision/hub/saturation_low_limit', 110,
                                          doc='Saturation low limit for thresholding (rrtarget mode)')
    hub_saturation_high_limit = ntproperty('/SmartDashboard/vision/hub/saturation_high_limit', 255,
                                           doc='Saturation high limit for thresholding (rrtarget mode)')

    hub_value_low_limit = ntproperty('/SmartDashboard/vision/hub/value_low_limit', 110,
                                     doc='Value low limit for thresholding (rrtarget mode)')
    hub_value_high_limit = ntproperty('/SmartDashboard/vision/hub/value_high_limit', 255,
                                      doc='Value high limit for thresholding (rrtarget mode)')

    # rrtarget_exposure = ntproperty('/SmartDashboard/vision/rrtarget/exposure', 0, doc='Camera exposure for rrtarget (0=auto)')

    def __init__(self, calib_dir, test_mode=False):
        super().__init__(initial_mode='shooter', test_mode=test_mode)

        self.camera_device_intake = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'
        self.camera_device_shooter = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e-video-index0'
        self.add_cameras(calib_dir)

        self.generic_finder = GenericFinder("shooter", "shooter", finder_id=4.0)
        self.add_target_finder(self.generic_finder)

        self.generic_finder_intake = GenericFinder("intake", "intake", finder_id=5.0)
        self.add_target_finder(self.generic_finder_intake)

        cam = self.cameras['shooter']
        self.hub_finder = HubFinder2022(cam.calibration_matrix, cam.distortion_matrix)
        self.add_target_finder(self.hub_finder)

        self.update_parameters()

        # DEBUG: capture an image every 100ms - this is too fast for normal testing
        self.image_writer.capture_period = 0.120

        # start in intake mode to get cameras going. Will switch to initial_mode after 1 sec.
        self.switch_mode('intake')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable
        self.hub_finder.low_limit_hsv[0] = int(self.hub_hue_low_limit)
        self.hub_finder.low_limit_hsv[1] = int(self.hub_saturation_low_limit)
        self.hub_finder.low_limit_hsv[2] = int(self.hub_value_low_limit)

        self.hub_finder.high_limit_hsv[0] = int(self.hub_hue_high_limit)
        self.hub_finder.high_limit_hsv[1] = int(self.hub_saturation_high_limit)
        self.hub_finder.high_limit_hsv[2] = int(self.hub_value_high_limit)
        return

    def add_cameras(self, calib_dir):
        '''Add the cameras'''

        # Use threading for the shooter camera.
        # The high res image slows the processing so we need all the time we can get
        cam = cameras.LogitechC930e(self.camera_server, 'shooter', self.camera_device_shooter, height=480, rotation=-90, threaded=True)
        cam.load_calibration(calib_dir)
        self.add_camera(cam, True)

        # do not need high res image for this, so also no threaded reader.
        cam = cameras.LogitechC930e(self.camera_server, 'intake', self.camera_device_intake, height=240, rotation=0)
        cam.load_calibration(calib_dir)
        self.add_camera(cam, False)
        return

    def mode_after_error(self):
        if self.active_mode == 'intake':
            return 'shooter'
        return 'intake'

    def switch_mode(self, new_mode):
        '''Switch processing mode. new_mode is the string name'''

        super().switch_mode(new_mode)

        # DEBUG: for debugging of finding the hub, save images when looking for the Hub
        self.image_writer_state = (new_mode == self.hub_finder.name)

        return


# Main routine
if __name__ == '__main__':
    main(VisionServer2022)
