#!/usr/bin/env python3

'''Vision server for 2020 Infinite Recharge'''

from networktables.util import ntproperty

from visionserver import VisionServer, main
import cameras
from genericfinder import GenericFinder
from goalfinder2020 import GoalFinder2020
from ballfinder2020 import BallFinder2020
from hopperfinder2020 import HopperFinder2020


class VisionServer2020(VisionServer):

    # Retro-reflective target finding parameters

    # Color threshold values, in HSV space
    rrtarget_hue_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_low_limit', 65,
                                        doc='Hue low limit for thresholding (rrtarget mode)')
    rrtarget_hue_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/hue_high_limit', 100,
                                         doc='Hue high limit for thresholding (rrtarget mode)')

    rrtarget_saturation_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_low_limit', 75,
                                               doc='Saturation low limit for thresholding (rrtarget mode)')
    rrtarget_saturation_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/saturation_high_limit', 255,
                                                doc='Saturation high limit for thresholding (rrtarget mode)')

    rrtarget_value_low_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_low_limit', 135,
                                          doc='Value low limit for thresholding (rrtarget mode)')
    rrtarget_value_high_limit = ntproperty('/SmartDashboard/vision/rrtarget/value_high_limit', 255,
                                           doc='Value high limit for thresholding (rrtarget mode)')

    rrtarget_exposure = ntproperty('/SmartDashboard/vision/rrtarget/exposure', 0, doc='Camera exposure for rrtarget (0=auto)')

    def __init__(self, calib_file, test_mode=False):
        super().__init__(initial_mode='intake', test_mode=test_mode)

        self.camera_device_shooter = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'    # for driver and rrtarget processing
        self.camera_device_intake = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_70E19A9E-video-index0'    # for line and hatch processing
        self.add_cameras()

        self.generic_finder = GenericFinder("shooter", "shooter", finder_id=4.0)
        self.add_target_finder(self.generic_finder)

        self.generic_finder_intake = GenericFinder("intake", "intake", finder_id=5.0)
        self.add_target_finder(self.generic_finder_intake)

        self.goal_finder = GoalFinder2020(calib_file)
        self.add_target_finder(self.goal_finder)

        self.ball_finder = BallFinder2020(calib_file)
        self.add_target_finder(self.ball_finder)

        self.hopper_finder = HopperFinder2020(calib_file)
        self.add_target_finder(self.hopper_finder)

        self.update_parameters()

        # start in shooter mode to get cameras going. Will switch to 'intake' after 1 sec.
        self.switch_mode('shooter')
        return

    def update_parameters(self):
        '''Update processing parameters from NetworkTables values.
        Only do this on startup or if "tuning" is on, for efficiency'''

        # Make sure to add any additional created properties which should be changeable

        self.goal_finder.set_color_thresholds(self.rrtarget_hue_low_limit, self.rrtarget_hue_high_limit,
                                              self.rrtarget_saturation_low_limit, self.rrtarget_saturation_high_limit,
                                              self.rrtarget_value_low_limit, self.rrtarget_value_high_limit)
        return

    def add_cameras(self):
        '''Add the cameras'''

        self.add_camera(cameras.LogitechC930e(self.camera_server, 'shooter', self.camera_device_shooter, height=480, rotation=90), True)
        self.add_camera(cameras.LogitechC930e(self.camera_server, 'intake', self.camera_device_intake, height=240, rotation=90), False)
        return

    def mode_after_error(self):
        if self.active_mode == 'intake':
            return 'shooter'
        return 'intake'


# Main routine
if __name__ == '__main__':
    main(VisionServer2020)
