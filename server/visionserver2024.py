#!/usr/bin/env python3

'''Vision server for 2024 Crescendo NOTE finding'''

from visionserver import VisionServer, main
import cameras
from genericfinder import GenericFinder
from notefinder2024 import NoteFinder2024


class VisionServer2024(VisionServer):
    def __init__(self, calib_dir, test_mode=False):
        super().__init__(initial_mode='notefinder', test_mode=test_mode)

        # self.camera_device = '/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_DF7AF0BE-video-index0'
        self.camera_device = '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB_Live_camera_SN0001-video-index0'
        self.add_cameras(calib_dir)

        # process purely for driver view
        self.generic_finder = GenericFinder(name="driver", camera="intake", finder_id=4.0)
        self.add_target_finder(self.generic_finder)

        cam = self.cameras['intake']
        self.note_finder = NoteFinder2024(cam.calibration_matrix, cam.distortion_matrix)
        self.add_target_finder(self.note_finder)

        # DEBUG: capture an image every 100ms - this is too fast for normal testing
        # self.image_writer.capture_period = 0.120

        # start in intake mode to get cameras going. Will switch to initial_mode after 1 sec.
        self.switch_mode('driver')
        return

    def add_cameras(self, calib_dir):
        '''Add the cameras'''

        # Use threading for the shooter camera.
        # The high res image slows the processing so we need all the time we can get
        cam = cameras.LogitechC930e(self.camera_server, 'intake', self.camera_device, height=480, rotation=0, threaded=True)
        cam.load_calibration(calib_dir)
        self.add_camera(cam, True)
        return

    # def switch_mode(self, new_mode):
    #     '''Switch processing mode. new_mode is the string name'''

    #     super().switch_mode(new_mode)

    #     # DEBUG: for debugging of finding the hub, save images when looking for the Hub
    #     self.image_writer_state = (new_mode == self.hub_finder.name)

    #     return


# Main routine
if __name__ == '__main__':
    main(VisionServer2024)
