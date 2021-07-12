#!/usr/bin/python3

from time import time
import logging


def opencv_time(device_name, width=640, height=480, fps=30, max_count=600):
    '''Use standard OpenCV read to test frame rate'''

    import cv2

    logging.info("Testing frame rate with OpenCV")

    cap = cv2.VideoCapture(device_name)

    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    # print('fourcc=', cap.get(cv2.CAP_PROP_FOURCC))

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    logging.info('auto exposure %d', cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # cap.set(cv2.CAP_PROP_BACKLIGHT, 1)

    logging.info('size %dx%d', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info('backlight %d', cap.get(cv2.CAP_PROP_BACKLIGHT))
    logging.info('fps %d', cap.get(cv2.CAP_PROP_FPS))

    fps_count = 0
    fps_startt = time()
    total_count = 0

    while total_count < max_count:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            logging.info('error capturing image ret=', ret)
            break

        fps_count += 1
        total_count += 1

        if fps_count >= 150:
            endt = time()
            dt = endt - fps_startt
            logging.info("{0} frames in {1:.3f} seconds = {2:.2f} FPS".format(fps_count, dt, fps_count/dt))
            fps_count = 0
            fps_startt = endt

    cap.release()
    return


def cscore_set_property(camera, name, value):
    '''Set a camera property, such as auto_focus'''

    logging.debug(f"Setting camera property '{name}' to '{value}'")
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


def cscore_time(device_name, width=640, height=480, fps=30, max_count=600):
    '''Use CSCore read to test frame rate'''

    import cscore

    logging.info("Testing frame rate with CSCore")

    camera_server = cscore.CameraServer.getInstance()

    camera = cscore.UsbCamera("camera", device_name)
    camera_server.startAutomaticCapture(camera=camera)
    # keep the camera open for faster switching
    # camera.setConnectionStrategy(cscore.VideoSource.ConnectionStrategy.kKeepOpen)

    camera.setResolution(width, height)
    camera.setFPS(fps)

    # set the camera for no auto focus, focus at infinity
    # NOTE: order does matter
    cscore_set_property(camera, 'focus_auto', 0)
    cscore_set_property(camera, 'focus_absolute', 0)
    sink = camera_server.getVideo(camera=camera)

    mode = camera.getVideoMode()
    logging.info("pixel format = %s, %dx%d, %dFPS", mode.pixelFormat, mode.width, mode.height, mode.fps)

    camera_frame = None
    # first frame is slow and often has an error
    frametime, camera_frame = sink.grabFrame(camera_frame)

    fps_count = 0
    fps_startt = time()
    total_count = 0

    while total_count < max_count:
        # Capture frame-by-frame
        frametime, camera_frame = sink.grabFrame(camera_frame)
        if frametime == 0:
            # ERROR!!
            error_msg = sink.getError()
            logging.error(error_msg)
            break

        fps_count += 1
        total_count += 1

        if fps_count >= 150:
            endt = time()
            dt = endt - fps_startt
            logging.info("{0} frames in {1:.3f} seconds = {2:.2f} FPS".format(fps_count, dt, fps_count/dt))
            fps_count = 0
            fps_startt = endt
    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Calibration utility')
    parser.add_argument('--opencv', action='store_true', help='Use OpenCV read')
    parser.add_argument('--cscore', action='store_true', help='Use CSCore read')
    parser.add_argument('--width', '-W', type=int, default=640, help='Image Width')
    parser.add_argument('--height', '-H', type=int, default=480, help='Image Height')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose. Turn up debug messages')
    parser.add_argument('camera', nargs=1, help='Camera device')

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    if args.cscore:
        cscore_time(args.camera[0], width=args.width, height=args.height)
    else:
        opencv_time(args.camera[0], width=args.width, height=args.height)
