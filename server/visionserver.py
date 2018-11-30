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

class visionserver:
    # NetworkTable parameters
    
    output_fps_limit = ntproperty('/SmartDashboard/vision/output_fps_limit', 17,
                                  doc='FPS limit of frames sent to MJPEG server')
    
    #TCP port for main image output
    output_port = output_port = ntproperty('/SmartDashboard/vision/output_port', 1190,
                             doc='TCP port for main image output')

    # Operation modes. Force the value on startup
        #Tuning mode. Reads processing parameters each time.
    tuning = ntproperty('/SmartDashboard/vision/tuning', False, writeDefault=True,
                        doc='Tuning mode. Reads processing parameters each time.')
        #Image width
    image_width = ntproperty('/SmartDashboard/vision/width', 320, writeDefault=False, doc='Image width')
        #Image height
    image_height = ntproperty('/SmartDashboard/vision/height', 240, writeDefault=False, doc='Image height')
        #FPS from camera
    camera_fps = ntproperty('/SmartDashboard/vision/fps', 30, writeDefault=False, doc='FPS from camera')

    # Object finding parameters -- add more for additional objects

    #TODO: where should the main thresholding values be kept under network tables?
        
    # Color threshold values, in HSV space
        #Hue low limit for thresholding
    main_hue_low_limit = ntproperty('/SmartDashboard/vision/main_object/hue_low_limit', 25,
                                    doc='Hue low limit for thresholding (cube mode)')
        #Hue high limit for thresholding
    main_hue_high_limit = ntproperty('/SmartDashboard/vision/main_object/hue_high_limit', 75,
                                     doc='Hue high limit for thresholding (cube mode)')
        #Saturation low limit for thresholding
    main_saturation_low_limit = ntproperty('/SmartDashboard/vision/main_object/saturation_low_limit', 95,
                                           doc='Saturation low limit for thresholding (cube mode)')
        #Saturation high limit for thresholding
    main_saturation_high_limit = ntproperty('/SmartDashboard/vision/main_object/saturation_high_limit', 255,
                                            doc='Saturation high limit for thresholding (cube mode)')
        #Value low limit for thresholding
    main_value_low_limit = ntproperty('/SmartDashboard/vision/main_object/value_low_limit', 95,
                                      doc='Value low limit for thresholding (cube mode)')
        #Value high limit for thresholding
    main_value_high_limit = ntproperty('/SmartDashboard/vision/main_object/value_high_limit', 255,
                                       doc='Value high limit for thresholding (cube mode)')

    main_exposure = ntproperty('/SmartDashboard/vision/main_object/exposure', 0, doc='Camera exposure for main_object (0=auto)')


    #Turn on saving of images
    image_writer_state = ntproperty('/SmartDashboard/vision/write_images', False, writeDefault=True,
                                    doc='Turn on saving of images')

    # This ought to be a Choosable, but the Python implementation is lame. Use a string for now.
    # This is the NT variable, which can be set from the Driver's station -- Active mode
    nt_active_mode = ntproperty('/SmartDashboard/vision/active_mode', INITIAL_MODE, doc='Active mode')  #TODO: create INITIAL_MODE

    # Targeting info sent to RoboRio
    # Send the results as one big array in order to guarantee that the results
    #  all arrive at the RoboRio at the same time
    # Maybe: Value is (Found, tvec, rvec) as a flat array. All values are floating point (required by NT).
    target_info = valueError

    
    def __init__(self, calib_file):
        self.valueError = ValueError("Cannot get value that is not yet set -- this is an abstract property!")
        
        self.INITIAL_MODE = 'main'
        
        
        
        
        
        
        
        