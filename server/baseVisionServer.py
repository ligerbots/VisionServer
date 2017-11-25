#!/usr/bin/env python3

# Base camera server
# Create a subclass with this years image processing to actually use

import cv2
import numpy 
import time

import cscore
from networktables.util import ntproperty

class BaseVisionServer( object ):
    # NetworkTable parameters
    output_fps_limit = ntproperty( '/vision/output_fps_limit', 15, doc='FPS limit of frames sent to MJPEG server' )

    # Operation modes. Force the value on startup.
    #tuning = ntproperty( '/vision/tuning', False, writeDefault=True, doc='Tuning mode. Reads processing parameters each time.' )
    restart = ntproperty( '/vision/restart', False, writeDefault=True, doc='Restart algorithm. Needed after certain param changes.' )
    
    image_width = ntproperty( '/vision/width', 640, writeDefault=False, doc='Image width' )
    image_height = ntproperty( '/vision/height', 480, writeDefault=False, doc='Image height' )
    camera_fps = ntproperty( '/vision/fps', 30, writeDefault=False, doc='FPS from camera' )

    def __init__( self ):
        # for processing stored files and no camera
        self.file_mode = False
        
        self.cameraServer = cscore.CameraServer.getInstance()
        self.cameraServer.enableLogging()

        self.cameraFeeds = {}
        self.currentCamera = None
        
        self.outputStream = None

        # rate limit parameters
        self.previousOutputTime = time.time()
        
        return

    #--------------------------------------------------------------------------------
    # Methods meant to be overridden by a subclass
    
    def addCameras( self ):
        # add a single camera at /dev/video0
        self.addCamera( 'main', 0, True )
        return
    
    def preallocateArrays( self ):
        # MAGIC: expect these two variables
        # Allocating new images is very expensive, always try to preallocate

        # NOTE: shape is (height, width,#bytes)
        self.camera_frame = numpy.zeros( shape=( self.image_height, self.image_width, 3 ), dtype=numpy.uint8 )
        self.output_frame = self.camera_frame
        return
    
    def processImage( self ):
        return

    def prepareOutputImage( self ):
        # Dummy: Put a rectangle on the image
        cv2.rectangle( self.camera_frame, (100, 100), (400, 400), (0, 255, 0), 5 )
        return

    #--------------------------------------------------------------------------------
    # Methods which hopefully don't need to be overridden
    
    def initialize( self ):
        if not self.file_mode:
            self.addCameras()
        
        # Output server
        self.outputStream = cscore.CvSource( 'camera', cscore.VideoMode.PixelFormat.kMJPEG,
                                             self.image_width, self.image_height,
                                             min( self.camera_fps, self.output_fps_limit ) )
        self.cameraServer.startAutomaticCapture( camera=self.outputStream )
        
        self.preallocateArrays()
        return
    
    def addCamera( self, name, device, active=True ):
        # base class: add a single camera. Override this if needed!
        # initialize the (main) camera
        camera = cscore.UsbCamera( name, device )
        self.cameraServer.startAutomaticCapture( camera=camera )
        camera.setResolution( self.image_width, self.image_height )

        sink = self.cameraServer.getVideo( camera=camera )
        self.cameraFeeds[ name ] = sink
        if active:
            self.currentCamera = sink
        else:
            # if not active, disable it to save CPU
            self.setEnabled( False )

        return

    def switchCamera( self, name ):
        newCam  = self.cameras.get( name, None )
        if newCam is not None:
            # disable the old camera feed
            self.currentCamera.setEnabled( False )
            # enable the new one. Do this 2nd in case it is the same as the old one.
            newCam.setEnabled( True )
            self.currentCamera = newCam

        return
    
    def run( self ):
        while True:
            if self.currentCamera is None or self.restart:
                self.initialize()
                self.restart = False

            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            frametime, self.camera_frame = self.currentCamera.grabFrame( self.camera_frame )
            if frametime == 0:
                # Send the output the error.
                self.outputStream.notifyError( self.currentCamera.getError() );
                # skip the rest of the current iteration
                continue

            self.processImage()
            
            # Done. Output the marked up image, if needed
            now = time.time()
            dt = now - self.previousOutputTime
            minDT = 1.0 / self.output_fps_limit
            if dt >= minDT:
                self.prepareOutputImage()

                self.outputStream.putFrame( self.output_frame )
                self.previousOutputTime = now
                
        return

    def run_files( self, file_list ):
        '''Run routine to loop through a set of files and process each. Waits a couple seconds between each, 
        and loops forever'''
        
        self.file_mode = True
        file_index = 0
        while True:
            if self.outputStream is None or self.restart:
                self.initialize()
                self.restart = False

            imageFile = file_list[ file_index ]
            print( 'Processing', imageFile )
            file_frame = cv2.imread( imageFile )
            numpy.copyto( self.camera_frame, file_frame )

            self.processImage()

            self.prepareOutputImage()
            
            self.outputStream.putFrame( self.output_frame )
            # probably don't want to use sleep. Want something thread-compatible
            for i in range(4): time.sleep(0.5)
                
            file_index = ( file_index + 1 ) % len( file_list )
        return
    
# Simple main routine for testing purposes
# Normally, you would create a subclass and put the main program there.
if __name__ == '__main__':
    from networktables import NetworkTables
    import argparse

    parser = argparse.ArgumentParser( description='Base Vision Server' )
    parser.add_argument( '--files', action='store_true', help='Process input files instead of camera' )
    parser.add_argument( 'input_files', nargs='*', help='input files' )

    args = parser.parse_args()
    
    # To see messages from networktables, you must setup logging
    import logging
    logging.basicConfig( level=logging.DEBUG )
    
    # FOR TESTING, set this box as the server
    NetworkTables.enableVerboseLogging()
    NetworkTables.initialize()

    server = BaseVisionServer()
    if args.files:
        server.run_files( args.input_files )
    else:
        server.run()
