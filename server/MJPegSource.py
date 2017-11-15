#!/usr/bin/env python3

# Class to add some functionality to the MJPEG server
# WARNING: because of the use of "ntproperty", there can be only one of this class

# Limit the FPS output by dropping frames

# NOTE: there is currently no programmatic way to set the "quality" (or compression) of 
#  the JPEG image. The only way to do it is in the URL used. Fetch image from:
#      http://localhost:1182/stream.mjpg?compression=30

import cscore
from networktables.util import ntproperty
from time import time

class MJPegSource( cscore.CvSource ):
    fps_limit = ntproperty( '/camera/fps_limit', 15 )

    def __init__( self, name, width, height, fps=15 ):
        cscore.CvSource.__init__( self, name, cscore.VideoMode.PixelFormat.kMJPEG, width, height, fps )
    
        self.prevFrameT = time()
        self.fps_limit = fps
        
        cscore.CameraServer.getInstance().startAutomaticCapture( camera=self )
        return
    
    def putFrame( self, frame ):
        now = time()
        dt = now - self.prevFrameT
        minDT = 1.0 / self.fps_limit()
        if dt < minDT: return

        cscore.CvSource.putFrame( self, frame )
        self.prevFrameT = now
        return
