# VisionServer

This is the LigerBots Python vision server, intended to run on a coprocessor on the robot.

The server can be "controlled" through NetworkTables. The NT variables are all under /SmartDashboard/vision.
Some interesting NT variables:
* /SmartDashboard/vision/target_info  = results of the target search. (to retrieve the array from NT, use 'vision/target_info')
All values are floats, because NT arrays need to be a uniform type. Values are:
  * target_info[0] = timestamp of image in seconds.
  * target_info[1] = success (1.0) OR failure (0.0)
  * target_info[2] = mode (1=driver, 2=rrtarget)
  * target_info[3] = distance to target (inches)
  * target_info[4] = angle1 to target (radians) -- angle displacement of robot to target
  * target_info[5] = angle2 of target (radians) -- angle displacement of target to robot

* /SmartDashboard/vision/active_mode = current vision processing mode (to retrieve the value from NT, use 'vision/active_mode')
This is a **string**, with specific values.
  * "driver" = Front camera, no target finding
  * "rrtarget" = Front camera, try to find the rrtarget. NOTE: RoboRio needs to turn **on** LED ring.

* /SmartDashboard/vision/camera_height = current height of Front camera (inches)

(The cube mode needs to know the height of the intake camera off the ground.
This needs to come from the RoboRio, based on the winch position.)

Indicators (will only be present on the final stream, 1190):
- Test Mode (text): The Odroid is in test mode and set up a local set of NT. WARNING: The Odroid will not commmunicate its NT with the Roborio's NT, since they are completely different tables. To change the mode, go into the Odroid and change start_server so that args includes/not include the tag '--test'
- Recording images (red dot): When the NT variable 'write_images' is turned on, images will be saved every 1/2 second to the directory saved_images under the server directory
- Tuning On (text): When the NT variable 'tuning' is turned on, the server's NT variables will communicate with the individual finders' attributes such as HSV values. However, only tuning for HSV values is currently set up.
