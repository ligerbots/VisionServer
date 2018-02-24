# VisionServer

This is the LigerBots Python vision server, intended to run on a coprocessor on the robot.

The server can be "controlled" through NetworkTables. The NT variables are all under /SmartDashboard/vision.
Some interesting NT variables:
* /SmartDashboard/vision/target_info  = results of the target search. 
All values are floats, because NT arrays need to be a uniform type. Values are:
  * target_info[0] = timestamp of image in seconds. NOTE: ODROID clock is often wildly off; you can use the clock change, but not the actual value.
  * target_info[1] = success (1.0) OR failure (0.0)
  * target_info[2] = mode (1=switch, 2=cube, 3=driver/intake)
  * target_info[3] = distance to target (inches)
  * target_info[4] = angle1 to target (radians)
  * target_info[5] = angle2 of target (radians)

* /SmartDashboard/vision/active_mode = current vision processing mode
This is a **string**, with specific values.
  * "driver" = Driver camera, no target finding
  * "cube" = Intake camera, try to find cube. NOTE: RoboRio needs to turn **off** LED ring.
  * "switch" = Intake camera, try to find switch. NOTE: RoboRio needs to turn **on** LED ring.

* /SmartDashboard/vision/camera_height = current height of Intake camera (inches)

The cube mode needs to know the height of the intake camera off the ground.
This needs to come from the RoboRio, based on the winch position.
