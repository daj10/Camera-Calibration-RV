# camera_calibration
Camera calibration using Python and OpenCV

Image calibration is the process of transforming 3D objects from the real world into a 2D image.
The goal is to maintain the size of the object whatever the position in the image by using a set of known 3D points (X,Y, Z) and their corresponding image coordinates (u, v).
To calibrate a camera, it is necessary to determine: 
* The distortion coefficients;
* The camera matrix, which allows the points of 3D objects to be transformed into 2D image points.
