import matplotlib.pyplot as plt
import Camera
import cv2 as cv
import numpy as np
import glob
from CameraPPTTypes import F32, CV_Matrix
from typing import Sequence

ROWS:    int = 5
COLUMNS: int = 7
WORLD_SCALING: F32  = F32(0.04)

image_location: str = "./lenovo_checker_calibration/*.jpg"
images: list[str]   = glob.glob(image_location)

cam: Camera.ImageReader = Camera.ImageReader(0, WORLD_SCALING, ROWS,
                                COLUMNS, camera_resolution=(1280, 720))


if len(images) == 0:
    print("No images found")
    exit()

cam.capture_calibrationImages(images)
cam.calibrate()
cam.export_calibrationDataJSON("./lenovo_checker_calibration/calibration_data.json")

rvecs: Sequence[CV_Matrix] = cam.rvecs
tvecs: Sequence[CV_Matrix] = cam.tvecs


def get_camera_positions(rvecs: Sequence[CV_Matrix], tvecs: Sequence[CV_Matrix]) -> np.ndarray:
    camera_positions: list[np.ndarray] = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv.Rodrigues(rvec)  # Convert rotation vector to matrix
        camera_position = -R.T @ tvec  # Camera position in world coordinates
        camera_positions.append(camera_position.flatten())
    return np.array(camera_positions)


camera_positions: np.ndarray = get_camera_positions(rvecs, tvecs)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the checkerboard at the origin
ax.scatter(0, 0, 0, c='r', marker='o', label='Checkerboard')

# Plot the camera positions
ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='b', marker='^', label='Camera Positions')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Checkerboard and Camera Positions')

plt.show()
