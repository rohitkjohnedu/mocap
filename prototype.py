import cv2 as cv
import numpy as np

from Camera import StereoCamera, Aruco, convert_Aruco2CorrespondingPointsList, WebCam
from CameraPPTTypes import F32

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ROWS = 5
COLUMNS = 7
WORLD_SCALING = F32(0.04)

camera_ids: list[int] = [0, 1]
camera_list: list[WebCam] = []

for id in camera_ids:
    cam = WebCam(id, WORLD_SCALING, ROWS, COLUMNS)
    cam.initialize_camera()
    camera_list.append(cam)

print("starting feed")


while True:
    for cam in camera_list:
        cam.capture_frame()
        cam.show_frame()

    key: int = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        print("done capturing images")
        break
    elif key == 13:  # Enter key
        print("capturing image")
        for cam in camera_list:
            cam.calibration_images.append(cam.current_frame)    


for cam in camera_list:
    cam.calibrate()


print("calibration done")

stereoSys = StereoCamera(camera_list)
stereoSys.calibrate_stereo()
print("stereo calibration done")


# get aruco positions
aruco_dict: cv.aruco.Dictionary         = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
parameters: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()

# Enable interactive mode
plt.ion()
fig  = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-0.5, 0.5)
ax.set_ylim3d(-0.5, 0.5)
ax.set_zlim3d(10, 30)


def plot_aruco(ax, p3ds, connections):
    for _c in connections:
        ax.plot(
            xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]],
            ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
            zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]],
            c='red')


connectivity: list[list[int]] = [[0, 1], [1, 2], [2, 3], [3, 0]]


while True:
    aruco_poses: list[dict[int, Aruco]] = []
    for cam in camera_list:
        id = cam.id
        cam.capture_frame()
        aru = cam.get_arucoImagePose(aruco_dict, parameters)
        aruco_poses.append(aru)

    for cam in camera_list:
        cam.capture_frame()
        cam.show_frame()

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    aruco_empty_q = [ar_dict != {} for ar_dict in aruco_poses]

    if not all(aruco_empty_q):
        print("Aruco not detected in all cameras. Skipping this frame.", end='\r')
        aruco_detected = False  # Update the flag to indicate markers are not detected
        continue
    else:
        print(" " * 55, end='\r')  # Clear the message
        aruco_detected = True  # Update the flag to indicate markers are detected

    poses_0 = convert_Aruco2CorrespondingPointsList([ar_dict[0] for ar_dict in aruco_poses])
    poses_0 = np.array(poses_0)
    pos_0_3D = np.array([stereoSys.triangulate_point(no) for no in poses_0])

    # Clear the previous plot
    ax.cla()
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(10, 30)

    # Plot the updated data
    plot_aruco(ax, pos_0_3D, connectivity)

    # Update the view
    # ax.view_init(elev=90, azim=90)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.1, 1]))

    # Draw the updated plot
    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update



[cam.release_camera() for cam in camera_list]
cv.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
# results = pstats.Stats(pr)
# results.sort_stats(pstats.SortKey.TIME)
# results.dump_stats("results2.prof")