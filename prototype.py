import cv2 as cv
import numpy as np
import glob
import multiprocessing
from multiprocessing import Queue

from Camera import ImageReader, StereoCamera, Aruco, convert_Aruco2CorrespondingPointsList, WebCam
from cv2 import aruco
from CameraPPTTypes import F32

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import cProfile
import pstats



def capture_images_from_camera(index, world_scaling, rows, columns, command_queue):
    cam = WebCam(index, world_scaling, rows, columns, command_queue)
    cam.capture_calibrationImages()
    cam.calibrate(True)
    return cam


def main():
    # List of camera indices
    camera_indices = [0, 1]  # Add all your camera indices here

    ROWS    = 5
    COLUMNS = 7
    WORLD_SCALING: F32 = np.float32(0.04)

    # Create queues for sending commands to each camera process
    command_queues = [Queue() for _ in camera_indices]

    # Create and start processes for each camera
    processes = []

    for i, command_queue in zip(camera_indices, command_queues):
        process = multiprocessing.Process(target=capture_images_from_camera, args=(i, WORLD_SCALING, ROWS, COLUMNS, command_queue))
        processes.append(process)
        process.start()

    try:
        while True:
            user_input = input("Press Enter to capture an image or type 'q' to quit: \n").strip().lower()
            if user_input == 'q':
                for queue in command_queues:
                    queue.put('quit')
                break
            elif user_input == '':
                for queue in command_queues:
                    queue.put('capture')
    finally:
        for process in processes:
            process.join()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()





# # # with cProfile.Profile() as pr:
# # for i in camera_indeces:
# #     cam = WebCam(i, WORLD_SCALING, ROWS, COLUMNS)
# #     cam.capture_calibrationImages()
# #     cam.show_frame()
# #     cam.calibrate()
# #     cameras.append(cam)


# stereoSys = StereoCamera(cameras)
# stereoSys.calibrate_stereo()

# # get aruco positions
# aruco_dict: cv.aruco.Dictionary         = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
# parameters: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()

# aruco_poses: list[dict[int, Aruco]] = []

# # Enable interactive mode
# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim3d(-15, 5)
# ax.set_ylim3d(-10, 10)
# ax.set_zlim3d(10, 30)


# def plot_aruco(ax, p3ds, connections):
#     for _c in connections:
#         ax.plot(
#             xs = [p3ds[_c[0], 0], p3ds[_c[1], 0]],
#             ys = [p3ds[_c[0], 1], p3ds[_c[1], 1]],
#             zs = [p3ds[_c[0], 2], p3ds[_c[1], 2]],
#             c  = 'red')

# con1 = [[0, 1], [1, 2], [2, 3], [3, 0]]

# while True:
#     for cam in cameras:
#         id = cam.id
#         cam.capture_frame()
#         aru = cam.get_arucoImagePose(aruco_dict, parameters)
#         aruco_poses.append(aru)


#     poses_0 = convert_Aruco2CorrespondingPointsList([ar_dict[0] for ar_dict in aruco_poses])
#     poses_0 = np.array(poses_0)
#     pos_0_3D = np.array([stereoSys.triangulate_point(no) for no in poses_0])
#     print(pos_0_3D)

#     # Clear the previous plot
#     ax.cla()
#     ax.set_xlim3d(-15, 5)
#     ax.set_ylim3d(-10, 10)
#     ax.set_zlim3d(10, 30)

#     # Plot the updated data
#     plot_aruco(ax, pos_0_3D, con1)

#     # Update the view
#     ax.view_init(elev=90, azim=90)
#     ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.1, 1]))

#     # Draw the updated plot
#     plt.draw()
#     plt.pause(0.01)  # Pause to allow the plot to update

#     # Break the loop on 'q' key press
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break


# [cam.release() for cam in cameras]
# cv.destroyAllWindows()
# plt.ioff()  # Turn off interactive mode
# # results = pstats.Stats(pr)
# # results.sort_stats(pstats.SortKey.TIME)
# # results.dump_stats("results2.prof")
