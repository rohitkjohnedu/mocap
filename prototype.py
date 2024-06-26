from Camera import ImageReader, StereoCamera, Aruco, convert_Aruco2CorrespondingPointsList
import cv2 as cv
import numpy as np
from cv2 import aruco
import glob
from CameraPPTTypes import F32

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cProfile
import pstats

ROWS    = 5
COLUMNS = 8
WORLD_SCALING: F32 = np.float32(0.4)

cameras: list[ImageReader] = []
camera_indeces      = [0, 1, 3, 4, 5]

with cProfile.Profile() as pr:
    for i in camera_indeces:
        calibration_images = glob.glob(f'images/*{i}.png')
        cam = ImageReader(i, WORLD_SCALING, ROWS, COLUMNS)
        cam.capture_calibrationImages(calibration_images)
        cam.calibrate()
        cameras.append(cam)
        print("id ", cam.id)


    stereoSys = StereoCamera(cameras)
    stereoSys.calibrate_stereo()

    # get aruco positions
    aruco_dict: cv.aruco.Dictionary         = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()

    aruco_poses: list[dict[int, Aruco]] = []

    for cam in cameras:
        id = cam.id
        cam.capture_frame(f'aruco/ar_{id}.png')
        aru = cam.get_arucoImagePose(aruco_dict, parameters)
        aruco_poses.append(aru)

    poses_2 = convert_Aruco2CorrespondingPointsList([ar_dict[2] for ar_dict in aruco_poses])
    poses_3 = convert_Aruco2CorrespondingPointsList([ar_dict[3] for ar_dict in aruco_poses])
    poses_0 = convert_Aruco2CorrespondingPointsList([ar_dict[0] for ar_dict in aruco_poses])

    poses_2 = np.array(poses_2)
    poses_3 = np.array(poses_3)
    poses_0 = np.array(poses_0)

    pos_2_3D = np.array([stereoSys.triangulate_point(no) for no in poses_2])
    pos_3_3D = np.array([stereoSys.triangulate_point(no) for no in poses_3])
    pos_0_3D = np.array([stereoSys.triangulate_point(no) for no in poses_0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)

    def plot_aruco(ax, p3ds, connections):
        for _c in connections:
            ax.plot(
                xs = [p3ds[_c[0], 0], p3ds[_c[1], 0]],
                ys = [p3ds[_c[0], 1], p3ds[_c[1], 1]],
                zs = [p3ds[_c[0], 2], p3ds[_c[1], 2]],
                c  = 'red')
            

    con1 = [[0,1],[1,2],[2,3],[3,0]]
    plot_aruco(ax, pos_2_3D, con1)
    plot_aruco(ax, pos_3_3D, con1)
    plot_aruco(ax, pos_0_3D, con1)
    # Make matplotlib orthographic
    ax.view_init(elev=90, azim=90)
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.1, 1]))
    plt.show()


results = pstats.Stats(pr)
results.sort_stats(pstats.SortKey.TIME)
results.dump_stats("results2.prof")
