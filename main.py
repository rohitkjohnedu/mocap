import cv2 as cv
from cv2 import aruco
import glob
import numpy as np
import matplotlib.pyplot as plt


ROWS = 5
COLUMNS = 8
WORLD_SCALING = 0.6


def calibrate_camera(images_folder, rows, columns, world_scaling ,show_image=False):
    images_names = glob.glob(images_folder)

    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)

            if show_image:
                cv.imshow('img', frame)
                cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)



    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                                        objpoints,
                                        imgpoints,
                                        (width, height),
                                        None,
                                        None
                                    )
    # print('rmse:', ret)
    # print('camera matrix:\n', mtx)
    # print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)

    return mtx, dist




def stereo_calibrate(
                    mtx1, dist1,
                    mtx2, dist2,
                    frames_folder_1, frames_folder_2,
                    rows, columns,
                    world_scaling,
                    show_image = False
                ):
    # read the synched frames
    c1_images_names = glob.glob(frames_folder_1)
    c2_images_names = glob.glob(frames_folder_2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            if show_image:
                cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
                cv.imshow('img', frame1)

                cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
                cv.imshow('img2', frame2)
                cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
                                                    objpoints,
                                                    imgpoints_left,
                                                    imgpoints_right,
                                                    mtx1, dist1,
                                                    mtx2, dist2,
                                                    (width, height),
                                                    criteria=criteria,
                                                    flags=stereocalibration_flags
                                                )

    return R, T


def get_arucoPosition(
                        frames_folder_1, frames_folder_2,
                        mtx1, dist1,
                        mtx2, dist2,
                        aruco_dict, aruco_params
                    ):
    c1_images_names = glob.glob(frames_folder_1)
    c2_images_names = glob.glob(frames_folder_2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        corners_1, ids_1, rejectedImgPoints_1 = aruco.detectMarkers(gray1, aruco_dict, parameters=aruco_params)
        corners_2, ids_2, rejectedImgPoints_2 = aruco.detectMarkers(gray2, aruco_dict, parameters=aruco_params)

    # if ids_1 is not None and ids_2 is not None:
    #     # Estimate pose of each marker
    #     rvecs_1, tvecs_1, _ = cv.aruco.estimatePoseSingleMarkers(corners_1, 0.05, mtx1, dist1)
    #     rvecs_2, tvecs_2, _ = cv.aruco.estimatePoseSingleMarkers(corners_2, 0.05, mtx2, dist2)

    #     for rvec1, tvec1 in zip(rvecs_1, tvecs_1):
    #         # Draw the marker axis
    #         cv.drawFrameAxes(frame1, mtx1, dist1, rvec1, tvec1, 0.1) 

    #     for rvec2, tvec2 in zip(rvecs_2, tvecs_2):
    #         # Draw the marker axis
    #         cv.drawFrameAxes(frame2, mtx2, dist2, rvec2, tvec2, 0.1)

    corner_dict_1 = {}
    corner_dict_2 = {}

    for i in range(len(ids_1)):
        id1 = ids_1[i][0]
        id2 = ids_2[i][0]
        corner_dict_1[id1] = corners_1[i]
        corner_dict_2[id2] = corners_2[i]

    return corner_dict_1, corner_dict_2


def triangulate_aruco(mtx1, mtx2, R, T, uvs1, uvs2):
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    print('uvs1: ', uvs1)
    print('uvs2: ', uvs2)

    # frame1 = cv.imread('testing/_C1.png')
    # frame2 = cv.imread('testing/_C2.png')

    # plt.imshow(frame1[:,:,[2,1,0]])
    # plt.scatter(uvs1[:,0], uvs1[:,1])
    # plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.

    # plt.imshow(frame2[:,:,[2,1,0]])
    # plt.scatter(uvs2[:,0], uvs2[:,1])
    # plt.show()#this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    def DLT(P1, P2, point1, point2):
        
        A = [
                point1[1]*P1[2, :] - P1[1, :],
                P1[0, :] - point1[0]*P1[2, :],
                point2[1]*P2[2, :] - P2[1, :],
                P2[0, :] - point2[0]*P2[2, :]
            ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        # print('Triangulated point: ')
        # print(Vh[3, 0:3]/Vh[3, 3])
        return Vh[3, 0:3]/Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)


    connections = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return p3ds, connections


if __name__ == "__main__":
    # Calibration
    mtx1, dist1 = calibrate_camera('images/*l.png', ROWS, COLUMNS, WORLD_SCALING)
    mtx2, dist2 = calibrate_camera('images/*r.png', ROWS, COLUMNS, WORLD_SCALING)

    print("single camera calibrated")

    R, T = stereo_calibrate(
                            mtx1, dist1,
                            mtx2, dist2,
                            'images/*l.png', 'images/*r.png',
                            ROWS, COLUMNS, WORLD_SCALING
                            )

    # Get Aruco
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    tag_id = 0  # ID of the tag you want to generate

    tag_size = 200  # Size of the tag image
    tag_image = cv.aruco.generateImageMarker(aruco_dict, tag_id, tag_size)
    parameters = cv.aruco.DetectorParameters()

    c1, c2 = get_arucoPosition(
        "aruco/*l.png", "aruco/*r.png",
        mtx1, dist1,
        mtx2, dist2,
        aruco_dict, parameters
    )


    # Triangulate
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
            

    
    p1, con1 = triangulate_aruco(mtx1, mtx2, R, T, c1[0][0], c2[0][0])
    plot_aruco(ax, p1, con1)

    p2, con2 = triangulate_aruco(mtx1, mtx2, R, T, c1[2][0], c2[2][0])
    plot_aruco(ax, p2, con2)

    p3, con3 = triangulate_aruco(mtx1, mtx2, R, T, c1[3][0], c2[3][0])
    plot_aruco(ax, p3, con3)

    plt.show()
