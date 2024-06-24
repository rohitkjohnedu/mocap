import cv2 as cv
from cv2 import aruco
import glob
import numpy as np
import matplotlib.pyplot as plt


ROWS = 5
COLUMNS = 8
WORLD_SCALING = 0.6


# ------------------------------------------------------------------------------------------------ #
#                                                                                 CALIBRATE_CAMERA #
# ------------------------------------------------------------------------------------------------ #
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

    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
                                        objpoints,
                                        imgpoints,
                                        (width, height),
                                        None,
                                        None
                                    )

    return mtx, dist


# ------------------------------------------------------------------------------------------------ #
#                                                                                 STEREO_CALIBRATE #
# ------------------------------------------------------------------------------------------------ #
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


# ------------------------------------------------------------------------------------------------ #
#                                                                                GET_ARUCOPOSITION #
# ------------------------------------------------------------------------------------------------ #
def get_arucoPosition(
                        frames_folder,
                        mtx_s, dist_s,
                        aruco_dict, aruco_params
                    ):
    
    no_cameras = len(mtx_s)

    # read the synched frames
    image_names = glob.glob(frames_folder)

    print(image_names)

    images = []
    for im_name in image_names:
        _im = cv.imread(im_name, 1)
        images.append(_im)

    # array to store the corners and ids of the aruco markers from list of 
    # camera images
    corners_list = []
    ids_list     = []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints_1 = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        corners_list.append(corners)
        ids_list.append(ids)

    # stores the 
    corners_list_dict = []
    print(len(ids_list))

    for j in range(no_cameras):
        corner_dict = {}
        for i in range(len(ids_list[j])):
            id = ids_list[j][i][0]
            corner_dict[id] = corners_list[j][i]
        corners_list_dict.append(corner_dict)

    return corners_list_dict


# ------------------------------------------------------------------------------------------------ #
#                                                                                TRIANGULATE_ARUCO #
# ------------------------------------------------------------------------------------------------ #
def triangulate_aruco(camera_matrix_list, Rs, Ts, image_points_list):
    image_points_list = [np.array(pts) for pts in image_points_list]
    no_points         = len(image_points_list[0]) 

    Ps = []

    for i in range(len(camera_matrix_list)):
        RT = np.concatenate([Rs[i], Ts[i]], axis=-1)
        P = camera_matrix_list[i] @ RT
        Ps.append(P)


    def DLT(Ps, image_points):   
        A = []
        for P, image_point in zip(Ps, image_points):
            A.append(image_point[1]*P[2, :] - P[1, :])
            A.append(P[0, :] - image_point[0]*P[2, :])

        A = np.array(A).reshape((len(Ps)*2, 4))
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        return Vh[3, 0:3]/Vh[3, 3]

    p3ds = []
    for i in range(no_points):
        image_points = [pts[i] for pts in image_points_list]
        _p3d = DLT(Ps, image_points)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)


    connections = [[0, 1], [1, 2], [2, 3], [3, 0]]
    return p3ds, connections


if __name__ == "__main__":
    # Calibration
    mtx1, dist1 = calibrate_camera('images/*0.png', ROWS, COLUMNS, WORLD_SCALING)
    mtx2, dist2 = calibrate_camera('images/*1.png', ROWS, COLUMNS, WORLD_SCALING)
    mtx3, dist3 = calibrate_camera('images/*2.png', ROWS, COLUMNS, WORLD_SCALING)


    camera_matrix_list  = [mtx1, mtx2, mtx3]
    camera_distor_list  = [dist1, dist2, dist3]

    print("single camera calibrated")

    Rs = [np.eye(3)]
    Ts = [np.zeros((3, 1))]

    prime_camera_matrix = camera_matrix_list[0]
    prime_camera_distor = camera_distor_list[0]

    for i, (camera_matrix, camera_distor) in enumerate(zip(camera_matrix_list[1:], camera_distor_list[1:])):
        R, T = stereo_calibrate(
                            prime_camera_matrix, prime_camera_distor,
                            camera_matrix, camera_distor,
                            'images/*0.png', f'images/*{i}.png',
                            ROWS, COLUMNS, WORLD_SCALING
                            )
        Rs.append(R)
        Ts.append(T)
 
    print(T)

    # Get Aruco
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    tag_id = 0  # ID of the tag you want to generate

    tag_size = 200  # Size of the tag image
    tag_image = cv.aruco.generateImageMarker(aruco_dict, tag_id, tag_size)
    parameters = cv.aruco.DetectorParameters()

    corners_list_dict = get_arucoPosition(
        "aruco/*.png",
        camera_matrix_list,
        camera_distor_list,
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
            

    
    p1, con1 = triangulate_aruco(camera_matrix_list, Rs, Ts, [i[0][0] for i in corners_list_dict])
    plot_aruco(ax, p1, con1)

    p2, con2 = triangulate_aruco(camera_matrix_list, Rs, Ts, [i[2][0] for i in corners_list_dict])
    plot_aruco(ax, p2, con2)

    p3, con3 = triangulate_aruco(camera_matrix_list, Rs, Ts, [i[3][0] for i in corners_list_dict])
    plot_aruco(ax, p3, con3)

    plt.show()
