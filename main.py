import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt


ROWS = 5
COLUMNS = 8
WORLD_SCALING = 1.0


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




if __name__ == "__main__":
    mtx1, dist1 = calibrate_camera('images/*l.png', ROWS, COLUMNS, WORLD_SCALING)
    mtx2, dist2 = calibrate_camera('images/*r.png', ROWS, COLUMNS, WORLD_SCALING)

    print("single camera calibrated")

    R, T = stereo_calibrate(
                            mtx1, dist1,
                            mtx2, dist2,
                            'images/*l.png', 'images/*r.png',
                            ROWS, COLUMNS, WORLD_SCALING
                            )
    print("Stereo calibration done")
    print("rotatuin matrix:\n", R)
    print("translation matrix:\n", T)