import numpy as np
import numpy.typing as npt
import cv2 as cv
import glob

from cv2 import aruco
from attrs import define, field
from numpy.typing import NDArray
from CameraPPTTypes import Vector3D, Matrix3D, F64, F32, CV_Image, CV_Matrix
from abc import ABC, abstractmethod


@define(slots=True)
class Aruco:
    id: int
    corners: CV_Matrix
    rvecs: NDArray[F32]
    tvecs: NDArray[F32]


@define(slots=True)
class Camera(ABC):
    id: int
    world_scaling: float
    camera_matrix: Matrix3D   = field(default=np.eye(3, dtype=np.float64))
    dist_coeffs: NDArray[F64] = field(default=np.zeros(5, dtype=np.float64))

    position: Vector3D        = field(default=np.zeros(3, dtype=np.float64))
    orientation: Matrix3D     = field(default=np.eye(3, dtype=np.float64))


    calibratedQ: bool         = field(default=False)
    calibration_images: list[CV_Image]  = field(default=[])
    calibration_error: float            = field(default=0.0)
    
    current_frame: CV_Image    = field(default=None)
    chess_rows_number: int     = field(default=None)
    chess_columns_number: int  = field(default=None)

    def calibrate(self, show_image: bool) -> None:
        # criteria used by checkerboard pattern detector.
        # Change this if the code can't find the checkerboard
        criteria: tuple = (
                            cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001
                        )

        rows: int    = self.chess_rows_number
        columns: int = self.chess_columns_number
        images: list[CV_Image]  = self.calibration_images

        # coordinates of squares in the checkerboard world space
        objp: npt.NDArray[F32] = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2]            = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp                   = self.world_scaling * objp

        # frame dimensions. Frames should be the same size.
        width:  int = images[0].shape[1]
        height: int = images[0].shape[0]

        # Pixel coordinates of checkerboards
        imgpoints: list = []  # 2d points in image plane.

        # coordinates of the checkerboard in checkerboard world space.
        objpoints: list = []  # 3d point in real world space

        for frame in images:
            gray: CV_Matrix = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # find the checkerboard
            ret, corners = cv.findChessboardCorners(
                    gray,
                    (rows, columns),
                    None
                )

            if ret:
                # Convolution size used to improve corner detection. 
                # Don't make this too large.
                conv_size: tuple[int, int] = (11, 11)

                # opencv can attempt to improve the checkerboard coordinates
                corners: CV_Matrix = cv.cornerSubPix(
                            gray,
                            corners,
                            conv_size,
                            (-1, -1),
                            criteria)

                if show_image:
                    cv.drawChessboardCorners(
                                frame,
                                (rows, columns),
                                corners,
                                ret)
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

        self.camera_matrix = mtx
        self.dist_coeffs   = dist
        self.calibratedQ   = True

    def get_arucoImagePose(self, aruco_dict, aruco_params) -> list[Aruco]:
        image: CV_Image = self.current_frame
        gray: CV_Image  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints_1 = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=aruco_params)

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            0.05,
            self.camera_matrix,
            self.dist_coeffs)

        tag_list: list[Aruco] = []
        for corner, id, rvec, tvec in zip(corners, ids, rvecs, tvecs):
            tag = Aruco(id, corner, rvec, tvec)
            tag_list.append(tag)

        return tag_list

    @abstractmethod
    def capture_frame(self, image_loc):
        pass

    @abstractmethod
    def capture_calibrationImages(self, image_loc_list, rows, columns):
        pass


@define(slots=True)
class ImageReader(Camera):
    def capture_frame(self, image_loc):
        self.current_frame = cv.imread(image_loc)

    def capture_calibrationImages(self, image_loc_list):
        for im_loc in image_loc_list:
            _im = cv.imread(im_loc)
            self.calibration_images.append(_im)


@define(slots=True)
class StereoCamera:
    camera_list: list[Camera]
    pos_wrt_prime_list: Vector3D = field(default=np.zeros(3, dtype=np.float64))
    rot_wrt_prime: Matrix3D      = field(default=np.eye(3, dtype=np.float64))

    origin: Vector3D      = field(default=np.zeros(3, dtype=np.float64))
    orientation: Matrix3D = field(default=np.eye(3, dtype=np.float64))

    prime_camera: Camera   = field(default=None)
    calibrateStereoQ: bool = False

    def __attrs_post_init__(self) -> None:
        self.prime_camera = self.camera_list[0]

    def calibrate_stereo(self):
        pass

    def change_prime(self, prime_camera_id):
        for cam in self.camera_list:
            if cam.id == prime_camera_id:
                self.prime_camera = cam
                break

    def change_origin(self):
        pass

    def change_orientation(self):
        pass

    def capture_images(self, image_getter):
        image_loc_list = image_getter()
        for cam, loc in zip(self.camera_list, image_loc_list):
            cam.capture_frame(loc)
