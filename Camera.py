import numpy as np
import numpy.typing as npt
import cv2 as cv

from cv2 import aruco
from attrs import define, field
from numpy.typing import NDArray
from CameraPPTTypes import NP_Vector_3D, NP_Matrix_3D, F64, F32, CV_Image, CV_Matrix, NP_Matrix_NxM, CV_Vector_3D 
from abc import ABC, abstractmethod
from multiprocessing import Queue


def DLT(Ps: list[NP_Matrix_NxM], image_points: NP_Matrix_NxM) -> NP_Matrix_NxM:
    A = []
    for P, image_point in zip(Ps, image_points):
        A.append(image_point[1] * P[2, :] - P[1, :])
        A.append(P[0, :] - image_point[0] * P[2, :])

    A = np.array(A).reshape((len(Ps) * 2, 4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


@define(slots=True)
class Aruco:
    id: int
    corners_image_position: CV_Matrix
    rvecs: NDArray[F32]
    tvecs: NDArray[F32]


def convert_Aruco2CorrespondingPointsList(aruco_list: list[Aruco]) -> list[list[CV_Matrix]]:
    NO_ARUCO_CORNERS = 4
    corner_point_list: list[list[CV_Matrix]] = []

    for i in range(NO_ARUCO_CORNERS):
        corner_point: list[CV_Matrix] = [aruco_list.corners_image_position[i] for aruco_list in aruco_list]
        corner_point_list.append(corner_point)

    return corner_point_list



@define(slots=True)
class Camera(ABC):
    id: int
    world_scaling: F32
    chess_rows_number: int
    chess_columns_number: int

    camera_matrix: NP_Matrix_3D   = field(factory=lambda: np.eye(3, dtype=np.float64))
    dist_coeffs: NDArray[F64]     = field(factory=lambda: np.zeros(5, dtype=np.float64))

    position: CV_Vector_3D     = field(factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: CV_Matrix     = field(factory=lambda: np.eye(3, dtype=np.float64))

    calibratedQ: bool         = field(default=False)
    calibration_images: list[CV_Image]  = field(factory=list)
    calibration_error: float            = field(default=0.0)

    current_frame: CV_Image    = field(default=None)


    def calibrate(self, show_image: bool = False) -> None:
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
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # find the checkerboard
            ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

            if ret:
                conv_size: tuple[int, int] = (11, 11)

                # opencv can attempt to improve the checkerboard coordinates
                corners: CV_Matrix = cv.cornerSubPix(
                    gray,
                    corners,
                    conv_size,
                    (-1, -1),
                    criteria)

                if show_image:
                    frame_copy: CV_Image = frame.copy()
                    cv.drawChessboardCorners(
                        frame_copy,
                        (rows, columns),
                        corners,
                        ret)
                    cv.imshow('img', frame_copy)
                    cv.waitKey(500)

                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print("Checkerboard not found in image")

        # calibrate the camera
        # print("len objpoints", len(objpoints))
        # print("len imgpoints", len(imgpoints))
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

    def get_arucoImagePose(self,
                           aruco_dict: cv.aruco.Dictionary,
                           aruco_params: cv.aruco.DetectorParameters
                           ) -> dict[int, Aruco]:
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

        tag_list: dict[int, Aruco] = {}
        for corner, id, rvec, tvec in zip(corners, ids, rvecs, tvecs):
            if len(id) != 1:
                raise ValueError("Multiple ids for one tag")
            tag = Aruco(id[0], corner[0], rvec, tvec)
            tag_list[id[0]] = tag

        return tag_list

    @abstractmethod
    def capture_frame(self, image_loc: str):
        pass

    @abstractmethod
    def capture_calibrationImages(self, image_loc_list, rows, columns):
        pass


@define(slots=True)
class ImageReader(Camera):
    def capture_frame(self, image_loc: str):
        self.current_frame = cv.imread(image_loc)

    def capture_calibrationImages(self, image_loc_list):
        for im_loc in image_loc_list:
            _im = cv.imread(im_loc)
            self.calibration_images.append(_im)


@define(slots=True)
class WebCam(Camera):
    command_queue: Queue = field(factory=lambda: None)
    cap: cv.VideoCapture = field(factory=lambda: None)

    def __init__(self, id, world_scaling, rows, columns, command_queue, **kwargs):
        super().__init__(id, world_scaling, rows, columns, **kwargs)
        self.command_queue = command_queue

    def initialize_camera(self):
        self.cap = cv.VideoCapture(self.id)
        if not self.cap.isOpened():
            print(f"Camera {self.id}: Failed to open.")
            return False
        return True

    def capture_frame(self, image_loc: str = "") -> bool:
        ret, frame         = self.cap.read()
        if not ret:
            raise ValueError(f"Camera ID {self.id}: Frame cannot captured")
        self.current_frame = frame
        return ret

    def show_frame(self):
        cv.imshow(f"Camera {self.id}", self.current_frame)
        cv.waitKey(1)

    def capture_calibrationImages(self):
        if not self.initialize_camera():
            return
        print(f"""\nCamera {self.id} is ready to capture.""")

        while True:
            if not self.capture_frame():
                break

            # Show the current frame
            cv.imshow(f'Camera Feed {self.id}', self.current_frame)
            cv.waitKey(1)

            if not self.command_queue.empty():
                command = self.command_queue.get()
                if command == 'capture':
                    self.calibration_images.append(self.current_frame.copy())
                    print(f"Camera {self.id}: Image captured.")
                elif command == 'quit':
                    break

        # Release the camera and close the window
        self.cap.release()
        cv.destroyAllWindows()

    def release_camera(self):
        self.cap.release()


@define(slots=True)
class StereoCamera:
    camera_list: list[Camera]
    pos_wrt_prime_list: NP_Vector_3D = field(factory=lambda: np.zeros(3, dtype=np.float64))
    rot_wrt_prime: NP_Matrix_3D      = field(factory=lambda: np.eye(3, dtype=np.float64))

    origin: NP_Vector_3D      = field(factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: NP_Matrix_3D = field(factory=lambda: np.eye(3, dtype=np.float64))

    prime_camera: Camera   = field(default=None)
    calibrateStereoQ: bool = False

    camera_poses: list[CV_Vector_3D]  = field(factory=list)
    camera_oris:  list[CV_Matrix]     = field(factory=list)

    @property
    def get_numberCameras(self) -> int:
        nos = len(self.camera_list)
        return nos

    def __attrs_post_init__(self) -> None:
        self.prime_camera = self.camera_list[0]

    def calibrate_stereo(self, show_image=False):
        prime_cam  = self.prime_camera
        other_cams = [cam for cam in self.camera_list if cam != prime_cam]

        prime_pos = np.array([[0], [0], [0]])
        prime_cam.position  = prime_pos
        self.camera_poses   = []
        self.camera_poses.append(prime_pos)

        prime_ori = np.eye(3)
        prime_cam.orientation = prime_ori
        self.camera_oris = []
        self.camera_oris.append(prime_ori)

        for cam in other_cams:
            R, T = self.calibrate_pairStereo(prime_cam, cam, show_image)
            cam.position    = T
            cam.orientation = R

            self.camera_poses.append(T)
            self.camera_oris.append(R)

    def calibrate_pairStereo(self, prime_cam, other_cam, show_image=False):
        prime_cam_images: CV_Image = prime_cam.calibration_images
        other_cam_images: CV_Image = other_cam.calibration_images

        # change this if stereo calibration not good.
        criteria: tuple = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # Check if the cameras use the same chessboard dimensions
        rows_sameQ: bool = prime_cam.chess_rows_number == other_cam.chess_rows_number
        cols_sameQ: bool = prime_cam.chess_columns_number == other_cam.chess_columns_number
        if not rows_sameQ or not cols_sameQ:
            print(f"prime_cam: {prime_cam.chess_rows_number} x {prime_cam.chess_columns_number}")
            print(f"other_cam: {other_cam.chess_rows_number} x {other_cam.chess_columns_number}")
            raise ValueError("Chessboard dimensions must be the same for both cameras")

        world_scaling_sameQ: float = prime_cam.world_scaling == other_cam.world_scaling
        if not world_scaling_sameQ:
            print(f"prime_cam: {prime_cam.world_scaling}")
            print(f"other_cam: {other_cam.world_scaling}")
            raise ValueError("World scaling of cameras not same")


        # coordinates of squares in the checkerboard world space
        rows: int    = prime_cam.chess_rows_number
        columns: int = prime_cam.chess_columns_number
        world_scaling: float = prime_cam.world_scaling

        objp: NP_Matrix_NxM = np.zeros((rows * columns, 3), np.float32)
        objp[:, :2]     = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
        objp            = world_scaling * objp

        # frame dimensions. Frames should be the same size.
        width: int  = prime_cam_images[0].shape[1]
        height: int = prime_cam_images[0].shape[0]

        # Pixel coordinates of checkerboards
        imgpoints_left: list  = []  # 2d points in image plane.
        imgpoints_right: list = []

        # coordinates of the checkerboard in checkerboard world space.
        objpoints: list = []  # 3d point in real world space

        for frame1, frame2 in zip(prime_cam_images, other_cam_images):
            gray1: CV_Image = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            gray2: CV_Image = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
            c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

            if c_ret1 and c_ret2:
                corners1: CV_Matrix = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                corners2: CV_Matrix = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

                if show_image:
                    frame1_copy: CV_Image = frame1.copy()
                    cv.drawChessboardCorners(frame1_copy, (rows, columns), corners1, c_ret1)
                    cv.imshow('img', frame1_copy)

                    frame2_copy: CV_Image = frame2.copy()
                    cv.drawChessboardCorners(frame2_copy, (rows, columns), corners2, c_ret2)
                    cv.imshow('img2', frame2_copy)
                    cv.waitKey(500)

                objpoints.append(objp)
                imgpoints_left.append(corners1)
                imgpoints_right.append(corners2)
            else:
                print(f"checkerboard not found in image, {c_ret1}, {c_ret2}")

        mtx1: CV_Matrix  = prime_cam.camera_matrix
        dist1: CV_Matrix = prime_cam.dist_coeffs
        mtx2: CV_Matrix  = other_cam.camera_matrix
        dist2: CV_Matrix = other_cam.dist_coeffs

        stereocalibration_flags: int = cv.CALIB_FIX_INTRINSIC
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


    def change_prime(self, new_prime_camera_id):
        for cam in self.camera_list:
            if cam.id == new_prime_camera_id:
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

    def triangulate_point(self, image_point_list: NP_Matrix_NxM) -> NP_Vector_3D:
        """
        Triangulate a point from multiple camera views using the Direct Linear Transform
        (DLT) method.

        Args:
            image_point_list (NP_Matrix_NxM): List of coordinates of a single point from
                                              the image spaces of multiple cameras.

        Returns:
            NP_Vector_3D: Coordinates in the world space of the stereo system. By
                          default, the origin is the prime camera.
        """
        Ps = []
        camera_matrix_list: list[NP_Matrix_3D] = [cam.camera_matrix for cam in self.camera_list]

        Ts: list[CV_Matrix] = self.camera_poses
        Rs: list[CV_Matrix] = self.camera_oris

        for i in range(self.get_numberCameras):
            RT = np.concatenate([Rs[i], Ts[i]], axis=-1)
            P = camera_matrix_list[i] @ RT
            Ps.append(P)

        return DLT(Ps, image_point_list)
