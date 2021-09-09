import os
import argparse
from typing import List, Iterable
import yaml
import pathlib

import jsonlines
import cv2
import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
from utils.numpy_json_encoder import NumpyJsonEncoder
from utils.rotated_bbox import RotatingRectangle
from matplotlib import pyplot as plt

from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX


class DetectionInfo:
    KEYS_TO_EXPORT = set(("source_bbox", "source_landmarks", "affine_face_to_aligned",
                          "aligned_landmarks", "face_index", "yaw_degree", "pitch_degree", "roll_degree"))

    def __init__(self,
                 bbox: List[float],
                 landmarks: np.ndarray,
                 yaw_degree: float,
                 pitch_degree: float,
                 roll_degree: float,
                 image_width: int,
                 image_height: int,
                 face_index: int):
        self._source_bbox = np.round(bbox)
        self._source_landmarks = np.round(landmarks[:, :-1])
        self._yaw = yaw_degree
        self._pitch = pitch_degree
        self._roll = roll_degree
        self._image_width = image_width
        self._image_height = image_height
        self._face_index = face_index

        coords = np.array(bbox).reshape(2, 2)
        bottom_left = coords.min(axis=0)
        width_height = np.diff(coords, axis=0).ravel()
        self._center_of_rotation = 0.5 * \
            (bottom_left + bottom_left + width_height)
        self._patch_bbox = RotatingRectangle(
            bottom_left, width_height[0], width_height[1], fill=None, edgecolor="black")
        self._aligned_matrix_face = None

        self._align_face_transform()

        self._aligned_landmarks = cv2.transform(
            self._source_landmarks[None, ...], self.get_face_alignment_matrix())[0]

    def get_rotated_bbox(self) -> RotatingRectangle:
        bbox = self.get_aabb()
        xy = bbox.get_xy()
        width = bbox.get_width()
        height = bbox.get_height()
        return RotatingRectangle(xy, width, height, rel_point_of_rot=tuple(self._center_of_rotation),
                                 angle=-self._roll, fill=None, edgecolor="red")

    def get_aabb(self) -> RotatingRectangle:
        return self._patch_bbox

    def get_face_alignment_matrix(self) -> np.ndarray:
        return self._aligned_matrix_face

    def _align_face_transform(self) -> None:
        aabb = self.get_aabb()
        width = aabb.get_width()
        height = aabb.get_height()
        src = self.get_rotated_bbox().get_verts()[:3].astype(np.float32)
        trg = np.array((
            (0, 0), (width - 1, 0), (width - 1, height - 1)), dtype=np.float32)

        self._aligned_matrix_face = cv2.getAffineTransform(src, trg)

    @property
    def source_bbox(self):
        return self._source_bbox.astype(np.int32)

    @property
    def source_landmarks(self):
        return self._source_landmarks.astype(np.int32)

    @property
    def affine_face_to_aligned(self):
        return self.get_face_alignment_matrix()

    @property
    def aligned_landmarks(self):
        return self.get_face_alignment_matrix()

    @property
    def face_index(self):
        return self._face_index

    @property
    def yaw_degree(self):
        return self._yaw

    @property
    def pitch_degree(self):
        return self._pitch

    @property
    def roll_degree(self):
        return self._roll

    def to_dict(self, keys: List[str]) -> dict:
        info = dict()

        for key in keys:
            if key not in self.KEYS_TO_EXPORT:
                raise ValueError("Found unexpected key to export")

            info[key] = self.__getattribute__(key)

        return info


def process_image(bgr_image: np.ndarray, bbox_model, landmark_model) -> List[DetectionInfo]:
    # Detect faces, get 3DMM params and roi boxes
    boxes = bbox_model(bgr_image)

    n = len(boxes)

    img_height, img_width = bgr_image.shape[:2]

    if n == 0:
        print('No face detected')
        return []

    print(f'Detect {n} faces')

    param_lst, roi_box_lst = landmark_model(bgr_image, boxes)

    # Visualization and serialization
    dense_flag = False
    ver_lst = landmark_model.recon_vers(
        param_lst, roi_box_lst, dense_flag=dense_flag)

    detection_res = []

    for face_index, (param, bbox, ver) in enumerate(zip(param_lst, roi_box_lst, ver_lst)):
        # yaw, pitch, roll
        _, pose = calc_pose(param)
        # ver shape is 3 X N
        detection_res.append(DetectionInfo(
            bbox, ver.T, pose[0], pose[1], pose[2], img_width, img_height, face_index))

    return detection_res


def scan_dir(path_to_dir: str, recursive: bool = False) -> Iterable[os.DirEntry]:
    dir_queue = [path_to_dir]

    while dir_queue:
        current_dir = dir_queue.pop()

        for entry in os.scandir(current_dir):
            if entry.is_dir() and recursive:
                dir_queue.append(entry.path)

            yield entry


def save_debug_image(bgr_image: np.ndarray, detection_res: List[DetectionInfo], debug_path: str) -> None:
    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(1, len(detection_res) + 1)
    ax = axes[0]
    ax.imshow(bgr_image[..., ::-1])

    for i, detection in enumerate(detection_res, 1):
        bbox = detection.get_aabb()
        ax.add_patch(detection.get_rotated_bbox())
        ax.add_patch(bbox)
        ax.scatter(
            detection._source_landmarks[:, 0], detection._source_landmarks[:, 1], alpha=0.5, marker="x")

        matrix = detection.get_face_alignment_matrix()

        new_width = int(bbox.get_width())
        new_height = int(bbox.get_height())

        new_image = cv2.warpAffine(bgr_image, matrix, (new_width, new_height))
        axes[i].imshow(new_image[..., ::-1])
        axes[i].scatter(detection._aligned_landmarks[:, 0],
                        detection._aligned_landmarks[:, 1], alpha=0.5, marker="x")
        axes[i].set_axis_off()

    ax.set_axis_off()
    fig.savefig(debug_path, bbox_inches="tight")


def check_args(args):
    for key in args.save_info:
        if key not in DetectionInfo.KEYS_TO_EXPORT:
            raise ValueError(
                f"Found invalid key for info: 'key'. Accepted keys: {' '.join(DetectionInfo.KEYS_TO_EXPORT)}")


def main(args):
    with open(args.config) as config_file:
        cfg = yaml.load(config_file, Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    debug_dir = out_dir / "debug_aligned"

    if args.debug:
        debug_dir.mkdir(exist_ok=True, parents=True)

    with jsonlines.open(out_dir / args.filename, "w", compact=True, dumps=NumpyJsonEncoder().encode) as json_annotation:
        for entry in scan_dir(args.image_dir, args.recursive):
            filename = entry.name

            if not entry.is_file() or os.path.splitext(filename)[1] not in args.image_ext:
                continue

            bgr_image = cv2.imread(entry.path)
            detected_info = process_image(bgr_image, face_boxes, tddfa)

            if args.debug:
                save_debug_image(bgr_image, detected_info,
                                 str(debug_dir / filename))

            info = dict()
            info["filename"] = filename
            info["relative_path"] = pathlib.Path(
                entry.path).relative_to(args.image_dir).as_posix()
            info["faces"] = []

            for detection in detected_info:
                info["faces"].append(detection.to_dict(args.save_info))

            json_annotation.write(info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str,
                        default='configs/mb1_120x120.yml')
    parser.add_argument("--image_dir", type=str, required=True,
                        help="A path to directory with images")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Scan directory recursively")
    parser.add_argument("--image_ext", nargs="+",
                        default=[".jpg"], help="A image extension to process")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="A path to output directory")
    parser.add_argument('-m', '--mode', choices=["cpu", "gpu"],
                        default='cpu', help='gpu or cpu mode')
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument("--debug", action="store_true",
                        help="Save debug photo")
    parser.add_argument("--filename", type=str,
                        default="annotation.json")
    parser.add_argument("--save_info", nargs="*", default=DetectionInfo.KEYS_TO_EXPORT,
                        help=f"Information to save: {' '.join(DetectionInfo.KEYS_TO_EXPORT)}")

    args = parser.parse_args()

    check_args(args)
    main(args)
