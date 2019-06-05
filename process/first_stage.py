from __future__ import division
from typing import Tuple

import numpy as np

import component
import process
from config import Box, Point
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager


def get_mid_num(
    label_nii: LabelNiiFileManager, rate: float=0.06,
    midbrain_label: int=25
) -> int:
    size = label_nii.size[2]
    min_volum = 500
    for num, label in enumerate(label_nii.get_slice(
        start=int((1-rate)*size/2),
        end=int((1+rate)*size/2),
        dim=2
    )):
        volum = (label == midbrain_label).sum()
        if volum < min_volum:
            min_volum = volum
            mid_num = int((1-rate)*size/2) + num
    return mid_num

def get_corpus(img, clahe_limit=0.02, bin_rate=1.6, center_rate=1/4,
               min_area=200, max_dis=45, clahe_row=8, clahe_col=8):
    _, bin_img = process.get_otsu(img, False)
    center = process.get_grav_center(bin_img)
    clahe_img = process.get_clahe_image(img, clahe_limit, clahe_row, clahe_col)
    clahe_otsu = process.get_otsu(clahe_img)
    clahe_bin_img = process.get_bin_image(clahe_img, clahe_otsu * bin_rate)
    components, label = component.get_connected_component(clahe_bin_img, min_area)
    components = [
        component for component in components
        if component.in_range(
            left=img.shape[1] * center_rate,
            up=img.shape[0] * center_rate,
            right=img.shape[1] * (1 - center_rate),
            down=img.shape[0] * (1 - center_rate)
        )
    ]
    components = process.get_near_component(components, center, max_dis)
    corpus = components[0]
    return corpus

def get_corpus_angle(corpus: component.ConnectedComponent):
    corpus_left, corpus_right = corpus.get_bound_point('l'), corpus.get_bound_point('r')
    corpus_angle = np.arctan((corpus_right[0] - corpus_left[0]) / (corpus_right[1] - corpus_left[1]))
    return corpus_angle

def get_rotated_quad_seg_point(quad_seg_point: Point, shape: Point, angle: float):
    center_y, center_x = tuple(map(lambda x: int(x / 2), shape))
    quad_y, quad_x = quad_seg_point
    # quad_angle = np.arctan((quad_y - center_y) / (quad_x - center_x))
    rotated_quad_x = int((quad_x-center_x)*np.cos(angle) + (quad_y-center_y)*np.sin(angle) + center_x)
    rotated_quad_y = int((quad_y-center_y)*np.cos(angle) + (quad_x-center_x)*np.sin(angle) + center_y)
    return (rotated_quad_y, rotated_quad_x)

def get_quad_seg_point(
    image: np.ndarray, label: np.ndarray,
    midbrain_label: int=25, rate: float=1.21,
    box: Box=((-15, 6), (-8, 8)), debug: bool=False
) -> Point:
    assert rate > 1
    right_bound = process.get_bound_point(label==midbrain_label, 'r')
    box_index = (
        slice(right_bound[0]+box[0][0], right_bound[0]+box[0][1]),
        slice(right_bound[1]+box[1][0], right_bound[1]+box[1][1])
    )
    otsu = process.get_otsu(image[box_index])
    bin_image = image > otsu * rate
    components, _ = component.get_connected_component(bin_image[box_index])
    components = sorted(
        [component for component in components if not component.img[box[0][1] - box[0][0] - 1].any()],
        key=lambda component: process.get_area_distance(
            component.img,
            (-box[0][0], -box[1][0])
        )
    )
    if debug:
        process.show([bin_image] + components)
    if not components:
        return get_quad_seg_point(image, label, midbrain_label, rate+0.02, box, debug)
    if components[0].img.sum() < 8:
        return get_quad_seg_point(image, label, midbrain_label, rate-0.02, box, debug)
    boxed_quad_seg_point = components[0].get_bound_point('d')
    quad_seg_point = (
        boxed_quad_seg_point[0]+right_bound[0]+box[0][0],
        boxed_quad_seg_point[1]+right_bound[1]+box[1][0]
    )
    return quad_seg_point

# def get_brainstem_seg_point(label, quad_seg_point, midbrain_label=25, pons_label=26):
#     box = ((-20, 20), (-50, -10))
#     for y in range(quad_seg_point[0]+box[0][0], quad_seg_point[0]+box[0][1]):
#         for x in range(quad_seg_point[1]+box[1][0], quad_seg_point[1]+box[1][1]):
#             block = label[y-1:y+2, x-1:x+2]
#             if not label[y, x] and midbrain_label in block and pons_label in block:
#                 brainstem_seg_point = (y, x)
#     return brainstem_seg_point

def get_pons_area(label: np.ndarray, pons_label: int=26) -> int:
    return (label==pons_label).sum()

def get_midbrain_area(label: np.ndarray, midbrain_label: int=25) -> int:
    return (label==midbrain_label).sum()

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager,
    mid_num_rate: float=0.06, quad_seg_rate: float=1.21,
    midbrain_label: int=25, pons_label: int=26,
    box: Box=((-15, 6), (-8, 8)), debug: bool=False
) -> Tuple[Point, int, int, int]:
    mid_num = get_mid_num(
        label_nii, rate=mid_num_rate, midbrain_label=midbrain_label,
    )
    mid_image = image_nii.get_slice(mid_num, dim=2)
    mid_label = label_nii.get_slice(mid_num, dim=2)
    quad_seg_point = get_quad_seg_point(
        mid_image, mid_label, rate=quad_seg_rate,
        midbrain_label=midbrain_label, box=box, debug=debug
    )
    midbrain_area = get_midbrain_area(mid_label, midbrain_label=midbrain_label)
    pons_area = get_pons_area(mid_label, pons_label=pons_label)
    corpus = get_corpus(mid_image)
    corpus_angle = get_corpus_angle(corpus)
    rotated_quad_seg_point = get_rotated_quad_seg_point(quad_seg_point, mid_image.shape, corpus_angle)
    return quad_seg_point, mid_num, pons_area, midbrain_area, corpus_angle, rotated_quad_seg_point

def show(
    image_nii: RotatedNiiFileManager, quad_seg_point: Point,
    mid_num: int, point_color: int=255
) -> None:
    process.show(
        process.add_points(
            image_nii.get_slice(mid_num, dim=2),
            [quad_seg_point], 1, point_color
        )
    )
