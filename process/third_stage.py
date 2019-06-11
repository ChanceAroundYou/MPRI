from typing import List, Sequence, Tuple

import cv2
import numpy as np

import component
import process
from config import Box, Point, Scp_components, Scp_show_info_item
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager
from scipy.ndimage import rotate as scipy_rotate


def get_scp_slice(image_nii: LabelNiiFileManager, quad_seg_point: Point, mid: int, size: int=10, otsu_rate: float=1.25):
    start = quad_seg_point[1] - 1
    center_y = quad_seg_point[0] * size
    center_x = mid * size
    resize_shape = (image_nii.size[0] * size, image_nii.size[2] * size)
    box = (
        slice(center_y-20, center_y+150),
        slice(center_x-130, center_x+130)
    )
    for i in range(5):
        img = image_nii.get_slice(start + i, dim=1)
        resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)
        part_img = resized[box]
        otsu = process.get_otsu(part_img)
        mask = process.get_bin_image(part_img, otsu * otsu_rate)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        low_components = [(num, stat) for num, stat in enumerate(stats) if stat[1]+stat[3] > 100 and num > 0]
        low_components = sorted(low_components, key=lambda x: x[1][1])
#         print(low_components)
        if low_components[0][1][1] > 20:
            return resized, center_y, center_x

def get_scp_width(scp: np.ndarray, size: int=10, add: float=0.3) -> float:
    scp_angle = process.first_stage.get_img_angle(scp)
    rotated_scp = scipy_rotate(scp, np.rad2deg(scp_angle), reshape=True)
    return rotated_scp.sum(axis=1).max() / size + add

def clear_scp(scp):
    max_low = 0

    if np.where(scp)[1].min() < 120:
        x_range = range(0, int(0.7*np.where(scp)[1].max()))
    else:
        x_range = range(scp.shape[1]-1, int(1.1*np.where(scp)[1].min()), -1)
    for x in x_range:
        col = scp[:, x]
        if col.any():
            low = np.where(col == 0)[0].max()
            if low > max_low and low < 129:
                max_low = low
                max_x = x
    if np.where(scp)[1].min() > 120:
        scp[:, max_x:] = 0
    else:
        scp[:, :max_x] = 0
    return scp

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager,
    quad_seg_point: Point, mid_num: int, angle: float, num: int=2, size: int=10, rate: float=0.92,
    slice_rate: float=1.25, part_rate: float=1.26, add: float=0.3, show: bool=False, debug: bool=False
) -> Tuple[float, List[Scp_show_info_item]]:
    image_nii.rotate(angle)
    label_nii.rotate(angle)
    scp_slice, center_y, center_x = get_scp_slice(image_nii, quad_seg_point, mid_num, size, slice_rate)
    part_img = scp_slice[center_y-10:center_y+120, center_x-120:center_x+120]
    otsu = process.get_otsu(part_img)
    mask = process.get_bin_image(part_img, otsu * part_rate)
    _, label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    buttom_components = [(num, stat)
        for num, stat in enumerate(stats)
        if stat[1]+stat[3] > 100 and num > 0
        ]
    scp_components = sorted(buttom_components, key=lambda x: x[1][4], reverse=True)
    scp_width = np.mean([
        get_scp_width(clear_scp((label==scp[0]).astype(np.uint8)), size, add)
        for scp in scp_components[:2]
        ])
    return scp_width, None


def show(image_nii: RotatedNiiFileManager, show_info: List[Scp_show_info_item], mask_color: int=25) -> None:
    process.show([
        process.add_mask(
            process.add_mask(
                image_nii.get_slice(index, dim=1), components[0].img_bool, mask_color
            ), components[1].img_bool, mask_color
        ) for index, components in show_info
    ])
