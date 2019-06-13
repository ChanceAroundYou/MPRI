from typing import List, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import component
import process
from config import Point, Scp_components, Scp_show_info_item
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
    for i in [1, 0, 2, 3]:
        img = image_nii.get_slice(start + i, dim=1)
        resized = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)
        part_img = resized[box]
        otsu = process.get_otsu(part_img)
        mask = process.get_bin_image(part_img, otsu * otsu_rate)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        low_components = [
            (num, stat) for num, stat in enumerate(stats)
            if stat[1]+stat[3] > 120 and num > 0 and (
                110 < stat[0] < 180 or 80 < stat[0] + stat[2] < 150
            )
        ]
        low_components = sorted(low_components, key=lambda x: x[1][1])
        # print(low_components)
        if len(low_components) >= 2 and low_components[0][1][1] > 20:
            return resized, center_y, center_x

def get_scp_width(scp: np.ndarray, size: int=10, add: float=0.3) -> float:
    scp_angle = process.first_stage.get_img_angle(scp)
    rotated_scp = scipy_rotate(scp, np.rad2deg(scp_angle), reshape=True)
    return rotated_scp.sum(axis=1).max() / size + add

def get_clear_scp(scp):
    left = np.where(scp)[1].min()
    right = np.where(scp)[1].max()
    # buttom = scp.shape[0]

    # max_low = 0
    # if left < 120:
    #     x_range = range(left, right)
    # else:
    #     x_range = range(right-1, left, -1)

    # for x in x_range:
    #     col = scp[:, x]
    #     if col.any():
    #         low = np.where(col == 0)[0].max()
    #         if low > max_low and low < buttom - 1:
    #             max_low = low
    #             max_x = x

    # if left > 120 and right - max_x > 5:
    #     scp[:, max_x-5:] = 0
    # elif left < 120 and max_x - left > 5:
    #     scp[:, :max_x+5] = 0
    # return scp

    if left < 120:
        low = np.array([np.where(scp[:, x])[0].min() for x in range(left, right)])
    else:
        low = np.array([np.where(scp[:, x])[0].min() for x in range(right-1, left, -1)])

    max_x = None
    diff = np.diff(low, 1)
    # print(low, diff)
    for x in range(5, len(low), 3):
        dleft = diff[max(x-5, 0):x].sum()
        dright = diff[x:x+5].sum()
        if dleft >= 0 and dright < 0:
            # print(x, diff[max(x-10, 0):x], diff[x:x+10])
            if diff[max(x-10, 0):x].sum() > 8 or diff[x:x+10].sum() < -8:
                max_x = x

    if left < 120 and max_x is not None and max_x > 5:
        scp[:, :left + max_x + 8] = 0
    elif left > 120 and max_x is not None and right - left - max_x > 5:
        scp[:, right - max_x - 8:] = 0
    return scp

def get_rotated_cut_points(rotated_scp):
    line_y = rotated_scp.sum(axis=1).argmax()
    line = np.where(rotated_scp[line_y])[0]
    assert line.size
    left_point = (line_y, line.min())
    right_point = (line_y, line.max())
    return left_point, right_point

def get_cut_points(scp, rotated_scp, angle):
    def _get_center(shape):
        return tuple(map(lambda x: int(x / 2), shape))

    scp_center = _get_center(scp.shape)
    rotated_scp_center = _get_center(rotated_scp.shape)

    rotated_left_point, rotated_right_point = get_rotated_cut_points(rotated_scp)
    left_point = process.get_rotated_point(rotated_left_point, -angle, rotated_scp_center, scp_center)
    right_point = process.get_rotated_point(rotated_right_point, -angle, rotated_scp_center, scp_center)
    return left_point, right_point

def run(
        image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager,
        quad_seg_point: Point, mid_num: int, angle: float, size: int=10,
        slice_rate: float=1.25, part_rate: float=1.26, add: float=0.3,
        box=[-10, 150, -130, 130], show_: bool=False
) -> Tuple[float, List[Scp_show_info_item]]:
    image_nii.rotate(angle)
    label_nii.rotate(angle)
    scp_slice, center_y, center_x = get_scp_slice(
        image_nii, quad_seg_point, mid_num, size, slice_rate
    )
    part_img = scp_slice[center_y+box[0]:center_y+box[1], center_x+box[2]:center_x+box[3]]
    otsu = process.get_otsu(part_img)
    mask = process.get_bin_image(part_img, otsu * part_rate)
    _, label, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    buttom_components = [
        (num, stat) for num, stat in enumerate(stats)
        if stat[1]+stat[3] > 130 and num > 0 and (
            110 < stat[0] < 180 or 80 < stat[0] + stat[2] < 150
        )
    ]
    scp_components = sorted(buttom_components, key=lambda x: x[1][4], reverse=True)[:2]
    assert len(scp_components) == 2

    scp_widths = []
    scp_cut_points = []

    for num, _ in scp_components:
        scp = get_clear_scp((label == num).astype(np.uint8))
        scp_angle = process.first_stage.get_img_angle(scp)
        scp[140:] = 0
        rotated_scp = scipy_rotate(scp, np.rad2deg(scp_angle), reshape=True)
        plt.imshow(rotated_scp)
        # print(rotated_scp.sum(axis=1).max())
        scp_widths.append(rotated_scp.sum(axis=1).max() / size + add)

        if show_:
            scp_cut_points.append(get_cut_points(scp, rotated_scp, scp_angle))

    scp_width = np.mean(scp_widths)
    # print(scp_widths)
    if show_:
        scp_image = scp_slice.copy()
        move = (center_y+box[0], center_x+box[2])
        for left_point, right_point in scp_cut_points:
            moved_left_point = tuple(map(sum, zip(left_point, move)))
            moved_right_point = tuple(map(sum, zip(right_point, move)))
            scp_image = process.add_line(
                scp_image, moved_left_point, moved_right_point,
                color=255, thickness=4
            )

        mid_image = image_nii.get_slice(mid_num, dim=2)
        mid_image = cv2.resize(
            mid_image,
            tuple(map(
                lambda x: x * size,
                reversed(mid_image.shape)))
        )
        mid_image = process.add_line(
            mid_image, (0, quad_seg_point[1]*size), 1e10, color=255, thickness=3
        )
        return scp_width, [mid_image, scp_image]
    return scp_width, None
