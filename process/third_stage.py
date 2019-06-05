from typing import List, Sequence, Tuple

import cv2
import numpy as np

import component
import process
from config import Box, Point, Scp_components, Scp_show_info_item
from DM.file_manage import LabelNiiFileManager, RotatedNiiFileManager

def _get_peak(image: np.ndarray, start: int, end: int) -> Tuple[int, int]:
    direct = 1 if start < end else -1
    min_y = 200
    min_x = start
    for x in range(start, end, direct):
        y = (np.where(image[:, x]))[0].min()
        if y < min_y:
            count = 0
            min_y = y
            min_x = x
        elif count < 3:
            count += 1
        else:
            break
    return min_y, min_x
    # left_peak_list = [
    #     (np.where(image[:, x])[0].min(), x)
    #     for x in range(start, end)
    # ]
    # left_peak_list.sort(key=lambda x: x[0])
    # return left_peak_list[0]

def get_scp_sum(label_nii: LabelNiiFileManager, quad_seg_point: Point, scp_label: int=28) -> int:
    quad_num = int(quad_seg_point[1])

    # for num in range(quad_num-3, quad_num+5):
    #     label = label_nii.get_slice(num, dim=1)
    #     scp = label == scp_label
    #     if scp.sum():
    #         return num
    return quad_num

def get_scp_components(
    image: np.ndarray, label: np.ndarray, quad_point: Point, accubrain=False,
    scp_label: int=28, pons_label: int=26, medulla_label: int=27, rate: float=0.92,
    box: Box=((3, 13), (-10, 10)), enhance_y: int=5, enhance_x: int=3,
    max_width: int=12, min_width: int=2, debug: bool=False
) -> Scp_components:
    if debug:
        process.show(process.add_points(image, quad_point, 1))
    mask_box = (
        slice(quad_point[0]+box[0][0], quad_point[0]+box[0][1]),
        slice(quad_point[1]+box[1][0], quad_point[1]+box[1][1])
    )
    mask = np.zeros(image.shape, bool)
    mask[mask_box] = True
    cere_pon_mask = np.zeros(image.shape, bool)
    cere_pon_mask[
        quad_point[0]+box[0][0]:quad_point[0]+50,
        quad_point[1]-40:quad_point[1]+40
    ] = True
    scp = np.logical_or(np.logical_or(label==scp_label, label==pons_label), label==medulla_label)
    scp_mean = image[np.where(scp*mask)].mean()
    bin_image = process.clear_bottleneck(image * cere_pon_mask > scp_mean * rate)
    if not accubrain:
        cere_pon = component.get_connected_component(bin_image)[0][0]
    else:
        cere_pon = component.get_connected_component(scp)[0][0]
    if debug:
        process.show([
            bin_image, cere_pon
        ])
    left_peak_y, left_peak_x = _get_peak(
        cere_pon.img,
        quad_point[1]-1,
        quad_point[1]-10
    )

    right_peak_y, right_peak_x = _get_peak(
        cere_pon.img,
        quad_point[1]+1,
        quad_point[1]+10
    )
    mean_peak = (
        int((left_peak_y + right_peak_y) / 2),
        int((left_peak_x + right_peak_x) / 2)
    )

    y, x = left_peak_y-1, left_peak_x
    while x > left_peak_x - 12:
        if not cere_pon.img[y+1, x]:
            y += 1
        elif x == cere_pon.left:
            break
        elif not cere_pon.img[y, x-1]:
            x -= 1
        else:
            corner = cere_pon.img[y+1:y+enhance_y+1, x-enhance_x+1:x+1]
            if corner.all():
                break
            else:
                false_pos = np.where(corner==False)
                _y = false_pos[0].min()
                _x = enhance_x - 1 - false_pos[1].max()
                cere_pon.img[y, x-_x:x] = False
                cere_pon.img[y:y+_y+1, x-_x] = False

    left_y, left_x = y, x

    y, x = right_peak_y-1, right_peak_x
    while x < right_peak_x + 12:
        if not cere_pon.img[y+1, x]:
            y += 1
        elif x == cere_pon.right:
            break
        elif not cere_pon.img[y, x+1]:
            x += 1
        else:
            corner = cere_pon.img[y+1:y+enhance_y+1, x:x+enhance_x]
            if corner.all():
                break
            else:
                false_pos = np.where(corner==False)
                _y = false_pos[0].min()
                _x = false_pos[1].max()
                cere_pon.img[y, x:x+_x+1] = False
                cere_pon.img[y:y+_y+1, x+_x] = False

    right_y, right_x = y, x

    for y in range(min(right_y, left_y), max(right_peak_y, left_peak_y), -1):
        row_pos = np.where(cere_pon.img[y]==False)[0]
        if row_pos.any():
            row_pos_bound = np.convolve(row_pos, [1, -1], 'vaild')
            row_pos_seg = np.where(row_pos_bound > 1)[0] + 1
            row_seg_list = np.split(row_pos, row_pos_seg)
            row_seg_list.sort(key=lambda x: abs(x.mean() - quad_point[1]))
            center_row_seg = row_seg_list[0]
            if abs(center_row_seg.mean() - quad_point[1]) < 5:
                center_y = y
                break

    cere_pon_image_mask = np.zeros(image.shape, bool)
    cere_pon_image_mask[:min(left_y, right_y), left_x:right_x] = True
    cere_pon_image = cere_pon.img_bool * cere_pon_image_mask
    components = component.get_connected_component(cere_pon_image, 10)[0]
    if debug:
        process.show([
            cere_pon_image,
            process.add_points(
                cere_pon_image, [
                    (right_peak_y, right_peak_x),
                    (left_peak_y, left_peak_x),
                    quad_point
                ], size=2, color=127
            )
        ])
    try:
        big_component = components[0]
    except IndexError:
        assert not accubrain
        return get_scp_components(
            image, label, quad_point, accubrain=True, rate=rate,
            scp_label=scp_label, pons_label=pons_label, medulla_label=medulla_label,
            box=box, enhance_y=enhance_y, enhance_x=enhance_x, debug=debug,
            max_width=max_width, min_width=min_width
        )
    if big_component.right - big_component.left > 15 or len(components) < 2:
        big_component.img[
            center_y-1:,
            [center_row_seg]
        ] = False
        scp_components, _ = component.get_connected_component(
            big_component.img_uint8
        )
    else:
        components.sort(key=lambda x: process.get_area_distance(x.img, mean_peak))
        scp_components = components[:2]
    assert len(scp_components) == 2

    for scp_component in scp_components:
        if scp_component.right > quad_point[1] + 3:
            for col in range(scp_component.right, scp_component.left, -1):
                if np.where(scp_component.img[:, col])[0].max() != scp_component.down:
                    scp_component.img[:, col] = False
                else:
                    break
            scp_component_row_sum = process.get_side_contour(scp_component.img, 'r').sum(1)
        else:
            for col in range(scp_component.left, scp_component.right):
                if np.where(scp_component.img[:, col])[0].max() != scp_component.down:
                    scp_component.img[:, col] = False
                else:
                    break
            scp_component_row_sum = process.get_side_contour(scp_component.img, 'l').sum(1)

        row_diff = np.convolve(scp_component_row_sum, [1, -1])
        row_grad_diff = np.convolve(row_diff, [1, -1])
        big_diff = np.where(np.logical_and(row_grad_diff >= 2, row_grad_diff < 10))[0]
        big_diff = big_diff[big_diff > scp_component.down - 3]
        if big_diff.any():
            scp_component.img[big_diff[0]:, :] = False
    if debug:
        process.show(scp_components)
    try:
        scp_widths = [
            get_scp_width(scp_component, max_width=max_width, min_width=min_width)
            for scp_component in scp_components
        ]
    except AssertionError:
        assert not accubrain
        return get_scp_components(
            image, label, quad_point, accubrain=True, rate=rate,
            scp_label=scp_label, pons_label=pons_label, medulla_label=medulla_label,
            box=box, enhance_y=enhance_y, enhance_x=enhance_x, debug=debug,
            max_width=max_width, min_width=min_width
        )
    return scp_components, scp_widths

def get_scp_width(
    component: component.ConnectedComponent,
    min_width: int=2, max_width: int=12
) -> int:
    contours = cv2.findContours(
        component.img_uint8,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE
        )
    min_rect = cv2.minAreaRect(contours[1][0])
    # box = np.int32(np.around(cv2.boxPoints(min_rect)))
    box = cv2.boxPoints(min_rect)
    width = min(
        process.get_distance(box[0], box[1]),
        process.get_distance(box[0], box[2]),
        process.get_distance(box[0], box[3])
    )
    assert width < max_width
    assert min_width < width
    return width

def run(
    image_nii: RotatedNiiFileManager, label_nii: LabelNiiFileManager,
    quad_seg_point: Point, mid_num: int, num: int=2, rate: float=0.92,
    scp_label: int=28, pons_label: int=26,  medulla_label: int=27,
    max_width: int=12, min_width: int=2, box: Box=((3, 13), (-10, 10)), show: bool=False, debug: bool=False
) -> Tuple[float, List[Scp_show_info_item]]:
    scp_num = get_scp_sum(label_nii, quad_seg_point, scp_label=scp_label)
    quad_point = (quad_seg_point[0], mid_num)
    scp_widths_list = []
    show_info = []

    count = 0
    for index, (image, label) in enumerate(zip(
        image_nii.get_slice(scp_num, scp_num+5, dim=1),
        label_nii.get_slice(scp_num, scp_num+5, dim=1)
    )):
        try:
            scp_components, scp_widths = get_scp_components(
                image, label, quad_point, accubrain=False, box=box,
                scp_label=scp_label, pons_label=pons_label, rate=rate, debug=debug,
                max_width=max_width, min_width=min_width
            )
            scp_widths_list.append(scp_widths)
            if show:
                show_info.append((scp_num+index, scp_components))
            count += 1
            if count >= 2:
                break
        except AssertionError:
            pass

    scp_mean_width = np.mean(scp_widths_list)
    return scp_mean_width, show_info

def show(image_nii: RotatedNiiFileManager, show_info: List[Scp_show_info_item], mask_color: int=25) -> None:
    process.show([
        process.add_mask(
            process.add_mask(
                image_nii.get_slice(index, dim=1), components[0].img_bool, mask_color
            ), components[1].img_bool, mask_color
        ) for index, components in show_info
    ])
