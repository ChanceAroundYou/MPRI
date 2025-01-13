from __future__ import print_function
from __future__ import division
from __future__ import with_statement
from typing import List, Tuple

import os
import cv2
import numpy as np
import SimpleITK as sitk

from config import Mcp_show_info_item, Point, Scp_show_info_item


def calc(
    dir_path: str, pons_area: int, midbrain_area: int, mcp_width: float, scp_width: float, file_name: str=None
) -> None:
    mrpi = (pons_area / midbrain_area) * (mcp_width / scp_width)
    report = 'Pons area: {}mm^2, Midbrain area: {}mm^2, MCP width: {}mm, SCP width: {}mm, MRPI: {}'. format(
        pons_area, midbrain_area, mcp_width, scp_width, mrpi
    )
    if file_name:
        with open(os.path.join(dir_path, file_name), 'w+') as file:
            file.write(report)
    else:
        print(report)

def save(
    dir_path: str, size: Tuple[int, int], mid_num: int, quad_seg_point: Point,
    mcp_show_info: List[Mcp_show_info_item], scp_show_info: Scp_show_info_item
) -> None:
    result = np.zeros(size, dtype=np.uint8)
    # quad_seg_point
    result[quad_seg_point][mid_num] = 1
    # mcp and mcp_seg_point
    for index, (up_point, down_point, mcp) in mcp_show_info:
        result[:, :, index][mcp.img_bool] = 2
        result[up_point][index] = 3
        result[down_point][index] = 3
    if scp_show_info is not None:
        mid_image, scp_image = scp_show_info
        cv2.imwrite(os.path.join(dir_path, 'SCP_cut_line.jpg'), mid_image)
        cv2.imwrite(os.path.join(dir_path, 'SCP.jpg'), scp_image)

    result = np.rot90(result, k=2, axes=(0, 1))
    result = sitk.GetImageFromArray(result)
    sitk.WriteImage(result, os.path.join(dir_path, 'Seg_result.nii.gz'))

def save_to_image():
    raise NotImplementedError()
