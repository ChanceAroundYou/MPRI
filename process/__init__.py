from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
from . import first_stage
from . import second_stage
from . import third_stage
from . import output


def clear_bottleneck(image):
    height, width = image.shape
    for y in range(1,height-1):
        for x in range(1,width-1):
            if image[y, x]:
                if not image[y-1, x] and not image[y+1, x]:
                    image[y, x] = False
                elif not image[y, x-1] and not image[y, x+1]:
                    image[y, x] = False
    return image

def get_diff(point_1, point_2):
    return point_1[0] - point_2[0], point_1[1] - point_2[1]

def get_length(y, x):
    return np.sqrt(np.square([y, x]).sum())

def get_grav_center(img):
    moments = cv2.moments(img)
    grav_center_x = int(moments['m10']/moments['m00'])
    grav_center_y = int(moments['m01']/moments['m00'])
    return grav_center_y, grav_center_x

def get_clahe_image(img, limit, row, col):
    height, width = img.shape
    clahe = cv2.createCLAHE(limit * 255, (int(height/row), int(width/col)))
    return clahe.apply(img)

def get_near_component(components, pos, limit, type_='point'):
    assert type_ in ('point', 'area')
    if type_ == 'point':
        near_component = [
            component for component in components
            if get_area_distance(component.img, pos) < limit
        ]
    elif type_ == 'area':
        near_component = [
            component for component in components
            if get_area_distance(pos, component.centroid) < limit
        ]
    return near_component

def get_distance(point_1, point_2):
    diff_y, diff_x = get_diff(point_1, point_2)
    return get_length(diff_y, diff_x)

def get_angle(point_1, point_2, isdeg=True):
    diff_y, diff_x = get_diff(point_1, point_2)
    rad = np.arctan(diff_y/diff_x)
    if isdeg:
        return np.rad2deg(rad)
    return rad

def get_otsu(img, only_otsu=True):
    otsu, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if only_otsu:
        return otsu
    return otsu, otsu_img

def get_area_distance(img, point):
    area_points = list(zip(*np.where(img)))
    distances = [get_distance(area_point, point) for area_point in area_points]
    return min(distances)

def get_bound_point(image, type_='l'):
    y_pos, x_pos = np.where(image)
    if type_ == 'l':
        left_x = x_pos.min()
        left_y = int(y_pos[np.where(x_pos == left_x)[0]].mean())
        bound_point = (left_y, left_x)
    elif type_ == 'r':
        right_x = x_pos.max()
        right_y = int(y_pos[np.where(x_pos == right_x)[0]].mean())
        bound_point = (right_y, right_x)
    elif type_ == 'u':
        up_y = y_pos.min()
        up_x = int(x_pos[np.where(y_pos == up_y)[0]].mean())
        bound_point = (up_y, up_x)
    elif type_ == 'd':
        down_y = y_pos.max()
        down_x = int(x_pos[np.where(y_pos == down_y)[0]].mean())
        bound_point = (down_y, down_x)
    elif 1 < len(type_) <= 4:
        bound_point = [get_bound_point(image, each_type_) for each_type_ in type_]
    else:
        raise TypeError()
    return bound_point

def get_bin_image(img, threshold):
    _, bin_image = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return bin_image

def get_side_contour(src, side='r'):
    img = src.copy()
    height, width = img.shape
    if side == 'r':
        for y in range(height):
            for x in range(width):
                if img[y, x:].any():
                    img[y, x] = img.max()
                else:
                    break
    elif side == 'l':
        for y in range(height):
            for x in range(width-1, 0, -1):
                if img[y, :x].any():
                    img[y, x] = img.max()
                else:
                    break
    elif side == 'u':
        for x in range(width):
            for y in range(height-1, 0, -1):
                if img[:y, x].any():
                    img[y, x] = img.max()
                else:
                    break
    elif side == 'd':
        for x in range(width):
            for y in range(height):
                if img[y:, x].any():
                    img[y, x] = img.max()
                else:
                    break
    return img

def get_rotated_point(point, angle, center, center_after=None):
    if center_after is None:
        center_after = center
    center_y, center_x = center
    center_after_y, center_after_x = center_after
    point_y, point_x = point
    # quad_angle = np.arctan((quad_y - center_y) / (quad_x - center_x))
    rotated_x = (point_x-center_x)*np.cos(angle) + (point_y-center_y)*np.sin(angle) + center_after_x
    rotated_y = (point_y-center_y)*np.cos(angle) - (point_x-center_x)*np.sin(angle) + center_after_y
    return int(rotated_y), int(rotated_x)
# def get_point_by_angle_and_distance(point, angle, distance):
#     point_y, point_x = point
#     angle_y, angle_x = angle
#     angle_distance = (angle_x ** 2 + angle_y ** 2) ** (1 / 2)
#     diff_x = distance / angle_distance * angle_x
#     diff_y = distance / angle_distance * angle_y
#     return (int(point_y + diff_y), int(point_x + diff_x))

# def get_rect_mask(shape, point, height, width, angle=0):
#     start_point = get_revese_point(point)
#     end_point = (point[1] + width, point[0] + height)
#     canvas_height, canvas_width = shape
#     canvas = np.zeros((
#         max(canvas_height, end_point[1]),
#         max(canvas_width, end_point[0])
#         ))
#     rect = cv2.rectangle(canvas, start_point, end_point, 255, -1)
#     if angle:
#         rot_matrix = cv2.getRotationMatrix2D(start_point, angle, 1)
#         rot_shape = (canvas_width, canvas_height)
#         rect = cv2.warpAffine(rect, rot_matrix, rot_shape, flags=cv2.INTER_LINEAR)
#     return rect.astype(bool)[:canvas_height+1, :canvas_width+1]

def get_revese_point(point):
    return point[1], point[0]

def is_point(point):
    check_dtype = lambda num: num.dtype in (np.int, np.int16, np.int32, np.int64)
    try:
        if (isinstance(point, (tuple, list, np.ndarray)) and
                len(point) == 2 and
                (isinstance(point[0], int) or check_dtype(point[0])) and
                (isinstance(point[1], int) or check_dtype(point[1]))):
            return True
    except AttributeError:
        pass
    return False

def add_points(img, points, size=5, color=255, reverse=True):
    if is_point(points):
        points = [points]
    else:
        for point in points:
            assert is_point(point)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    background = img.copy()
    for point in points:
        if reverse:
            point = reversed(point)
        backgorund = cv2.circle(background, tuple(point), size, color, -1)
    return backgorund

def add_line(img, point, param, type_='seg', reverse=True, color=255, thickness=3):
    height, width = img.shape
    assert is_point(point)
    if not is_point(param):
        line_k = param # y/x
        type_ = 'line'
    elif type_ == 'line':
        line_k = (param[0] - point[0]) / (param[1] - point[1])

    if type_ == 'line':
        start = (0, int(point[1] - point[0] / line_k))
        end = (height, int(point[1] + (height - point[0]) / line_k))
    else:
        start = point
        end = param

    if reverse:
        start = reversed(start)
        end = reversed(end)

    return cv2.line(img.copy(), tuple(start), tuple(end), color, thickness)

def add_mask(img, mask, color=255):
    img_copy = img.copy()
    img_copy[mask] = color
    return img_copy

def show(imgs, dim=0, size=6):
    if isinstance(imgs, np.ndarray):
        if len(imgs.shape) == 2:
            imgs = [imgs]
        elif len(imgs.shape) == 3:
            imgs = np.rollaxis(imgs, dim) if dim != 0 else imgs
        else:
            raise TypeError()
    elif hasattr(imgs, 'img'):
        imgs = [imgs]
    assert isinstance(imgs, (list, tuple, np.ndarray))
    img_num = len(imgs)
    col = min(img_num, 2)
    row = np.ceil(img_num / col).astype(dtype=np.int32)
    _, axes = plt.subplots(row, col, figsize=(size * col, size * row))
    axes = axes.reshape(-1) if isinstance(axes, np.ndarray) else [axes]
    for img, ax in zip(imgs, axes):
        if hasattr(img, 'img'):
            ax.imshow(img.img, 'gray')
        else:
            assert isinstance(img, np.ndarray)
            ax.imshow(img, 'gray')

# def normalize(img):
#     return (img / img.max() * 255).astype(np.uint8)
