import glob
import os
import torch.nn as nn
import numpy as np
from acktr.envs import VecNormalize


def check_box(plain, x, y, lx, ly, z, container_size):
    """
    Keep the same placement feasibility rule as envs/bpp0/space.py::Space.check_box
    to avoid train-time/action-mask mismatch with env execution.
    """
    if lx + x > container_size[0] or ly + y > container_size[1]:
        return -1
    if lx < 0 or ly < 0:
        return -1

    rec = plain[lx:lx + x, ly:ly + y]
    max_h = np.max(rec)

    assert max_h >= 0
    if max_h + z > container_size[2]:
        return -1

    center_x = lx + x / 2.0
    center_y = ly + y / 2.0
    cx_floor = int(np.floor(center_x))
    cy_floor = int(np.floor(center_y))
    cx_floor = min(cx_floor, lx + x - 1)
    cy_floor = min(cy_floor, ly + y - 1)

    center_height = rec[cx_floor - lx, cy_floor - ly]

    max_height_mask = (rec == max_h)
    max_height_points = np.argwhere(max_height_mask)
    if len(max_height_points) == 0:
        return -1

    support_area = len(max_height_points)
    box_area = x * y
    min_support_ratio = 0.15
    if support_area / box_area < min_support_ratio:
        return -1

    if center_height == max_h:
        if box_area <= 2 or support_area / box_area > 0.35:
            return max_h

    if len(max_height_points) == 1:
        px, py = max_height_points[0]
        point_x = lx + px + 0.5
        point_y = ly + py + 0.5
        if x <= 2 and y <= 2:
            dist_sq = (center_x - point_x) ** 2 + (center_y - point_y) ** 2
            threshold = 0.4 if (x == 2 and y == 2) else 0.5
            if dist_sq <= threshold:
                return max_h
        return -1

    support_points = []
    for px, py in max_height_points:
        point_x = lx + px + 0.5
        point_y = ly + py + 0.5
        support_points.append([point_x, point_y])
    support_points = np.array(support_points)

    if len(max_height_points) == 2:
        p1, p2 = support_points[0], support_points[1]
        line_vec = p2 - p1
        point_vec = np.array([center_x, center_y]) - p1
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-6:
            dist_sq = np.dot(point_vec, point_vec)
            return max_h if dist_sq <= 0.5 else -1

        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)
        projection = p1 + t * line_vec
        dist_sq = np.sum((np.array([center_x, center_y]) - projection) ** 2)
        return max_h if dist_sq <= 0.5 else -1

    min_x = support_points[:, 0].min()
    max_x = support_points[:, 0].max()
    min_y = support_points[:, 1].min()
    max_y = support_points[:, 1].max()

    support_span_x = max_x - min_x
    support_span_y = max_y - min_y
    min_span_ratio = 0.4
    if support_span_x < x * min_span_ratio and support_span_y < y * min_span_ratio:
        return -1

    if min_x <= center_x <= max_x and min_y <= center_y <= max_y:
        return max_h

    dx = max(0, min_x - center_x, center_x - max_x)
    dy = max(0, min_y - center_y, center_y - max_y)
    dist_sq = dx ** 2 + dy ** 2
    if dist_sq <= 0.5:
        return max_h

    return -1

def _fit_height_boxes(plain):
    """
    Fit the height map into axis-aligned rectangles using only height values.
    Adjacent grids with the same height are treated as one region and then
    decomposed into rectangles (fitted boxes): (x', y', l', w', h').
    """
    width, length = plain.shape
    fitted_boxes = []

    for h in np.unique(plain):
        mask = (plain == h).astype(np.int32)

        while mask.any():
            points = np.argwhere(mask == 1)
            order = np.lexsort((points[:, 1], points[:, 0]))
            x0, y0 = points[order[0]]

            w = 0
            y = y0
            while y < length and mask[x0, y] == 1:
                w += 1
                y += 1

            l = 1
            x = x0 + 1
            while x < width and np.all(mask[x, y0:y0 + w] == 1):
                l += 1
                x += 1

            mask[x0:x0 + l, y0:y0 + w] = 0
            fitted_boxes.append({
                "x": int(x0),
                "y": int(y0),
                "l": int(l),
                "w": int(w),
                "h": int(h),
                "z": 0,
            })

    return fitted_boxes


def _nearest_negative_x_intersection(vx, vy, fitted_boxes):
    x_hit = 0
    for b in fitted_boxes:
        bx0, bx1 = b["x"], b["x"] + b["l"]
        by0, by1 = b["y"], b["y"] + b["w"]
        if by0 <= vy <= by1 and bx1 <= vx:
            x_hit = max(x_hit, bx1)
    return x_hit


def _nearest_negative_y_intersection(vx, vy, fitted_boxes):
    y_hit = 0
    for b in fitted_boxes:
        bx0, bx1 = b["x"], b["x"] + b["l"]
        by0, by1 = b["y"], b["y"] + b["w"]
        if bx0 <= vx <= bx1 and by1 <= vy:
            y_hit = max(y_hit, by1)
    return y_hit

def generate_candidate_map(plain):
    """
    EP-style potential placement position generation from a height map.

    1) Build alternative state by fitting equal-height regions into boxes.
    2) For fitted-box vertices, project along negative x/y to nearest
       intersections with fitted boxes or boundary.
    3) Use intersection points as candidate placement positions.
    """
    width, length = plain.shape
    fitted_boxes = _fit_height_boxes(plain)

    candidates = set()
    candidates.add((0, 0))

    for b in fitted_boxes:
        x0, y0 = b["x"], b["y"]
        x1, y1 = x0 + b["l"], y0 + b["w"]

        # Corresponding to (x'+l', y', z'), (x', y'+w', z'), (x', y', z'+h')
        vertices = [
            (x1, y0),
            (x0, y1),
            (x0, y0),
        ]

        for vx, vy in vertices:
            x_proj = _nearest_negative_x_intersection(vx, vy, fitted_boxes)
            y_proj = _nearest_negative_y_intersection(vx, vy, fitted_boxes)

            candidates.add((x_proj, vy))
            candidates.add((vx, y_proj))
            candidates.add((x_proj, y_proj))

    candidate_map = np.zeros((width, length), dtype=np.int32)
    for x, y in candidates:
        if 0 <= x < width and 0 <= y < length:
            candidate_map[x, y] = 1

    return candidate_map

def get_possible_position(observation, container_size):
    if not isinstance(observation, np.ndarray):
        box_info = observation.cpu().numpy()
    else:
        box_info = observation
    box_info = box_info.reshape((4,-1))
    x = int(box_info[1][0])
    y = int(box_info[2][0])
    z = int(box_info[3][0])

    plain = box_info[0].reshape((container_size[0],container_size[1]))

    width = container_size[0]
    length = container_size[1]

    action_mask = np.zeros(shape=(width, length), dtype=np.int32)

    for i in range(width - x + 1):
        for j in range(length - y + 1):
            if check_box(plain, x, y, i, j, z, container_size) >= 0:
                action_mask[i, j] = 1

    candidate_map = generate_candidate_map(plain)
    if candidate_map.sum() > 0:
        filtered_mask = action_mask * candidate_map
        valid_cnt = int(action_mask.sum())
        filtered_cnt = int(filtered_mask.sum())
        min_keep = max(3, int(0.2 * valid_cnt))
        if filtered_cnt >= min_keep:
            action_mask = filtered_mask

    if action_mask.sum() == 0:
        action_mask[:, :] = 1

    return action_mask.reshape((-1,)).tolist()

def get_rotation_mask(observation, container_size):
    box_info = observation.cpu().numpy()
    box_info = box_info.reshape((4,-1))
    x = int(box_info[1][0])
    y = int(box_info[2][0])
    z = int(box_info[3][0])

    plain = box_info[0].reshape((container_size[0],container_size[1]))

    width = container_size[0]
    length = container_size[1]

    action_mask1 = np.zeros(shape=(width, length), dtype=np.int32)
    action_mask2 = np.zeros(shape=(width, length), dtype=np.int32)

    for i in range(width - x + 1):
        for j in range(length - y + 1):
            if check_box(plain, x, y, i, j, z, container_size) >= 0:
                action_mask1[i, j] = 1

    for i in range(width - y + 1):
        for j in range(length - x + 1):
            if check_box(plain, y, x, i, j, z, container_size) >= 0:
                action_mask2[i, j] = 1

    candidate_map = generate_candidate_map(plain)
    if candidate_map.sum() > 0:
        filtered_mask1 = action_mask1 * candidate_map
        filtered_mask2 = action_mask2 * candidate_map

        valid_cnt1 = int(action_mask1.sum())
        valid_cnt2 = int(action_mask2.sum())
        min_keep1 = max(3, int(0.2 * valid_cnt1))
        min_keep2 = max(3, int(0.2 * valid_cnt2))

        if int(filtered_mask1.sum()) >= min_keep1:
            action_mask1 = filtered_mask1
        if int(filtered_mask2.sum()) >= min_keep2:
            action_mask2 = filtered_mask2

    action_mask = np.hstack((action_mask1.reshape((-1,)), action_mask2.reshape((-1,))))

    if action_mask.sum() == 0:
        action_mask[:] = 1

    return action_mask

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
