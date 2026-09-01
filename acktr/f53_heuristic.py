"""
F53 Online Heuristic for 3D Bin Packing (Ali et al., 2024).

Based on the Empty Maximal Space (EMS) paradigm:
- Space Selection Strategy 5: smallest X, smallest Z, closest to rear-bottom corners
- Item Placement Strategy 3: minimum gap (wasted space) within the chosen EMS

This is an ONLINE algorithm: items are processed one-by-one in arrival order.
"""

import numpy as np
import copy
import time


class EMS:
    """Empty Maximal Space — a rectangular empty region in the container."""
    __slots__ = ('x', 'y', 'z', 'l', 'w', 'h')

    def __init__(self, x, y, z, l, w, h):
        self.x = x  # lower-left-back corner X
        self.y = y  # lower-left-back corner Y
        self.z = z  # lower-left-back corner Z
        self.l = l  # length along X
        self.w = w  # width  along Y
        self.h = h  # height along Z

    @property
    def volume(self):
        return self.l * self.w * self.h

    def contains(self, other):
        """Whether self fully contains other EMS."""
        return (self.x <= other.x and self.y <= other.y and self.z <= other.z and
                self.x + self.l >= other.x + other.l and
                self.y + self.w >= other.y + other.w and
                self.z + self.h >= other.z + other.h)

    def __repr__(self):
        return (f"EMS(x={self.x},y={self.y},z={self.z},"
                f"l={self.l},w={self.w},h={self.h})")


def _split_ems(ems, px, py, pz, il, iw, ih):
    """Split an EMS after placing an item of size (il,iw,ih) at (px,py,pz).

    Returns a list of new EMS fragments (may be empty if item fills the EMS).
    """
    ex, ey, ez = ems.x, ems.y, ems.z
    el, ew, eh = ems.l, ems.w, ems.h

    fragments = []

    # Left of the item
    if px > ex:
        fragments.append(EMS(ex, ey, ez, px - ex, ew, eh))
    # Right of the item
    if px + il < ex + el:
        fragments.append(EMS(px + il, ey, ez, ex + el - px - il, ew, eh))
    # Front of the item (smaller Y)
    if py > ey:
        fragments.append(EMS(px, ey, ez, il, py - ey, eh))
    # Back of the item (larger Y)
    if py + iw < ey + ew:
        fragments.append(EMS(px, py + iw, ez, il, ey + ew - py - iw, eh))
    # Below the item
    if pz > ez:
        fragments.append(EMS(px, py, ez, il, iw, pz - ez))
    # Above the item
    if pz + ih < ez + eh:
        fragments.append(EMS(px, py, pz + ih, il, iw, ez + eh - pz - ih))

    return fragments


def _remove_redundant_ems(ems_list):
    """Remove EMS that are fully contained within another EMS or have zero volume."""
    # Filter zero-volume
    kept = [e for e in ems_list if e.volume > 0]
    # Remove contained
    result = []
    for i, ei in enumerate(kept):
        contained = False
        for j, ej in enumerate(kept):
            if i != j and ej.contains(ei):
                contained = True
                break
        if not contained:
            result.append(ei)
    return result


def _strategy5_sort_key(ems, bin_w):
    """Space Selection Strategy 5: smallest X, smallest Z,
    closest to one of the two rear-bottom corners (0,0,0) or (0,W,0)."""
    dist_left = ems.x ** 2 + ems.y ** 2 + ems.z ** 2
    dist_right = ems.x ** 2 + (bin_w - ems.y) ** 2 + ems.z ** 2
    return (ems.x, ems.z, min(dist_left, dist_right))


def _strategy3_place(item_dims, ems):
    """Item Placement Strategy 3: among all 6 rotations that fit,
    choose the one with minimum gap (sum of wasted lengths on each axis)."""
    d, w, h = item_dims
    rotations = [(d, w, h), (w, d, h), (d, h, w),
                 (h, d, w), (w, h, d), (h, w, d)]
    best_rot = None
    min_gap = float('inf')
    for rl, rw, rh in rotations:
        if rl <= ems.l and rw <= ems.w and rh <= ems.h:
            gap = (ems.l - rl) + (ems.w - rw) + (ems.h - rh)
            if gap < min_gap:
                min_gap = gap
                best_rot = (rl, rw, rh)
    return best_rot


def f53_online_pack(env):
    """Run the F53 online heuristic.

    Peek at all items, then process them one-by-one in arrival order using
    the EMS-based Strategy 5 (space selection) + Strategy 3 (placement).

    Returns (ratio, counter, elapsed, depen_rate, center_offset, com_x, com_y).
    """
    start = time.perf_counter()

    container_w = int(env.bin_size[0])  # X
    container_l = int(env.bin_size[1])  # Y
    container_h = int(env.bin_size[2])  # Z

    # 1. Collect all items
    all_boxes = []
    temp_env = copy.deepcopy(env)
    temp_env.reset()
    temp_env.box_creator.preview(500)
    while True:
        preview = temp_env.box_creator.preview(1)
        if not preview:
            break
        box = preview[0]
        if tuple(box) == (10, 10, 10):
            break
        all_boxes.append(tuple(box))
        temp_env.box_creator.drop_box()
        temp_env.box_creator.preview(500)

    if not all_boxes:
        elapsed = time.perf_counter() - start
        return 0.0, 0, elapsed, 0.0, 0.0, 0.0, 0.0

    # 2. Initialize EMS list with the whole container
    ems_list = [EMS(0, 0, 0, container_w, container_l, container_h)]
    placed_boxes = []

    # 3. Process each item online
    for box_dims in all_boxes:
        d, w, h = box_dims  # (depth=X, width=Y, height=Z)

        # Sort EMS by Strategy 5
        ems_list.sort(key=lambda e: _strategy5_sort_key(e, container_l))

        placed = False
        for ems in ems_list:
            best_rot = _strategy3_place((d, w, h), ems)
            if best_rot is None:
                continue

            rl, rw, rh = best_rot
            px, py, pz = ems.x, ems.y, ems.z

            # Place the box in the environment
            from envs.bpp0.space import Box
            box = Box(x=rl, y=rw, z=rh, lx=px, ly=py, lz=pz)
            env.space.boxes.append(box)
            env.space.flags.append(False)
            placed_boxes.append((rl, rw, rh, px, py, pz))

            # Split EMS
            new_ems = _split_ems(ems, px, py, pz, rl, rw, rh)
            ems_list.remove(ems)
            ems_list.extend(new_ems)
            ems_list = _remove_redundant_ems(ems_list)
            placed = True
            break

        if not placed:
            # Item doesn't fit anywhere — skip it (online algorithm constraint)
            pass

    env.space.rebuild_state()

    # 4. Compute metrics
    total_volume = sum(rl * rw * rh for rl, rw, rh, _, _, _ in placed_boxes)
    container_volume = container_w * container_l * container_h
    ratio = total_volume / container_volume

    center_offset = env.space.get_relative_offset_ratio() if hasattr(env.space, 'get_relative_offset_ratio') else 0.0
    com = env.space.calculate_center_of_mass() if hasattr(env.space, 'calculate_center_of_mass') else None
    com_x, com_y = com if com else (0.0, 0.0)

    elapsed = time.perf_counter() - start
    return ratio, len(placed_boxes), elapsed, 0.0, center_offset, com_x, com_y
