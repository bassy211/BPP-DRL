"""
Two-phase MILP heuristic for 3D Bin Packing.

Phase 1 (Layer Building): group items by similar height, then solve 2D
    rectangle packing within each layer via MILP.
Phase 2 (Layer Stacking): stack the built layers into the container using
    a greedy height-ordered strategy.
"""

import numpy as np
import copy
from collections import defaultdict

from acktr.Layer_Building_MILP import solve_layer_building


def _greedy_pack_layer(items, pallet_width, pallet_length):
    """Greedy 2D rectangle packing within a single layer.

    Places items in order (largest area first) at the first feasible position.
    Uses a simple heightmap (1D) approach similar to the existing best_fit.

    Returns (placed_items, unplaced_items) where placed_items is a list of dicts
    with pos_x, pos_y, w_x, l_y, h, orig.
    """
    # Sort by area descending
    sorted_items = sorted(items, key=lambda it: it['width'] * it['length'],
                          reverse=True)
    # 2D grid tracking max y-coordinate at each x-position (similar to heightmap)
    heightmap = np.zeros(pallet_width, dtype=int)
    placed = []
    unplaced = []

    for item in sorted_items:
        d, w, h = item['orig']
        best_x, best_y = -1, -1
        best_support = float('inf')

        # Try both orientations
        for orient in [(w, d), (d, w)]:
            wx, ly = orient
            if wx > pallet_width or ly > pallet_length:
                continue
            for px in range(pallet_width - wx + 1):
                support_y = int(np.max(heightmap[px:px + wx]))
                if support_y + ly > pallet_length:
                    continue
                if support_y < best_support or (support_y == best_support and px < best_x):
                    best_support = support_y
                    best_x = px
                    best_y = support_y
                    best_wx, best_ly = wx, ly

        if best_x >= 0:
            heightmap[best_x:best_x + best_wx] = best_y + best_ly
            placed.append({
                'pos_x': best_x, 'pos_y': best_y,
                'w_x': best_wx, 'l_y': best_ly,
                'h': h, 'orig': (d, w, h),
            })
        else:
            unplaced.append(item)

    return placed, unplaced


def _build_layers(items, pallet_width, pallet_length, height_gap=1,
                  time_limit_s=15, max_items_per_batch=8):
    """Group items by height and greedily pack each group into layers.

    For small groups (≤8 items), uses CP-SAT for optimal 2D packing.
    For larger groups, uses greedy best-fit 2D packing.
    """
    from ortools.sat.python import cp_model

    groups = defaultdict(list)
    for idx, (d, w, h) in enumerate(items):
        groups[h].append({'id': idx, 'width': w, 'length': d, 'height': h,
                          'orig': (d, w, h)})

    sorted_heights = sorted(groups.keys())
    super_groups = []
    current_group = []
    current_h = None
    for h in sorted_heights:
        if current_h is None or h - current_h <= height_gap:
            current_group.extend(groups[h])
            current_h = h if current_h is None else max(current_h, h)
        else:
            super_groups.append((current_h, current_group))
            current_group = list(groups[h])
            current_h = h
    if current_group:
        super_groups.append((current_h, current_group))

    all_layers = []

    for group_max_h, group_items in super_groups:
        remaining = list(group_items)

        while remaining:
            if len(remaining) <= max_items_per_batch:
                # CP-SAT for optimal small-batch packing
                status, cp_solver, p_vars, x_vars, y_vars, lx_vars = solve_layer_building(
                    remaining, pallet_width, pallet_length, height_gap,
                    time_limit_s=time_limit_s)
                if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    layer_items = []
                    still_remain = []
                    for i, item in enumerate(remaining):
                        if cp_solver.Value(p_vars[i]):
                            px = cp_solver.Value(x_vars[i])
                            py = cp_solver.Value(y_vars[i])
                            is_lx = cp_solver.Value(lx_vars[i])
                            d, w, h = item['orig']
                            w_along_x = w if is_lx else d
                            l_along_y = d if is_lx else w
                            layer_items.append({
                                'pos_x': int(px), 'pos_y': int(py),
                                'w_x': w_along_x, 'l_y': l_along_y,
                                'h': h, 'orig': (d, w, h),
                            })
                        else:
                            still_remain.append(item)
                    if layer_items:
                        all_layers.append({'items': layer_items, 'height': int(group_max_h)})
                    remaining = still_remain
                    if not still_remain:
                        break
                    continue  # try CP-SAT again with remaining

            # Fallback: greedy packing
            placed, unplaced = _greedy_pack_layer(remaining, pallet_width, pallet_length)
            if placed:
                all_layers.append({'items': placed, 'height': int(group_max_h)})
            remaining = unplaced
            if not unplaced:
                break
            if len(unplaced) == len(remaining):
                # No progress, force-split
                remaining = remaining[:max_items_per_batch]

    return all_layers


def milp_two_phase_pack(env, time_limit_s=30):
    """Run the two-phase MILP heuristic on an environment.

    This is an OFFLINE method: it peeks at all remaining items in the sequence,
    solves the MILP globally, then places boxes accordingly.

    Returns (ratio, counter, elapsed_time).
    """
    import time
    start = time.perf_counter()

    container_w = int(env.bin_size[0])
    container_l = int(env.bin_size[1])
    container_h = int(env.bin_size[2])

    # 1. Collect all items from the box creator
    all_boxes = []
    temp_env = copy.deepcopy(env)
    temp_env.reset()
    temp_env.box_creator.preview(500)  # get all boxes

    # Read boxes until the sentinel (10,10,10) appears
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
        return 0.0, 0, elapsed

    # 2. Build layers via MILP
    layers = _build_layers(all_boxes, container_w, container_l,
                           height_gap=1, time_limit_s=time_limit_s)

    # 3. Greedy layer stacking: stack layers from bottom to top
    # Sort by height (taller layers at bottom for stability)
    layers.sort(key=lambda L: L['height'], reverse=True)

    current_z = 0
    placed_count = 0
    total_volume = 0

    for layer in layers:
        if current_z + layer['height'] > container_h:
            continue  # skip layers that don't fit

        for item in layer['items']:
            d, w, h = item['orig']
            lx, ly = item['pos_x'], item['pos_y']
            wx, ly_dim = item['w_x'], item['l_y']

            # Check bounds
            if lx + wx > container_w or ly + ly_dim > container_l:
                continue
            if current_z + h > container_h:
                continue

            # Create and place the box in the environment
            from envs.bpp0.space import Box
            box = Box(x=wx, y=ly_dim, z=h, lx=lx, ly=ly, lz=current_z)
            env.space.boxes.append(box)
            env.space.flags.append(False)

            placed_count += 1
            total_volume += d * w * h

        current_z += layer['height']

    env.space.rebuild_state()

    # 4. Compute utilization
    container_volume = container_w * container_l * container_h
    ratio = total_volume / container_volume

    # Compute stability metrics
    center_offset = env.space.get_center_offset() if hasattr(env.space, 'get_center_offset') else 0.0
    com = env.space.calculate_center_of_mass() if hasattr(env.space, 'calculate_center_of_mass') else None
    com_x, com_y = com if com else (0.0, 0.0)

    elapsed = time.perf_counter() - start
    return ratio, placed_count, elapsed, 0.0, center_offset, com_x, com_y
