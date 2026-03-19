import numpy as np
from acktr.heuristic_runtime import get_pack_score_fn, get_unpack_score_fn


def parse_pusnet_observation(observation, container_size):
    if not isinstance(observation, np.ndarray):
        obs = observation.detach().cpu().numpy()
    else:
        obs = observation
    w, l, h = container_size
    area = w * l
    voxel_len = area * h
    voxel = obs[:voxel_len].reshape((w, l, h))
    size_maps = obs[voxel_len: voxel_len + 3 * area].reshape((3, w, l))
    item = (
        int(size_maps[0, 0, 0]),
        int(size_maps[1, 0, 0]),
        int(size_maps[2, 0, 0]),
    )
    return voxel, size_maps, item


def voxel_to_height_map(voxel):
    w, l, h = voxel.shape
    occ = voxel == 2
    plain = np.zeros((w, l), dtype=np.int32)
    for i in range(w):
        for j in range(l):
            col = np.where(occ[i, j])[0]
            plain[i, j] = 0 if len(col) == 0 else int(col[-1] + 1)
    return plain


def _check_box(plain, x, y, lx, ly, z, container_size):
    if lx + x > container_size[0] or ly + y > container_size[1]:
        return -1
    if lx < 0 or ly < 0:
        return -1
    rec = plain[lx:lx + x, ly:ly + y]
    max_h = np.max(rec)
    max_area = np.sum(rec == max_h)
    area = x * y

    if max_h + z > container_size[2]:
        return -1

    lu = int(rec[0, 0] == max_h)
    ld = int(rec[x - 1, 0] == max_h)
    ru = int(rec[0, y - 1] == max_h)
    rd = int(rec[x - 1, y - 1] == max_h)

    if max_area / area > 0.95:
        return max_h
    if lu + ld + ru + rd == 3 and max_area / area > 0.85:
        return max_h
    if lu + ld + ru + rd == 4 and max_area / area > 0.50:
        return max_h
    return -1


def pack_feasibility_map(voxel, item, container_size):
    w, l, _ = container_size
    x, y, z = item
    plain = voxel_to_height_map(voxel)
    feasible = np.zeros((w, l), dtype=np.int32)
    for i in range(w - x + 1):
        for j in range(l - y + 1):
            if _check_box(plain, x, y, i, j, z, container_size) >= 0:
                feasible[i, j] = 1
    return feasible, plain


def pack_heuristic_map(plain, item, feasible):
    if feasible.sum() == 0:
        return feasible.copy()
    w, l = feasible.shape
    x, y, _ = item

    heights = plain.astype(np.float32)
    max_h = max(1.0, float(np.max(heights)))
    low_height_score = 1.0 - heights / max_h

    edge = np.zeros_like(heights)
    edge[0, :] = 1
    edge[:, 0] = 1
    edge[w - 1, :] = 1
    edge[:, l - 1] = 1

    support = np.zeros_like(heights)
    for i in range(w):
        for j in range(l):
            i2 = min(w, i + x)
            j2 = min(l, j + y)
            patch = heights[i:i2, j:j2]
            if patch.size > 0:
                support[i, j] = float(np.mean(patch == np.max(patch)))

    score = 0.45 * low_height_score + 0.30 * edge + 0.25 * support
    score = score * feasible
    valid_scores = score[feasible == 1]
    threshold = np.median(valid_scores) if valid_scores.size > 0 else 0.0
    heur = (score >= threshold).astype(np.int32) * feasible
    if heur.sum() == 0:
        return feasible.copy()
    return heur


def unpack_feasibility_map(voxel):
    w, l, h = voxel.shape
    occ = voxel == 2
    feasible = np.zeros((w, l), dtype=np.int32)
    for i in range(w):
        for j in range(l):
            col = np.where(occ[i, j])[0]
            if len(col) > 0:
                feasible[i, j] = 1
    return feasible


def unpack_heuristic_map(voxel, item, feasible):
    if feasible.sum() == 0:
        return feasible.copy()

    w, l, h = voxel.shape
    occ = voxel == 2
    cur_vol = max(1, int(item[0] * item[1] * item[2]))
    score = np.zeros((w, l), dtype=np.float32)

    for i in range(w):
        for j in range(l):
            if feasible[i, j] == 0:
                continue
            top = np.where(occ[i, j])[0][-1]
            removed_vol = 1
            if removed_vol > cur_vol:
                continue

            free_above = h - 1 - top
            cavity_gain = 0.0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = i + dx, j + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= l:
                    continue
                ncol = np.where(occ[nx, ny])[0]
                ntop = -1 if len(ncol) == 0 else ncol[-1]
                cavity_gain += max(0, ntop - top)

            score[i, j] = 0.6 * free_above + 0.4 * cavity_gain

    valid_scores = score[feasible == 1]
    threshold = np.percentile(valid_scores, 60) if valid_scores.size > 0 else 0.0
    heur = (score >= threshold).astype(np.int32) * feasible
    if heur.sum() == 0:
        return feasible.copy()
    return heur


def _to_binary_map(score, feasible, percentile=60):
    score = np.asarray(score, dtype=np.float32)
    if score.shape != feasible.shape:
        score = np.reshape(score, feasible.shape)
    score = score * feasible
    valid = score[feasible == 1]
    if valid.size == 0:
        return feasible.copy()
    thr = np.percentile(valid, percentile)
    out = (score >= thr).astype(np.int32) * feasible
    if out.sum() == 0:
        return feasible.copy()
    return out


def build_pusnet_action_mask(observation, container_size, use_modulation=True):
    voxel, _, item = parse_pusnet_observation(observation, container_size)
    pack_feasible, plain = pack_feasibility_map(voxel, item, container_size)
    unpack_feasible = unpack_feasibility_map(voxel)

    if not use_modulation:
        pack_fused = pack_feasible
        unpack_fused = unpack_feasible
    else:
        pack_fn = get_pack_score_fn()
        unpack_fn = get_unpack_score_fn()

        if pack_fn is not None:
            try:
                pack_heur = _to_binary_map(pack_fn(plain, item, pack_feasible, container_size), pack_feasible)
            except Exception:
                pack_heur = pack_heuristic_map(plain, item, pack_feasible)
        else:
            pack_heur = pack_heuristic_map(plain, item, pack_feasible)

        if unpack_fn is not None:
            try:
                unpack_heur = _to_binary_map(unpack_fn(voxel, item, unpack_feasible, container_size), unpack_feasible)
            except Exception:
                unpack_heur = unpack_heuristic_map(voxel, item, unpack_feasible)
        else:
            unpack_heur = unpack_heuristic_map(voxel, item, unpack_feasible)

        pack_fused = (pack_feasible * pack_heur).astype(np.int32)
        unpack_fused = (unpack_feasible * unpack_heur).astype(np.int32)

    if pack_fused.sum() == 0:
        pack_fused = pack_feasible
    if unpack_fused.sum() == 0:
        unpack_fused = unpack_feasible

    full = np.hstack((pack_fused.reshape(-1), unpack_fused.reshape(-1))).astype(np.int32)
    if full.sum() == 0:
        full[:] = 1
    return full.tolist()
