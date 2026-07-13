from ortools.sat.python import cp_model


def solve_layer_building(items, pallet_width, pallet_length, G_max_height_gap,
                         time_limit_s=30):
    """Solve the Layer Building sub-problem using CP-SAT.

    Items are placed on a rectangular pallet (2D). The solver selects a subset
    of items that fit without overlap and have similar heights.

    Args:
        items: list of dicts [{'id': i, 'width': w, 'length': l, 'height': h}, ...]
        pallet_width: int, width of the pallet (x-axis)
        pallet_length: int, length of the pallet (y-axis)
        G_max_height_gap: int, max allowed height difference within one layer
        time_limit_s: float, solver time limit in seconds

    Returns:
        (status, solver, p, x, y, lx) where:
        - status: cp_model CpSolverStatus
        - solver: CpSolver instance (call solver.Value(var) to extract)
        - p: BoolVar dict, whether item i is selected
        - x, y: IntVar dicts, lower-left corner position
        - lx: BoolVar dict, True = width parallel to x-axis
    """
    model = cp_model.CpModel()
    N = len(items)

    p, x, y, lx = {}, {}, {}, {}
    le, ri, fr, ba = {}, {}, {}, {}

    for i in range(N):
        p[i] = model.NewBoolVar(f'p_{i}')
        x[i] = model.NewIntVar(0, pallet_width - 1, f'x_{i}')
        y[i] = model.NewIntVar(0, pallet_length - 1, f'y_{i}')
        lx[i] = model.NewBoolVar(f'lx_{i}')  # True => width along x

        for j in range(i + 1, N):
            le[i, j] = model.NewBoolVar(f'le_{i}_{j}')
            ri[i, j] = model.NewBoolVar(f'ri_{i}_{j}')
            fr[i, j] = model.NewBoolVar(f'fr_{i}_{j}')
            ba[i, j] = model.NewBoolVar(f'ba_{i}_{j}')

    for i in range(N):
        wi, li, hi = items[i]['width'], items[i]['length'], items[i]['height']
        max_dim = max(wi, li)

        # Encode rotated dimensions via lx flag
        dim_x = model.NewIntVar(0, max_dim, f'dim_x_{i}')
        dim_y = model.NewIntVar(0, max_dim, f'dim_y_{i}')
        model.Add(dim_x == wi).OnlyEnforceIf(lx[i])
        model.Add(dim_x == li).OnlyEnforceIf(lx[i].Not())
        model.Add(dim_y == li).OnlyEnforceIf(lx[i])
        model.Add(dim_y == wi).OnlyEnforceIf(lx[i].Not())

        # Boundary constraints (only when selected)
        model.Add(x[i] + dim_x <= pallet_width).OnlyEnforceIf(p[i])
        model.Add(y[i] + dim_y <= pallet_length).OnlyEnforceIf(p[i])

        for j in range(i + 1, N):
            wj, lj, hj = items[j]['width'], items[j]['length'], items[j]['height']
            max_dim_j = max(wj, lj)
            dim_xj = model.NewIntVar(0, max_dim_j, f'dim_x_{j}')
            dim_yj = model.NewIntVar(0, max_dim_j, f'dim_y_{j}')
            model.Add(dim_xj == wj).OnlyEnforceIf(lx[j])
            model.Add(dim_xj == lj).OnlyEnforceIf(lx[j].Not())
            model.Add(dim_yj == lj).OnlyEnforceIf(lx[j])
            model.Add(dim_yj == wj).OnlyEnforceIf(lx[j].Not())

            # Non-overlap only needed when both selected
            both = model.NewBoolVar(f'both_{i}_{j}')
            model.AddBoolAnd([p[i], p[j]]).OnlyEnforceIf(both)
            model.AddBoolOr([p[i].Not(), p[j].Not()]).OnlyEnforceIf(both.Not())

            model.Add(le[i, j] + ri[i, j] + fr[i, j] + ba[i, j] >= 1).OnlyEnforceIf(both)

            model.Add(x[i] + dim_x <= x[j]).OnlyEnforceIf(le[i, j])
            model.Add(x[j] + dim_xj <= x[i]).OnlyEnforceIf(ri[i, j])
            model.Add(y[i] + dim_y <= y[j]).OnlyEnforceIf(fr[i, j])
            model.Add(y[j] + dim_yj <= y[i]).OnlyEnforceIf(ba[i, j])

            # Height gap: items in same layer must have similar height
            if abs(hi - hj) > G_max_height_gap:
                model.AddBoolOr([p[i].Not(), p[j].Not()])

    # Objective: maximize total area
    total_area = sum(
        p[i] * (items[i]['width'] * items[i]['length']) for i in range(N))
    model.Maximize(total_area)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.log_search_progress = False
    solver.parameters.num_search_workers = 4

    status = solver.Solve(model)
    return status, solver, p, x, y, lx