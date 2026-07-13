from ortools.linear_solver import pywraplp

def solve_bin_building(layers, max_bins, max_bin_height, max_pallet_weight, max_area_gap, L_big_m=10000):
    """
    复现 Bin Building Sub-Problem
    layers: 预先建好的层级列表，包含每层的高度 H, 重量 W, 面积 A, 稳定性 S
    """
    # 显式调用 SCIP 求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None

    V_num = len(layers)
    
    # --- 1. 决策变量 ---
    n = {} # n[k]: 托盘 k 是否被使用
    v = {} # v[j, k]: 层 j 是否放入托盘 k
    z = {} # z[j]: 层 j 的底部 Z 坐标 (高度)
    o = {} # o[j, j']: 层 j 是否在层 j' 之上
    u = {} # u[j, j']: 层 j 是否在层 j' 之下

    for k in range(max_bins):
        n[k] = solver.IntVar(0, 1, f'n_{k}')
        for j in range(V_num):
            v[j, k] = solver.IntVar(0, 1, f'v_{j}_{k}')
            
    for j in range(V_num):
        z[j] = solver.NumVar(0, max_bin_height, f'z_{j}')
        for j_prime in range(j + 1, V_num):
            o[j, j_prime] = solver.IntVar(0, 1, f'o_{j}_{j_prime}')
            u[j, j_prime] = solver.IntVar(0, 1, f'u_{j}_{j_prime}')

    # --- 2. 约束条件 ---
    for j in range(V_num):
        # 每个层只能且必须放入一个托盘
        solver.Add(sum(v[j, k] for k in range(max_bins)) == 1)

    for k in range(max_bins):
        # 建立 n_k 和 v_j_k 的逻辑联系
        solver.Add(sum(v[j, k] for j in range(V_num)) <= V_num * n[k])
        
        # 重量限制 (公式 30)
        solver.Add(sum(v[j, k] * layers[j]['weight'] for j in range(V_num)) <= max_pallet_weight)
        
        for j in range(V_num):
            # 高度限制: 不超载托盘最高限度 (公式 25)
            solver.Add(z[j] + layers[j]['height'] <= max_bin_height + (1 - v[j, k]) * L_big_m)
            
            for j_prime in range(j + 1, V_num):
                # 相对位置分配逻辑与 Z 轴不重叠 (公式 26-28)
                solver.Add(o[j, j_prime] + u[j, j_prime] >= v[j, k] + v[j_prime, k] - 1)
                solver.Add(z[j] + layers[j]['height'] <= z[j_prime] + (1 - o[j, j_prime]) * L_big_m)
                solver.Add(z[j_prime] + layers[j_prime]['height'] <= z[j] + (1 - u[j, j_prime]) * L_big_m)
                
                # 稳定性约束: 上层稳定性应大于等于下层 (公式 31-32)
                solver.Add(layers[j]['stability'] * o[j, j_prime] <= layers[j_prime]['stability'] * u[j, j_prime])
                
                # 相邻层面积差距约束 (公式 33-34)
                area_diff = layers[j]['area'] - layers[j_prime]['area']
                solver.Add(area_diff * o[j, j_prime] <= max_area_gap)
                solver.Add(area_diff * u[j, j_prime] <= max_area_gap)

    # 对称性破坏：优先使用索引靠前的托盘 (公式 36)
    for k in range(max_bins - 1):
        solver.Add(n[k] >= n[k+1])

    # --- 3. 目标函数 ---
    # 最小化使用的托盘数量
    objective = solver.Objective()
    for k in range(max_bins):
        objective.SetCoefficient(n[k], 1)
    objective.SetMinimization()

    solver.SetTimeLimit(90000)
    status = solver.Solve()
    return status, n, v, z