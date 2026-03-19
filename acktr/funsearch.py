import json
import os
import random
import time
import uuid

import numpy as np

from acktr.heuristic_runtime import set_runtime_heuristics
from acktr.llm_heuristic import LLMHeuristicClient
from acktr.pusnet_utils import parse_pusnet_observation, pack_feasibility_map, unpack_feasibility_map


def _safe_compile(code, fn_name):
    ctx = {'np': np}
    try:
        exec(code, ctx)
    except Exception:
        return None
    fn = ctx.get(fn_name)
    return fn if callable(fn) else None


def _default_pack_program_code():
    return (
        'def score_pack(plain, item, feasible, container_size):\n'
        '    heights = plain.astype(np.float32)\n'
        '    max_h = max(1.0, float(np.max(heights)))\n'
        '    score = (1.0 - heights / max_h) * feasible\n'
        '    edge = np.zeros_like(score)\n'
        '    edge[0,:] = 1\n'
        '    edge[:,0] = 1\n'
        '    edge[-1,:] = 1\n'
        '    edge[:,-1] = 1\n'
        '    score = 0.7 * score + 0.3 * edge\n'
        '    return score * feasible\n'
    )


def _default_unpack_program_code():
    return (
        'def score_unpack(voxel, item, feasible, container_size):\n'
        '    w,l,h = voxel.shape\n'
        '    occ = (voxel == 2)\n'
        '    score = np.zeros((w,l), dtype=np.float32)\n'
        '    for i in range(w):\n'
        '        for j in range(l):\n'
        '            if feasible[i,j] == 0:\n'
        '                continue\n'
        '            col = np.where(occ[i,j])[0]\n'
        '            if len(col) == 0:\n'
        '                continue\n'
        '            top = int(col[-1])\n'
        '            score[i,j] = (h - 1 - top)\n'
        '    return score * feasible\n'
    )


class OnlineFunSearchManager(object):
    def __init__(self, save_dir, container_size, llm_enabled=False, llm_model='gpt-4o-mini', topk=8):
        self.save_dir = save_dir
        self.container_size = container_size
        self.topk = topk
        self.db_path = os.path.join(save_dir, 'funsearch_db.json')
        self.llm = LLMHeuristicClient(model=llm_model, enabled=llm_enabled)
        self.entries = []
        self._load_or_init()
        self._activate_best()

    def _load_or_init(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []
        if len(self.entries) == 0:
            self.entries.append(self._make_entry('pack', _default_pack_program_code(), 'seed', None, 0.0))
            self.entries.append(self._make_entry('unpack', _default_unpack_program_code(), 'seed', None, 0.0))
            self._save()

    def _save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, ensure_ascii=True, indent=2)

    def _make_entry(self, kind, code, source, parents, score):
        return {
            'id': str(uuid.uuid4()),
            'kind': kind,
            'code': code,
            'source': source,
            'parents': parents or [],
            'score': float(score),
            'created_at': int(time.time()),
        }

    def _ranked(self, kind):
        vals = [e for e in self.entries if e['kind'] == kind]
        vals.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return vals

    def _pick_parents(self, kind):
        ranked = self._ranked(kind)
        if len(ranked) == 0:
            return []
        top = ranked[:self.topk]
        if len(top) == 1:
            return [top[0], top[0]]
        return random.sample(top, k=2)

    def _mutate_code(self, kind, parent_code):
        coeffs = [round(random.uniform(0.1, 0.9), 2) for _ in range(3)]
        if kind == 'pack':
            return (
                'def score_pack(plain, item, feasible, container_size):\n'
                '    heights = plain.astype(np.float32)\n'
                '    max_h = max(1.0, float(np.max(heights)))\n'
                '    low = (1.0 - heights / max_h)\n'
                '    x,y,_ = item\n'
                '    support = np.zeros_like(low)\n'
                '    w,l = low.shape\n'
                '    for i in range(w):\n'
                '        for j in range(l):\n'
                '            i2 = min(w, i + x)\n'
                '            j2 = min(l, j + y)\n'
                '            patch = heights[i:i2, j:j2]\n'
                '            if patch.size > 0:\n'
                '                support[i,j] = float(np.mean(patch == np.max(patch)))\n'
                '    edge = np.zeros_like(low)\n'
                '    edge[0,:]=1; edge[:,0]=1; edge[-1,:]=1; edge[:,-1]=1\n'
                f'    score = {coeffs[0]}*low + {coeffs[1]}*support + {coeffs[2]}*edge\n'
                '    return score * feasible\n'
            )
        return (
            'def score_unpack(voxel, item, feasible, container_size):\n'
            '    w,l,h = voxel.shape\n'
            '    occ = (voxel == 2)\n'
            '    score = np.zeros((w,l), dtype=np.float32)\n'
            '    cur_vol = max(1, int(item[0]*item[1]*item[2]))\n'
            '    for i in range(w):\n'
            '        for j in range(l):\n'
            '            if feasible[i,j] == 0:\n'
            '                continue\n'
            '            col = np.where(occ[i,j])[0]\n'
            '            if len(col) == 0:\n'
            '                continue\n'
            '            top = int(col[-1])\n'
            '            free_above = h - 1 - top\n'
            '            cavity = 0.0\n'
            '            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:\n'
            '                nx,ny = i+dx,j+dy\n'
            '                if nx < 0 or ny < 0 or nx >= w or ny >= l:\n'
            '                    continue\n'
            '                ncol = np.where(occ[nx,ny])[0]\n'
            '                ntop = -1 if len(ncol) == 0 else int(ncol[-1])\n'
            '                cavity += max(0, ntop - top)\n'
            f'            score[i,j] = {coeffs[0]}*free_above + {coeffs[1]}*cavity - {coeffs[2]}*(1.0/cur_vol)\n'
            '    return score * feasible\n'
        )

    def _llm_prompt(self, kind, parent_a, parent_b):
        fn_name = 'score_pack' if kind == 'pack' else 'score_unpack'
        sig = (
            'score_pack(plain, item, feasible, container_size)'
            if kind == 'pack' else
            'score_unpack(voxel, item, feasible, container_size)'
        )
        return (
            'Create a new heuristic function for online 3D bin packing with unpacking synergy.\n'
            f'Output only Python code for function {sig}.\n'
            'Rules:\n'
            '- Must return a float score map with same shape as feasible.\n'
            '- Respect feasibility by multiplying result with feasible.\n'
            '- Balance physical stability and space utilization.\n'
            '- No imports, use only numpy as np.\n\n'
            'Parent A:\n' + parent_a + '\n\nParent B:\n' + parent_b
        )

    def _generate_candidate(self, kind, parent_a, parent_b):
        if self.llm.available():
            code = self.llm.generate(self._llm_prompt(kind, parent_a['code'], parent_b['code']))
            if code:
                fn = _safe_compile(code, 'score_pack' if kind == 'pack' else 'score_unpack')
                if fn is not None:
                    return code, 'llm'
        code = self._mutate_code(kind, parent_a['code'])
        return code, 'mutation'

    def _pack_objective(self, score, feasible, plain):
        if feasible.sum() == 0:
            return 0.0
        masked = score * feasible
        v = masked[feasible == 1]
        if v.size == 0:
            return 0.0
        thr = np.percentile(v, 60)
        chosen = (masked >= thr).astype(np.int32) * feasible
        if chosen.sum() == 0:
            return 0.0
        mean_h = float(np.mean(plain[chosen == 1]))
        spread = float(np.std(plain[chosen == 1]))
        return -mean_h - 0.1 * spread + 0.2 * float(chosen.sum()) / float(feasible.sum())

    def _unpack_objective(self, score, feasible):
        if feasible.sum() == 0:
            return 0.0
        masked = score * feasible
        v = masked[feasible == 1]
        if v.size == 0:
            return 0.0
        return float(np.mean(v))

    def _evaluate_program(self, kind, code, observations):
        fn_name = 'score_pack' if kind == 'pack' else 'score_unpack'
        fn = _safe_compile(code, fn_name)
        if fn is None:
            return -1e9

        vals = []
        for obs in observations:
            voxel, _, item = parse_pusnet_observation(obs, self.container_size)
            if kind == 'pack':
                feasible, plain = pack_feasibility_map(voxel, item, self.container_size)
                if feasible.sum() == 0:
                    continue
                try:
                    score = fn(plain, item, feasible, self.container_size)
                except Exception:
                    return -1e9
                vals.append(self._pack_objective(np.asarray(score), feasible, plain))
            else:
                feasible = unpack_feasibility_map(voxel)
                if feasible.sum() == 0:
                    continue
                try:
                    score = fn(voxel, item, feasible, self.container_size)
                except Exception:
                    return -1e9
                vals.append(self._unpack_objective(np.asarray(score), feasible))

        if len(vals) == 0:
            return -1e6
        return float(np.mean(vals))

    def iterate(self, observations, candidates=4):
        if observations is None or len(observations) == 0:
            return {'updated': False, 'reason': 'no-observation'}

        updates = []
        for kind in ['pack', 'unpack']:
            for _ in range(max(1, candidates)):
                pa, pb = self._pick_parents(kind)
                code, source = self._generate_candidate(kind, pa, pb)
                score = self._evaluate_program(kind, code, observations)
                entry = self._make_entry(kind, code, source, [pa['id'], pb['id']], score)
                self.entries.append(entry)
                updates.append((kind, source, score))

        self._save()
        self._activate_best()
        return {'updated': True, 'details': updates}

    def _activate_best(self):
        best_pack = self._ranked('pack')[0]
        best_unpack = self._ranked('unpack')[0]
        pack_fn = _safe_compile(best_pack['code'], 'score_pack')
        unpack_fn = _safe_compile(best_unpack['code'], 'score_unpack')
        set_runtime_heuristics(pack_fn=pack_fn, unpack_fn=unpack_fn)

    def best_scores(self):
        bp = self._ranked('pack')[0]['score']
        bu = self._ranked('unpack')[0]['score']
        return {'pack': float(bp), 'unpack': float(bu)}
