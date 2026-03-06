import numpy as np
import copy
import math

class Node(object):
    def __init__(self, parent, number, height):
        self.max_value = None
        self.action = None  # 记录最好的值和对应的动作
        self.number = number
        self.parent = parent

        self.height = height
        assert self.height is not None
        self.children = []
        self.dis_num = height

        self.visit = 0
        if self.height != -1:
            self.max_v = math.factorial(self.height)
        else:
            self.max_v = 1

    def get_q_value(self): # 直接继承父节点的q值吗？
        if self.max_value is None:
            return self.parent.get_q_value()
        if self.visit >= self.max_v:
            return -1000
        if self.dis_num <= 0:
            return -10000
        return self.max_value

    def get_u_value(self): # 没有用到policy,直接就展开了
        c = 0.5
        return c * np.sqrt(self.parent.visit) / (self.visit + 1)

    def get_value(self):
        return self.get_q_value() + self.get_u_value()

    def disable(self):
        self.dis_num = 0
        if self.parent is None:
            return
        self.parent.dis_num -= 1
        if self.parent.dis_num == 0:
            self.parent.disable()

    def update(self, value, action):
        self.visit += 1
        if self.max_value is None or value > self.max_value:
            self.max_value = value
            self.action = action
        if self.parent is not None:
            self.parent.update(value, action)

# FOR ENV 'MASK'
class ReorderTree(object):
    def __init__(self, nmodel, box_list, env, encode=True, p_bound=0.8, v_bound=0.1, times=100):
        self.encode = encode
        # the box number used for reordering
        self.box_num = len(box_list)
        # the network of single step
        self.nmodel = nmodel
        # the shape of the action mask
        self.mask_shape = env.bin_size[:2]
        self.mask_len = self.mask_shape[0] * self.mask_shape[1]
        # keep env reference; copy only when simulation is really needed
        self.env = env
        self.box_list = list(box_list)
        # threshold
        self.p_bound = p_bound
        self.v_bound = v_bound
        self.pos_num = max(1, int(1 / self.p_bound))
        self.times = min(times, math.factorial(self.box_num - 1))
        self.eval_cache = {}

    def get_order_mask(self, smask, box_size):
        emask = np.array(smask, copy=True).reshape(self.mask_shape)
        smask_2d = np.asarray(smask).reshape(self.mask_shape)
        ex = emask.shape[0] - box_size[0] + 1
        ey = emask.shape[1] - box_size[1] + 1
        for i in range(ex):
            for j in range(ey):
                if emask[i][j] == 1:
                    if smask_2d[i:i + box_size[0], j:j + box_size[1]].min() == 0:
                        emask[i][j] = 0
        return emask.reshape(-1)

    def get_mixed_mask(self, masks, real_idx, box_size, raw_mask):
        action_mask = np.asarray(raw_mask, dtype=np.int32).copy()
        if real_idx + 1 < self.box_num:
            stacked_mask = np.all(masks[real_idx + 1:] == 1, axis=0).astype(np.int32)
        else:
            stacked_mask = np.ones_like(masks[0], dtype=np.int32)
        order_mask = self.get_order_mask(stacked_mask, box_size)
        mixed_mask = (action_mask == 1) & (order_mask == 1)
        mixed_mask = mixed_mask.astype(np.int32)
        return mixed_mask

    def get_mixed_obs(self, masks, real_idx, raw_obs):
        max_height = self.env.bin_size[-1]
        new_obs = np.asarray(raw_obs).reshape(-1).copy()
        if real_idx + 1 < self.box_num:
            stacked_mask = np.all(masks[real_idx + 1:] == 1, axis=0).astype(new_obs.dtype)
            new_obs = max_height * (1 - stacked_mask) + new_obs * stacked_mask
        return new_obs

    def update_mask(self, mask, box, pos):
        pos = (pos // self.mask_shape[1], pos % self.mask_shape[1])
        cmask = np.asarray(mask).reshape(self.mask_shape).copy()
        cmask[pos[0]:pos[0] + box[0], pos[1]:pos[1] + box[1]] = 0
        return cmask.reshape(-1)

    def will_terminate(self,  mixed_obs):
        max_height = self.env.bin_size[-1]
        # rsum = np.sum(raw_mask)
        ssum = np.sum(mixed_obs)
        # print(rsum, ssum)
        return ssum == self.mask_len * max_height
        # return rsum > 0.8 * self.mask_len or ssum == self.mask_len * max_height

    def evaluate(self, obs, masks, real_idx):
        # 4 channels
        revised_obs = np.asarray(obs).reshape(4, -1).copy()
        raw_obs = revised_obs[0]
        new_obs = self.get_mixed_obs(masks, real_idx, raw_obs)
        revised_obs[0] = new_obs
        revised_obs = revised_obs.reshape((-1,))

        cache_key = (real_idx, revised_obs.tobytes())
        cached = self.eval_cache.get(cache_key, None)
        if cached is None:
            val, poss = self.nmodel.evaluate(revised_obs)
            poss = np.asarray(poss).reshape(-1)
            self.eval_cache[cache_key] = (float(val), poss)
        else:
            val, poss = cached

        k = min(self.pos_num, poss.shape[0])
        topk_idx = np.argpartition(poss, -k)[-k:]
        topk_sorted = topk_idx[np.argsort(poss[topk_idx])]
        pos_candidates = list(topk_sorted)
        wt = self.will_terminate(new_obs)
        return val, pos_candidates, wt

    def search(self, masks, cur_env, res_idxs, cur_node, cur_value, action):
        assert cur_node is not None
        # print('DISNUM: ', cur_node.dis_num)
        next_eval = -1e18
        next_node = None
        # print(res_idxs)

        if len(cur_node.children) == 0:
            for idx in res_idxs:
                if idx == self.box_num - 1 and len(res_idxs) > 1:
                    continue
                cur_node.children.append(Node(parent=cur_node, number=idx, height=cur_node.height - 1))

        # find next node with max evaluation
        for node in cur_node.children:
            node_eval = node.get_value()
            assert node.dis_num >= -1
            # print(node.number, node_eval, node.dis_num)
            if node_eval > next_eval:
                next_eval = node_eval
                next_node = node

        # using single-step model to evaluate future
        if next_node is None:
            cur_node.update(-10000, None)
            return
        idx = next_node.number
        cur_box = self.box_list[idx]
        cur_env.box_creator.box_list = [cur_box, self.env.bin_size]
        cur_obs = cur_env.cur_observation

        # print(cur_obs[0:100].reshape(10,10))
        # print(idx, cur_box)

        val, pos_candidates, will_terminate = self.evaluate(cur_obs, masks, idx)
        pos = pos_candidates[-1]
        assert len(pos_candidates) == 1

        # will_terminate = False
        # # if may_terminate:
        # tmp_env = copy.deepcopy(cur_env)
        # next_obs, reward, done, _ = tmp_env.step([pos])
        # if done:
        #     will_terminate = True

        next_obs, reward, done, _ = cur_env.step([pos])

        if done or will_terminate: 
            fail_flag = False
            for i in range(self.box_num):
                if (i in res_idxs and i < idx) or (i not in res_idxs and i >= idx):
                    fail_flag = True
                    break  
            if fail_flag:
                next_node.disable()
                cur_node.update(-10000, None)
                return
            if idx == 0:
                cur_node.update(cur_value + 0, 0)
                return
            else:
                cur_node.update(cur_value + 0, action)
                return

        # reach the evaluation point
        if len(res_idxs) == 1:
            assert idx == self.box_num - 1
            if idx == 0:
                cur_node.update(cur_value + val, pos)
                return
            else:
                cur_node.update(cur_value + val, action)
                return

        # copy and update [res_idxs]
        next_idxs = [i for i in res_idxs if i != idx]

        # In-place mask update + rollback to avoid full copy in recursion
        old_mask = masks[idx].copy()
        masks[idx] = self.update_mask(masks[idx], cur_box, pos)
        # next value
        next_value = cur_value + reward
        # recursion
        if action is None and idx == 0:
            action = pos
        self.search(masks, cur_env, next_idxs, next_node, next_value, action)
        masks[idx] = old_mask

    def get_baseline(self):
        env = copy.deepcopy(self.env)
        obs = env.cur_observation
        nor_exp = 0
        nor_act = None
        for i in range(self.box_num):
            val, poss = self.nmodel.evaluate(obs)
            act = np.argmax(poss)
            if i == 0:
                nor_act = act
            obs, reward, done, info = env.step([act])
            if done:
                nor_exp += 0
                return nor_exp, nor_act
            if i == self.box_num - 1:
                nor_exp += val
                return nor_exp, nor_act
            nor_exp += reward

    def reorder_search(self):
        # Fast path: when only one simulation is requested, tree rollout + baseline
        # overhead dominates. Use single-step greedy directly.
        if self.times <= 1 or self.box_num <= 1:
            val, poss = self.nmodel.evaluate(self.env.cur_observation)
            act = int(np.argmax(poss))
            return act, float(val), True

        self.eval_cache.clear()
        nor_exp, nor_act = self.get_baseline()
        root = Node(None, None, self.box_num - 1)
        root.max_value = nor_exp 
        root.action = nor_act
        best_value = nor_exp
        no_improve_rounds = 0
        patience = 4
        for i in range(self.times):
            sim_env = copy.deepcopy(self.env)
            res_idxs = list(range(self.box_num))
            masks = np.ones((self.box_num, self.mask_len), dtype=np.int8)
            self.search(masks, sim_env, res_idxs, root, 0, None)
            if root.max_value is not None and root.max_value > best_value + 1e-8:
                best_value = root.max_value
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= patience and root.action is not None:
                break
        max_exp = root.max_value
        max_act = root.action
        if max_act != nor_act and max_exp - nor_exp < self.v_bound:
            # print('conservative!')
            max_exp = nor_exp
            max_act = nor_act
        default = (max_act == nor_act)
        return max_act, max_exp, default
