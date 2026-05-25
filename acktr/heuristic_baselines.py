import numpy as np
import copy
from acktr.utils import get_rotation_mask, get_possible_position, check_box

class HeuristicModel:
    def __init__(self, heuristic_type, args):
        self.heuristic_type = heuristic_type
        self.enable_rotation = getattr(args, 'enable_rotation', False)
        self.container_size = tuple(args.container_size)
    
    def evaluate(self, obs, use_mask=True):
        if not isinstance(obs, np.ndarray):
            obs = obs.cpu().numpy()
        
        # Get action mask
        if self.enable_rotation:
            action_mask = np.asarray(get_rotation_mask(obs, self.container_size), dtype=np.float32)
        else:
            action_mask = np.asarray(get_possible_position(obs, self.container_size), dtype=np.float32)
            
        if np.sum(action_mask) <= 0:
            action_mask = np.ones_like(action_mask)
            
        action_mask = action_mask.reshape(-1)
        valid_indices = np.where(action_mask > 0)[0]
        
        if len(valid_indices) == 0:
            probs = np.ones_like(action_mask) / len(action_mask)
            return 0.0, probs
            
        probs = np.zeros_like(action_mask)
        
        if self.heuristic_type == 'random':
            probs[valid_indices] = 1.0 / len(valid_indices)
            return 0.0, probs
        elif self.heuristic_type == 'first_fit':
            probs[valid_indices[0]] = 1.0
            return 0.0, probs
        else:
            # For complex heuristics, we need to evaluate the placement
            box_info = obs.reshape((4, -1))
            x, y, z = int(box_info[1][0]), int(box_info[2][0]), int(box_info[3][0])
            plain = box_info[0].reshape((self.container_size[0], self.container_size[1]))
            area = self.container_size[0] * self.container_size[1]
            
            best_score = -float('inf')
            best_idx = valid_indices[0]
            
            for idx in valid_indices:
                rot = idx >= area
                pos_idx = idx - area if rot else idx
                lx, ly = pos_idx // self.container_size[1], pos_idx % self.container_size[1]
                
                bx, by = (y, x) if rot else (x, y)
                new_h = check_box(plain, bx, by, lx, ly, z, self.container_size)
                
                score = 0
                if self.heuristic_type == 'best_fit':
                    # Minimize final height, so negative
                    score = -new_h
                elif self.heuristic_type in ['corner_point', 'extreme_point', 'ems']:
                    # Simple proxies for these concepts on heightmap
                    touch_walls = 0
                    if lx == 0 or lx + bx == self.container_size[0]: touch_walls += 1
                    if ly == 0 or ly + by == self.container_size[1]: touch_walls += 1
                    if new_h == 0: touch_walls += 1
                    
                    if self.heuristic_type == 'corner_point':
                        # Favor touching corners
                        score = touch_walls - new_h * 0.1
                    elif self.heuristic_type == 'extreme_point':
                        # Extreme point proxy: prioritize lowest Z, then lowest Y, then lowest X
                        # This tightly packs items to the origin corner (0,0,0) bounds
                        score = -(new_h * 10000 + ly * 100 + lx)
                    elif self.heuristic_type == 'ems':
                        # EMS / Maximized Contact Area (MCA) heuristic for 3D BPP
                        # Priority 1: Keep placement height as low as possible
                        # Priority 2: Maximize 3D surface contact area with items and walls
                        # Priority 3: Pack closely to corner
                        z_top = new_h + z
                        bottom_contact = np.sum(plain[lx:lx+bx, ly:ly+by] == new_h)
                        contact = bottom_contact
                        
                        if lx == 0: contact += by * z
                        else: contact += np.sum(np.clip(plain[lx-1, ly:ly+by], new_h, z_top) - new_h)
                        
                        if lx + bx >= self.container_size[0]: contact += by * z
                        else: contact += np.sum(np.clip(plain[lx+bx, ly:ly+by], new_h, z_top) - new_h)
                        
                        if ly == 0: contact += bx * z
                        else: contact += np.sum(np.clip(plain[lx:lx+bx, ly-1], new_h, z_top) - new_h)
                        
                        if ly + by >= self.container_size[1]: contact += bx * z
                        else: contact += np.sum(np.clip(plain[lx:lx+bx, ly+by], new_h, z_top) - new_h)
                        
                        score = -(new_h * 100000) + contact * 100 - (lx + ly)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            probs[best_idx] = 1.0
            return 0.0, probs

    def sample_action(self, obs):
        _, probs = self.evaluate(obs)
        action = np.argmax(probs)
        return 0.0, int(action)
