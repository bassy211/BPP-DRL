import numpy as np

def run_heuristic_sequence(heuristic_type, raw_env, preview_num):
    import copy
    from time import perf_counter
    env = raw_env.clone_for_search() if hasattr(raw_env, 'clone_for_search') else copy.deepcopy(raw_env)
    
    start = perf_counter()
    while True:
        box_list = env.box_creator.preview(preview_num)
        next_box = box_list[0]
        
        # Valid placements
        valid_mask = env.get_possible_position(env.space.plain)
        
        valid_indices = []
        if env.can_rotate:
            valid_mask_rot = env.get_possible_position(env.space.plain) # Need to handle rotation properly
            # Wait, bin3D's get_possible_position uses env.next_box. We need to set next box or rotation.
            # It's better to use get_possible_position from utils?
        
        # Actually, let's look at `acktr.utils.get_possible_position`
