import numpy as np
import torch

from acktr import utils
from acktr.envs import make_vec_envs
from acktr.utils import get_possible_position, get_rotation_mask, get_pusnet_action_mask


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, args=None):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        location_masks = []
        for observation in obs:
            if args is not None and args.use_pusnet:
                box_mask = get_pusnet_action_mask(observation, args.container_size, use_modulation=args.use_action_modulation)
            elif args is not None and args.enable_rotation:
                box_mask = get_rotation_mask(observation, args.container_size)
            else:
                container = args.container_size if args is not None else [10, 10, 10]
                box_mask = get_possible_position(observation, container)
            location_masks.append(box_mask)
        location_masks = torch.FloatTensor(location_masks).to(device)

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                location_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
