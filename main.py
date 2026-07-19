import sys
import os
import time
from collections import deque
import numpy as np
import torch
from shutil import copyfile
from acktr import algo, utils
from acktr.utils import get_possible_position, get_rotation_mask, get_pusnet_action_mask
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from acktr.funsearch import OnlineFunSearchManager
from evaluation import evaluate
from tensorboardX import SummaryWriter
from unified_test import unified_test
from gym.envs.registration import register

def main(args):
    # input arguments about environment
    if args.test:
        test_model(args)
    else:
        train_model(args)

def test_model(args):
    assert args.test is True
    if args.algorithm in ['random', 'first_fit', 'best_fit', 'corner_point', 'extreme_point', 'ems', 'macs', 'layer_building', 'f53']:
        model_url = None
    else:
        model_url = args.load_dir + args.load_name
    unified_test(model_url, args)

def train_model(args):
    if args.algorithm in ['random', 'first_fit', 'best_fit', 'corner_point', 'extreme_point', 'ems', 'macs', 'layer_building', 'f53']:
        print(f"Skipping PPO/A2C/ACKTR training since selected algorithm is a heuristic baseline: {args.algorithm}.")
        # Use unified_test to evaluate heuristic directly
        unified_test(None, args)
        return

    custom = input('please input the test name: ')
    time_now = time.strftime('%Y.%m.%d-%H-%M', time.localtime(time.time()))

    env_name = args.env_name
    if args.device != 'cpu':
        torch.cuda.set_device(torch.device(args.device))
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    save_path = args.save_dir
    load_path = args.load_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    data_path = os.path.join(save_path, custom)
    try:
        os.makedirs(data_path)
    except OSError:
        pass

    log_dir = './log'  # directory to save agent logs (default: ./log)
    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    envs = make_vec_envs(env_name, args.seed, args.num_processes, args.gamma, log_dir, device, False, args = args)

    if args.pretrain:
        model_pretrained, ob_rms = torch.load(os.path.join(load_path, args.load_name))
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
        load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
        for k, v in load_dict.items():
            if len(v.size()) <= 3:
                load_dict[k] = v.squeeze(dim=-1)
        actor_critic.load_state_dict(load_dict)
        setattr(utils.get_vec_normalize(envs), 'ob_rms', ob_rms)
    else:
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size,'args': args})
    print(actor_critic)
    print("Rotation:", args.enable_rotation)
    actor_critic.to(device)

    # leave a backup for parameter tuning
    copyfile('main.py', os.path.join(data_path, 'main.py'))
    copyfile('./acktr/envs.py', os.path.join(data_path, 'envs.py'))
    copyfile('./acktr/distributions.py', os.path.join(data_path, 'distributions.py'))
    copyfile('./acktr/storage.py', os.path.join(data_path, 'storage.py'))
    copyfile('./acktr/model.py', os.path.join(data_path, 'model.py'))
    copyfile('./acktr/algo/acktr_pipeline.py', os.path.join(data_path, 'acktr_pipeline.py'))

    if args.algorithm == 'a2c':
        agent = algo.ACKTR(actor_critic,
                       args.value_loss_coef,
                       args.entropy_coef,
                       args.invalid_coef,
                       args.lr,
                       args.eps,
                       args.alpha,
                       max_grad_norm = 0.5
                           )
    elif args.algorithm == 'acktr':
        agent = algo.ACKTR(actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.invalid_coef,
            acktr=True,
            args=args)

    rollouts = RolloutStorage(args.num_steps,  # forward steps
                              args.num_processes,  # agent processes
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.pallet_size,
                              use_pusnet=args.use_pusnet)

    obs = envs.reset()
    location_masks = []
    for observation in obs:
        if args.use_pusnet:
            box_mask = get_pusnet_action_mask(observation, args.container_size, use_modulation=args.use_action_modulation)
        elif not args.enable_rotation:
            box_mask = get_possible_position(observation, args.container_size)
        else:
            box_mask = get_rotation_mask(observation, args.container_size)
        location_masks.append(box_mask)
    location_masks = torch.as_tensor(np.asarray(location_masks, dtype=np.float32), device=device)

    rollouts.obs[0].copy_(obs)
    rollouts.location_masks[0].copy_(location_masks)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_ratio = deque(maxlen=10)

    start = time.time()

    tbx_dir = './runs'
    if not os.path.exists('{}/{}/{}'.format(tbx_dir, env_name, custom)):
        os.makedirs('{}/{}/{}'.format(tbx_dir, env_name, custom))
    if args.tensorboard:
        writer = SummaryWriter(logdir='{}/{}/{}'.format(tbx_dir, env_name, custom))

    funsearch_manager = None
    funsearch_obs_buffer = []
    if args.use_pusnet and args.enable_online_funsearch:
        funsearch_dir = os.path.join(data_path, 'funsearch')
        funsearch_manager = OnlineFunSearchManager(
            save_dir=funsearch_dir,
            container_size=args.container_size,
            llm_enabled=args.llm_enable,
            llm_model=args.llm_model,
            topk=args.funsearch_topk,
        )

    index = 0
    for j in range(1, args.train_epochs + 1):
        iter_t_act = 0.0
        iter_t_env = 0.0
        iter_t_mask = 0.0
        iter_t_update = 0.0
        for step in range(args.num_steps):
            # Sample actions
            t_act0 = time.perf_counter()
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], location_masks)
            iter_t_act += (time.perf_counter() - t_act0)

            location_masks = []
            t_env0 = time.perf_counter()
            obs, reward, done, infos = envs.step(action)
            iter_t_env += (time.perf_counter() - t_env0)

            if funsearch_manager is not None:
                obs_np = obs.detach().cpu().numpy()
                for ob in obs_np:
                    funsearch_obs_buffer.append(ob.copy())
                if len(funsearch_obs_buffer) > args.funsearch_sample_budget:
                    funsearch_obs_buffer = funsearch_obs_buffer[-args.funsearch_sample_budget:]

            for i in range(len(infos)):
                if 'episode' in infos[i].keys():
                    episode_rewards.append(infos[i]['episode']['r'])
                    episode_ratio.append(infos[i]['ratio'])
            t_mask0 = time.perf_counter()
            for observation in obs:
                if args.use_pusnet:
                    box_mask = get_pusnet_action_mask(observation, args.container_size, use_modulation=args.use_action_modulation)
                elif not args.enable_rotation:
                    box_mask = get_possible_position(observation, args.container_size)
                else:
                    box_mask = get_rotation_mask(observation, args.container_size)
                location_masks.append(box_mask)
            location_masks = torch.as_tensor(np.asarray(location_masks, dtype=np.float32), device=device)
            iter_t_mask += (time.perf_counter() - t_mask0)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks, location_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
        # value_loss, action_loss, dist_entropy, prob_loss = agent.update(rollouts)
        t_upd0 = time.perf_counter()
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)
        iter_t_update += (time.perf_counter() - t_upd0)

        rollouts.after_update()
        if args.save_model:
            if (j % args.save_interval == 0) and args.save_dir != "":
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(data_path, env_name + time_now + ".pt"))

        if funsearch_manager is not None and j % args.funsearch_interval == 0:
            evolve_result = funsearch_manager.iterate(
                observations=funsearch_obs_buffer,
                candidates=args.funsearch_candidates,
            )
            best_scores = funsearch_manager.best_scores()
            print('FunSearch update:', evolve_result)
            print('FunSearch best scores:', best_scores)
            if args.tensorboard:
                writer.add_scalar('FunSearch/best_pack_score', best_scores['pack'], j)
                writer.add_scalar('FunSearch/best_unpack_score', best_scores['unpack'], j)

        # print useful information of training
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = j * args.num_processes * args.num_steps
            end = time.time()
            index += 1
            print(
                "The algorithm is {}, the recurrent policy is {}\nThe env is {}, the version is {}".format(
                    args.algorithm, False, env_name, custom))
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                "The dist entropy {:.5f}, The value loss {:.5f}, the action loss {:.5f}\n"
                "The mean space ratio is {}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss, np.mean(episode_ratio)))
            if args.use_pusnet:
                print('Active branch: {}, branch samples: {}'.format(
                    getattr(agent, 'last_active_branch', 'na'),
                    getattr(agent, 'last_branch_samples', -1)))
            if args.profile_train_speed:
                iter_total = iter_t_act + iter_t_env + iter_t_mask + iter_t_update
                denom = max(iter_total, 1e-8)
                print('Timing breakdown (ms/update): act={:.2f} ({:.1f}%), env={:.2f} ({:.1f}%), mask={:.2f} ({:.1f}%), update={:.2f} ({:.1f}%)'.format(
                    iter_t_act * 1000.0, iter_t_act * 100.0 / denom,
                    iter_t_env * 1000.0, iter_t_env * 100.0 / denom,
                    iter_t_mask * 1000.0, iter_t_mask * 100.0 / denom,
                    iter_t_update * 1000.0, iter_t_update * 100.0 / denom,
                ))

            if args.tensorboard:
                writer.add_scalar('The average rewards', np.mean(episode_rewards), j)
                writer.add_scalar("The mean ratio", np.mean(episode_ratio), j)
                writer.add_scalar('Distribution entropy', dist_entropy, j)
                writer.add_scalar("The value loss", value_loss, j)
                writer.add_scalar("The action loss", action_loss, j)
                writer.add_scalar('Probability loss', prob_loss, j)
                writer.add_scalar("Mask loss", graph_loss, j) # add mask loss
                if args.use_pusnet:
                    branch_id = 0 if getattr(agent, 'last_active_branch', 'pack') == 'pack' else 1
                    writer.add_scalar('Branch/active', branch_id, j)
                    writer.add_scalar('Branch/samples', getattr(agent, 'last_branch_samples', 0), j)


def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )


if __name__ == "__main__":
    registration_envs()
    args = get_args()
    main(args)

