import numpy as np
import torch
import gym
import copy
from acktr.model import Policy
from acktr.utils import get_rotation_mask, get_possible_position


class nnModel(object):
    def __init__(self, url, args):
        area = args.container_size[0]*args.container_size[1]
        self.alen = area * (1+args.enable_rotation)
        self.olen = args.channel * area
        self.height = args.container_size[2]
        self.container_size = args.container_size
        self.enable_rotation = args.enable_rotation
        self.device = torch.device(args.device)
        self._model = self._load_model(url, args)


    def _load_model(self, url, args):
        model_pretrained, ob_rms = torch.load(url, map_location=self.device)
        observation_space = gym.spaces.Box(low=0.0, high=self.height, shape=(self.olen, ))
        action_space = gym.spaces.Discrete(self.alen)
        actor_critic = Policy(obs_shape=observation_space.shape, action_space=action_space, base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        # print(actor_critic)

        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
        load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}

        model_state = actor_critic.state_dict()
        for k, v in load_dict.items():
            if k in model_state:
                target = model_state[k]
                if v.dim() == 0 and target.dim() == 1 and target.numel() == 1:
                    load_dict[k] = v.view(1)
                elif v.shape != target.shape and v.numel() == target.numel():
                    load_dict[k] = v.view_as(target)

        actor_critic.load_state_dict(load_dict, strict=False)
        actor_critic = actor_critic.to(self.device)
        return actor_critic

    def evaluate(self, obs, use_mask=True):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).to(self.device)

        value, logits, _, _ = self._model.base(x, 0, 0)
        poss = self._model.dist.get_policy_distribution(logits)
        if use_mask:
            if self.enable_rotation:
                pred = get_rotation_mask(torch.tensor(obs), self.container_size)
            else:
                pred = np.array(get_possible_position(torch.tensor(obs), self.container_size))

        value = float(value)
        poss = poss.cpu().detach().numpy()
        if use_mask:
            pred = np.array(pred).reshape((-1,))

        # np.set_printoptions(precision=3, suppress=True)
        # print('---------------------------')
        # print(pred1.reshape(10,10))
        # print(pred2.reshape(10,10))

        if use_mask:
            masked_poss = np.array(poss).reshape((-1,))
            masked_poss[pred <= 0] = -1e9
            probs = np.exp(masked_poss - np.max(masked_poss))
            probs_sum = np.sum(probs)
            poss_in_actions = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
        else:
            raw_poss = np.array(poss).reshape((-1,))
            probs = np.exp(raw_poss - np.max(raw_poss))
            poss_in_actions = probs / np.sum(probs)
        poss_in_actions = np.reshape(poss_in_actions, newshape=(-1,))
        return value, poss_in_actions

    def sample_action(self, obs):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).to(self.device)

        value, logits, _, _ = self._model.base(x, 0, 0)
        poss = self._model.dist.get_policy_distribution(logits)
        if self.enable_rotation:
            pred = get_rotation_mask(torch.tensor(obs), self.container_size)
        else:
            pred = np.array(get_possible_position(torch.tensor(obs), self.container_size))

        value = float(value)
        pred = torch.tensor(np.array(pred).reshape((-1,)), device=poss.device, dtype=poss.dtype)
        masked_logits = poss.clone().reshape((-1,))
        masked_logits[pred <= 0] = -1e9
        cat = torch.distributions.Categorical(logits=masked_logits)
        action = int(cat.sample())

        return value, action


