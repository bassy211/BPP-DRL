import numpy as np
import torch
import gym
import copy
from acktr.model import Policy
from acktr.utils import get_rotation_mask, get_possible_position

def normalize_state_dict(state_dict):
    """将 checkpoint 中的参数名归一化为当前 Policy 模型的命名风格：
    去除 module./add_bias. 前缀、把 _bias 改成 bias，并压缩多余维度。"""
    load_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    for k, v in load_dict.items():
        if len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    return load_dict
class nnModel(object):
    def __init__(self, url, args):
        area = args.pallet_size * args.pallet_size
        self.use_pusnet = bool(getattr(args, 'use_pusnet', False))
        self.enable_rotation = bool(getattr(args, 'enable_rotation', False))
        self.container_size = tuple(args.container_size)
        self.alen = area * 2 if self.use_pusnet else area * (1+args.enable_rotation)
        self.olen = area * args.container_size[2] + 3 * area if self.use_pusnet else args.channel * area
        self.height = args.container_size[2]
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

        for k, v in load_dict.items():
            if len(v.size()) <= 3:
                load_dict[k] = v.squeeze(dim=-1)

        actor_critic.load_state_dict(load_dict)
        actor_critic = actor_critic.to(self.device)
        return actor_critic

    def evaluate(self, obs, use_mask=True):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        if self.use_pusnet:
            vp, vu, pack_logits, unpack_logits, _ = self._model.base(x, 0, 0)
            choose_pack = (vp >= vu).float()
            choose_unpack = 1.0 - choose_pack
            pack_gate = torch.where(choose_pack > 0, torch.zeros_like(pack_logits), torch.full_like(pack_logits, -1e8))
            unpack_gate = torch.where(choose_unpack > 0, torch.zeros_like(unpack_logits), torch.full_like(unpack_logits, -1e8))
            poss = torch.cat((pack_logits + pack_gate, unpack_logits + unpack_gate), dim=-1)
            pred = torch.ones_like(poss)
            value = torch.maximum(vp, vu)
        else:
            value, logits, _, pred = self._model.base(x, 0, 0)
            poss = self._model.dist.get_policy_distribution(logits)
            pred = self._model.binary(pred)
        # pred = get_rotation_mask(torch.tensor(obs), [10,10,10])
        # pred = np.array(get_possible_position(torch.tensor(obs), [10,10,10]))

        value = float(value.squeeze(0))
        poss = poss.squeeze(0).cpu().detach().numpy()
        pred = pred.squeeze(0).cpu().detach().numpy()

        # np.set_printoptions(precision=3, suppress=True)
        # print('---------------------------')
        # print(pred1.reshape(10,10))
        # print(pred2.reshape(10,10))

        def softmax(x):
            probs = np.exp(x - np.max(x))
            probs /= np.sum(probs)
            return probs

        poss_in_actions = softmax(poss)
        if use_mask:
            if self.use_pusnet:
                action_mask = pred
            elif self.enable_rotation:
                action_mask = np.asarray(get_rotation_mask(x.squeeze(0), self.container_size), dtype=np.float32)
            else:
                action_mask = np.asarray(get_possible_position(x.squeeze(0), self.container_size), dtype=np.float32)

            if np.sum(action_mask) <= 0:
                action_mask = np.ones_like(poss_in_actions)
            poss_in_actions = poss_in_actions * action_mask
            if np.sum(poss_in_actions) <= 0:
                poss_in_actions = softmax(poss)
        poss_in_actions = np.reshape(poss_in_actions, newshape=(-1,))
        return value, poss_in_actions

    def sample_action(self, obs):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        if self.use_pusnet:
            vp, vu, pack_logits, unpack_logits, _ = self._model.base(x, 0, 0)
            choose_pack = (vp >= vu).float()
            choose_unpack = 1.0 - choose_pack
            pack_gate = torch.where(choose_pack > 0, torch.zeros_like(pack_logits), torch.full_like(pack_logits, -1e8))
            unpack_gate = torch.where(choose_unpack > 0, torch.zeros_like(unpack_logits), torch.full_like(unpack_logits, -1e8))
            poss = torch.cat((pack_logits + pack_gate, unpack_logits + unpack_gate), dim=-1)
            value = torch.maximum(vp, vu)
            cat = torch.distributions.Categorical(logits=poss)
        else:
            value, logits, _, pred= self._model.base(x, 0, 0)
            poss = self._model.dist.get_policy_distribution(logits)
            pred = self._model.binary(pred)
            cat = torch.distributions.Categorical(logits=poss+pred*7)

        value = float(value.squeeze(0))
        action = int(cat.sample().squeeze(0))

        return value, action


