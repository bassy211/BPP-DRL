import argparse
import math
import os
from types import SimpleNamespace

import gym
import torch
import torch.nn as nn

from acktr.model import Policy


def _clean_state_dict(state_dict):
    load_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
    load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
    # Some historical checkpoints store linear/bias tensors with a trailing singleton dim.
    for k, v in load_dict.items():
        if torch.is_tensor(v) and len(v.size()) <= 3:
            load_dict[k] = v.squeeze(dim=-1)
    return load_dict


def _extract_state_dict(ckpt):
    if isinstance(ckpt, (list, tuple)) and len(ckpt) >= 1 and isinstance(ckpt[0], dict):
        return ckpt[0]
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            return ckpt['state_dict']
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError('Unsupported checkpoint format, cannot find state_dict.')


def _infer_model_config(state_dict, container_height):
    # Newer CNNPro checkpoints: base.share/base.mask naming.
    if 'base.mask.5.weight' in state_dict and 'base.share.0.weight' in state_dict:
        pred_len = int(state_dict['base.mask.5.weight'].shape[0])
        hidden_size = int(state_dict['base.mask.5.weight'].shape[1])
        channel = int(state_dict['base.share.0.weight'].shape[1])

        flat_dim = int(state_dict['base.actor.3.weight'].shape[1])
        if flat_dim % 8 != 0:
            raise ValueError(f'Unexpected actor flatten dim={flat_dim}, cannot infer pallet_size.')
        area = flat_dim // 8

        pallet_size = int(round(math.sqrt(area)))
        if pallet_size * pallet_size != area:
            raise ValueError(f'Action area={area} is not a perfect square, cannot infer pallet_size.')

        enable_rotation = pred_len == area * 2
        action_dim = pred_len if enable_rotation else area
        obs_len = channel * area

        args = SimpleNamespace(
            container_size=(pallet_size, pallet_size, container_height),
            pallet_size=pallet_size,
            enable_rotation=enable_rotation,
            channel=channel,
            use_pusnet=False,
        )

        return {
            'args': args,
            'obs_shape': (obs_len,),
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'area': area,
            'pred_len': pred_len,
            'model_family': 'cnnpro',
        }

    # PUSNet checkpoints: pack/unpack actor heads.
    if 'base.pack_actor.weight' in state_dict and 'base.shared.0.weight' in state_dict:
        area = int(state_dict['base.pack_actor.weight'].shape[0])
        hidden_size = int(state_dict['base.pack_actor.weight'].shape[1])
        pallet_size = int(round(math.sqrt(area)))
        if pallet_size * pallet_size != area:
            raise ValueError(f'Action area={area} is not a perfect square, cannot infer pallet_size.')

        obs_len = area * container_height + 3 * area
        action_dim = area * 2

        args = SimpleNamespace(
            container_size=(pallet_size, pallet_size, container_height),
            pallet_size=pallet_size,
            enable_rotation=False,
            channel=4,
            use_pusnet=True,
        )

        return {
            'args': args,
            'obs_shape': (obs_len,),
            'action_dim': action_dim,
            'hidden_size': hidden_size,
            'area': area,
            'pred_len': action_dim,
            'model_family': 'pusnet',
        }

    # Legacy checkpoints: position_head/box_mlp/fusion naming.
    try:
        area = int(state_dict['base.position_head.2.weight'].shape[0])
        box_feature_dim = int(state_dict['base.box_mlp.0.weight'].shape[1])
        hidden_size = int(state_dict['base.fusion.0.weight'].shape[0])
        pred_len = int(state_dict['base.mask.2.weight'].shape[0])
    except KeyError as exc:
        sample_keys = ', '.join(list(state_dict.keys())[:12])
        raise KeyError(
            f'Missing key for config inference: {exc}. Sample checkpoint keys: {sample_keys}'
        ) from exc

    pallet_size = int(round(math.sqrt(area)))
    if pallet_size * pallet_size != area:
        raise ValueError(f'Action area={area} is not a perfect square, cannot infer pallet_size.')

    enable_rotation = pred_len == area * 2
    obs_len = area * 2 + box_feature_dim
    action_dim = pred_len if enable_rotation else area

    args = SimpleNamespace(
        container_size=(pallet_size, pallet_size, container_height),
        pallet_size=pallet_size,
        enable_rotation=enable_rotation,
        channel=4,
        use_pusnet=False,
    )

    return {
        'args': args,
        'obs_shape': (obs_len,),
        'action_dim': action_dim,
        'hidden_size': hidden_size,
        'area': area,
        'pred_len': pred_len,
        'box_feature_dim': box_feature_dim,
        'model_family': 'legacy',
    }


class FlopCounter:
    def __init__(self, model):
        self.model = model
        self.macs = 0
        self.flops = 0
        self.handles = []
        self._register()

    def _register(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv2d_hook))
            elif isinstance(m, nn.Linear):
                self.handles.append(m.register_forward_hook(self._linear_hook))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.handles.append(m.register_forward_hook(self._bn_hook))
            elif isinstance(m, nn.ReLU):
                self.handles.append(m.register_forward_hook(self._relu_hook))

    def _conv2d_hook(self, module, inputs, output):
        out = output if torch.is_tensor(output) else output[0]
        out_elements = out.numel()
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        macs = out_elements * kernel_ops
        flops = 2 * macs
        if module.bias is not None:
            flops += out_elements
        self.macs += int(macs)
        self.flops += int(flops)

    def _linear_hook(self, module, inputs, output):
        out = output if torch.is_tensor(output) else output[0]
        batch = out.shape[0]
        macs = batch * module.in_features * module.out_features
        flops = 2 * macs
        if module.bias is not None:
            flops += batch * module.out_features
        self.macs += int(macs)
        self.flops += int(flops)

    def _bn_hook(self, module, inputs, output):
        out = output if torch.is_tensor(output) else output[0]
        self.flops += int(2 * out.numel())

    def _relu_hook(self, module, inputs, output):
        out = output if torch.is_tensor(output) else output[0]
        self.flops += int(out.numel())

    def close(self):
        for h in self.handles:
            h.remove()


def _human_size(num_bytes):
    mb = num_bytes / (1024 ** 2)
    return f'{mb:.4f} MB'


def main():
    parser = argparse.ArgumentParser(description='Compute model params, size and FLOPs for a checkpoint.')
    parser.add_argument('--model-path', type=str, default='pretrained_models/yang.pt', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for FLOPs profiling')
    parser.add_argument('--container-height', type=int, default=10, help='Container height used by model args')
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    raw_state_dict = _extract_state_dict(checkpoint)
    state_dict = _clean_state_dict(raw_state_dict)

    cfg = _infer_model_config(state_dict, args.container_height)

    obs_shape = cfg['obs_shape']
    action_space = gym.spaces.Discrete(cfg['action_dim'])
    policy = Policy(
        obs_shape=obs_shape,
        action_space=action_space,
        base_kwargs={'recurrent': False, 'hidden_size': cfg['hidden_size'], 'args': cfg['args']},
    )
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    device = torch.device(args.device)
    policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    param_bytes = sum(p.numel() * p.element_size() for p in policy.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in policy.buffers())
    memory_bytes = param_bytes + buffer_bytes
    file_bytes = os.path.getsize(model_path)

    flop_counter = FlopCounter(policy)

    with torch.no_grad():
        x = torch.randn(args.batch_size, obs_shape[0], device=device)
        rnn_hxs = torch.zeros(args.batch_size, policy.recurrent_hidden_state_size, device=device)
        masks = torch.ones(args.batch_size, 1, device=device)

        if bool(getattr(policy, 'use_pusnet', False)):
            vp, vu, pack_logits, unpack_logits, _ = policy.base(x, rnn_hxs, masks)
            _ = vp + vu + pack_logits.sum(dim=-1, keepdim=True) + unpack_logits.sum(dim=-1, keepdim=True)
        else:
            _, logits, _, _ = policy.base(x, rnn_hxs, masks)
            _ = policy.dist.get_policy_distribution(logits)

    macs_total = flop_counter.macs
    flops_total = flop_counter.flops
    flop_counter.close()

    macs_per_sample = macs_total / args.batch_size
    flops_per_sample = flops_total / args.batch_size

    print('========== Model Profile ==========' )
    print(f'Model path: {model_path}')
    print(f'Obs shape: {obs_shape}')
    print(f'Action dim: {cfg["action_dim"]}')
    print(f'Pallet size: {cfg["args"].pallet_size}x{cfg["args"].pallet_size}')
    print(f'Enable rotation: {cfg["args"].enable_rotation}')
    print(f'Model family: {cfg["model_family"]}')
    print('-----------------------------------')
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Params+buffers memory: {memory_bytes:,} bytes ({_human_size(memory_bytes)})')
    print(f'Checkpoint file size: {file_bytes:,} bytes ({_human_size(file_bytes)})')
    print('-----------------------------------')
    print(f'MACs (batch={args.batch_size}): {macs_total:,}')
    print(f'FLOPs (batch={args.batch_size}): {flops_total:,}')
    print(f'MACs per sample: {macs_per_sample:,.0f}')
    print(f'FLOPs per sample: {flops_per_sample:,.0f}')
    print('===================================')


if __name__ == '__main__':
    main()