import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from acktr.algo.kfac import KFACOptimizer
import sys

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 lr = None,
                 eps = None,
                 alpha = None,
                 max_grad_norm = None,
                 acktr = False,
                 args = None):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduce=False, size_average=True)
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

        self.branch_cursor = 0
        self.last_active_branch = 'pack'
        self.last_branch_samples = 0

    def _set_branch_trainable(self, branch):
        if (not self.args.use_pusnet) or (not hasattr(self.actor_critic.base, 'pack_actor')):
            return
        pack_train = branch == 'pack'
        for p in self.actor_critic.base.pack_actor.parameters():
            p.requires_grad = pack_train
        for p in self.actor_critic.base.pack_critic.parameters():
            p.requires_grad = pack_train
        for p in self.actor_critic.base.unpack_actor.parameters():
            p.requires_grad = (not pack_train)
        for p in self.actor_critic.base.unpack_critic.parameters():
            p.requires_grad = (not pack_train)

    def _reset_trainable(self):
        if (not self.args.use_pusnet) or (not hasattr(self.actor_critic.base, 'pack_actor')):
            return
        for p in self.actor_critic.base.pack_actor.parameters():
            p.requires_grad = True
        for p in self.actor_critic.base.pack_critic.parameters():
            p.requires_grad = True
        for p in self.actor_critic.base.unpack_actor.parameters():
            p.requires_grad = True
        for p in self.actor_critic.base.unpack_critic.parameters():
            p.requires_grad = True

    def _pick_active_branch(self, pack_count, unpack_count):
        mode = getattr(self.args, 'branch_update_mode', 'alternating')
        if mode == 'pack':
            return 'pack'
        if mode == 'unpack':
            return 'unpack'
        if mode == 'auto':
            return 'pack' if pack_count >= unpack_count else 'unpack'
        branch = 'pack' if (self.branch_cursor % 2 == 0) else 'unpack'
        self.branch_cursor += 1
        return branch


    def update(self, rollouts):
        # check_nan(self.actor_critic, 1)
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        mask_size = rollouts.location_masks.size()[-1]

        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.location_masks[:-1].view(-1, mask_size))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values

        if self.args.use_pusnet:
            area = self.args.container_size[0] * self.args.container_size[1]
            actions_flat = rollouts.actions.view(-1, action_shape).squeeze(-1)
            is_pack = actions_flat < area
            pack_count = int(is_pack.sum().item())
            unpack_count = int((~is_pack).sum().item())
            active_branch = self._pick_active_branch(pack_count, unpack_count)
            selected = is_pack if active_branch == 'pack' else (~is_pack)
            if int(selected.sum().item()) == 0:
                active_branch = 'unpack' if active_branch == 'pack' else 'pack'
                selected = is_pack if active_branch == 'pack' else (~is_pack)

            self._set_branch_trainable(active_branch)

            adv_flat = advantages.view(-1, 1)
            logp_flat = action_log_probs.view(-1, 1)
            if int(selected.sum().item()) > 0:
                value_loss = adv_flat[selected].pow(2).mean()
                action_loss = -(adv_flat[selected].detach() * logp_flat[selected]).mean()
            else:
                value_loss = adv_flat.pow(2).mean() * 0.0
                action_loss = logp_flat.mean() * 0.0

            self.last_active_branch = active_branch
            self.last_branch_samples = int(selected.sum().item())
        else:
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()

        mask_len = self.args.container_size[0]*self.args.container_size[1]
        if self.args.use_pusnet:
            mask_len = mask_len * 2
        else:
            mask_len = mask_len * (1 + self.args.enable_rotation)

        if self.args.use_pusnet:
            graph_loss = torch.zeros(1, device=values.device).mean()
        else:
            pred_mask = pred_mask.reshape((num_steps, num_processes, mask_len))
            mask_truth = rollouts.location_masks[0:num_steps]
            graph_loss = self.loss_func(pred_mask, mask_truth).mean()
        dist_entropy = dist_entropy.mean()
        prob_loss = torch.zeros(1, device=values.device).mean() if self.args.use_pusnet else bad_prob.mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() # detach

            fisher_loss = pg_fisher_loss + vf_fisher_loss + graph_loss * 1e-8
            # fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        force = 0.5 * 10
        self.optimizer.zero_grad()
        loss = value_loss * self.value_loss_coef
        loss += action_loss
        loss += prob_loss * self.invaild_coef
        loss -= dist_entropy * self.entropy_coef
        loss += force * graph_loss
        loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self._reset_trainable()

        # return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item()
        return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item(), graph_loss.item()

def check_nan(model,index):
    for p in model.parameters():
        if np.isnan(p.grad.data.mean().item()):
            print('index '+ str(index) +' happened an error!')