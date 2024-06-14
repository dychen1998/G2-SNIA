import os
from typing import Tuple, Optional, List
import time

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from tqdm import trange

from config import cfg
from algorithms.models.actor import ActorLayer
from algorithms.models.critic import CriticLayer
from algorithms.utils.load import set_seed
from algorithms.tools.environment import Environment


class PPO:
    def __init__(self, dataset_name: str, model_name: str, input_dims: int, output_dims: int, device: Optional[torch.device] = None):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device
        # 设置随机种子
        set_seed(cfg.rl.seed)
        # 初始化Actor
        self.actor = ActorLayer(input_dims, output_dims).to(self.device)
        # 初始化Critic
        self.critic = CriticLayer(input_dims).to(device)
        self.critic_loss_func = torch.nn.SmoothL1Loss(reduction="mean")
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            [{'params': self.actor.parameters()}, {'params': self.critic.parameters()}],
            cfg.rl.lr,
            eps=1e-5)
        # 一轮采样中的最后一个state
        self.last_state = None
        self.last_mask = None
        # 参数设置
        self.discount = cfg.rl.discount
        self.lambda_gae_adv = cfg.rl.lambda_gae_adv
        self.lambda_entropy = cfg.rl.lambda_entropy
        self.ratio_clip = cfg.rl.ratio_clip
        self.batch_size = cfg.rl.batch_size
        self.repeat_times = cfg.rl.repeat_times

    @torch.no_grad()
    def _policy(self, states: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param states: (num_envs,state_dim)
        :param masks:  (num_envs,F)
        :return  actions, action_log_pros: ((num_envs,1), (num_envs,1))
        """
        states_embed, feats_mask = states, masks
        # (num_envs,F)
        feats_output = self.actor(states_embed)
        # (num_envs,F)
        feats_dist = feats_output
        feats_dist[~feats_mask] = float('-inf')
        feats_dist = torch.softmax(feats_dist, dim=1)
        # [(num_envs,F)]
        dists = [feats_dist]
        # [(num_envs,1)]
        actions = [self.gumbel_max(dists[i]).unsqueeze(dim=1) for i in range(len(dists))]
        # [(num_envs,1)]
        actions_log_prob = [torch.gather(dists[i], 1, actions[i]).log() for i in range(len(dists))]
        return torch.cat(actions, dim=-1), torch.cat(actions_log_prob, dim=-1)

    @torch.no_grad()
    def evaluate_policy(self, states: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Agent的策略函数
        :param states: (num_envs,state_dim)
        :param masks:  (num_envs,F)
        :return:  action, action_log_pro Agent做出的动作 动作概率的对数
        """
        self.actor.eval()
        states_embed, feats_mask = states, masks
        # (num_envs,F)
        feats_output = self.actor(states_embed)
        # (num_envs,F)
        feats_dist = feats_output
        feats_dist[~feats_mask] = float('-inf')
        feats_dist = torch.softmax(feats_dist, dim=1)
        # [(num_envs,F)]
        dists = [feats_dist]
        # [(num_envs,1)]
        actions = [torch.argmax(dists[i], dim=-1).unsqueeze(dim=1) for i in range(len(dists))]
        # [(num_envs,1)]
        actions_log_prob = [torch.gather(dists[i], 1, actions[i]).log() for i in range(len(dists))]
        return torch.cat(actions, dim=-1), torch.cat(actions_log_prob, dim=-1)

    def explore_vec_env(self, env: Environment, num_envs: int, target_step: int, ) -> [Tensor]:
        """
        :param env: the DRL environment
        :param target_step: the target step for the interaction
        :param num_envs: the number of environment
        :return: [buf_s, buf_m, buf_a, buf_a_log, buf_r, buf_un_dones]
        """
        self.actor.eval()
        self.critic.eval()
        states = torch.empty((target_step, num_envs, env.state_dim), dtype=torch.float, device=self.device)
        masks = torch.empty((target_step, num_envs, env.action_dim), dtype=torch.bool, device=self.device)

        actions = torch.empty((target_step, num_envs, 1), dtype=torch.float, device=self.device)
        log_probs = torch.empty((target_step, num_envs, 1), dtype=torch.float, device=self.device)
        rewards = torch.empty((target_step, num_envs, 1), dtype=torch.float, device=self.device)
        dones = torch.empty((target_step, num_envs, 1), dtype=torch.bool, device=self.device)
        if self.last_state is not None:
            state = self.last_state
            mask = self.last_mask
        else:
            state, mask = env.reset(num_envs=num_envs)
        count = 0
        episode_num = 0
        for t in trange(target_step, colour='white', desc='Experience Sample'):
            a, a_prob = self._policy(state, mask)
            states[t], masks[t] = state, mask
            nx_state, nx_mask, r, done, info = env.step(a, a_prob)
            cur_episode = done.sum()
            if cur_episode <= 0:
                state, mask = nx_state, nx_mask
            else:
                episode_num += cur_episode
                count += (done * info).sum()
                state, mask = env.reset(num_envs=num_envs)
            actions[t] = a
            log_probs[t] = a_prob
            rewards[t] = r
            dones[t] = done
        self.last_state = state
        self.last_mask = mask
        un_dones = 1.0 - dones.float()
        print(f"共进行{episode_num}个episode 攻击成功次数为: {count} r_exp: {rewards.mean().item()}")
        return states, masks, actions, log_probs, rewards, un_dones, self.last_state

    def get_reward_sum_gae(self, buf_len: int, buf_rewards: Tensor, buf_un_dones: Tensor, buf_values: Tensor,
                           last_state):
        """
        计算 A(s,a)

        :param buf_len:
        :param buf_rewards:
        :param buf_un_dones:
        :param buf_values:
        :param last_state:
        :return:
        """
        # (steps,num_envs,1)
        buf_adv_v = torch.empty_like(buf_values)
        # (steps,num_envs,1)
        buf_masks = buf_un_dones * self.discount
        # (num_envs,1)
        nx_value = self.critic(last_state)
        # (num_envs,1)
        adv = torch.zeros_like(nx_value)
        for t in range(buf_len - 1, -1, -1):
            nx_value = buf_rewards[t] + buf_masks[t] * nx_value
            buf_adv_v[t] = adv = nx_value - buf_values[t] + buf_masks[t] * self.lambda_gae_adv * adv
            nx_value = buf_values[t]

        return buf_adv_v

    def get_batch_log_prob_entropy(self, states: Tensor, masks: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param states: (B,state_dim)
        :param masks: (B,F)
        :param actions: (B,1)
        :return: log_prob, average_entropy: ((B, 1), (1,))
        """
        states_embed, feats_mask = states, masks
        # (B,F)
        feats_output = self.actor(states_embed)
        # (B,F)
        feats_dist = feats_output
        feats_dist[~feats_mask] = float('-inf')
        feats_dist = torch.softmax(feats_dist, dim=1)

        dists = [Categorical(feats_dist)]
        actions_log_prob = [dists[i].log_prob(actions[:, i]) for i in range(len(dists))]
        actions_entropy = [dists[i].entropy() for i in range(len(dists))]
        # (B,1)
        actions_log_prob = torch.stack(actions_log_prob, dim=1)
        # (B,1)
        actions_entropy = torch.stack(actions_entropy, dim=0)
        return actions_log_prob, actions_entropy

    def update_net(self, buffer: [Tensor]) -> List[float]:
        """
        update actor and critic
        :param buffer: [buf_s, buf_m, buf_a, buf_a_log, buf_r, buf_un_d]
        :return:
        """
        # buf_s, buf_m: (steps,num_envs,state_dim), (steps,num_envs,F)
        # buf_a, buf_a_prob, buf_r, buf_un_dones: (steps,num_envs,1)
        buf_s, buf_m, buf_a, buf_a_prob, buf_r, buf_un_dones, last_state = buffer
        '''calculate gae'''
        buf_len = buf_r.shape[0]
        buf_num = buf_r.shape[1]
        with torch.no_grad():
            # (steps,num_envs,1)
            buf_values = self.critic(buf_s)
            # (steps,num_envs,1)
            buf_log_prob = buf_a_prob
            # (steps,num_envs,1)
            buf_adv_v = self.get_reward_sum_gae(buf_len, buf_r, buf_un_dones, buf_values, last_state)
            # (steps,num_envs,1)
            buf_r_sum = buf_adv_v + buf_values
            # (steps,num_envs,1)
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-4)
        del buf_r, buf_un_dones, buf_values
        '''update'''
        self.actor.train()
        self.critic.train()
        # [average_actor_loss, average_actor_loss, average_entropy_loss, average_approxkl]
        record_curve = torch.zeros(4, dtype=torch.float, device=self.device, requires_grad=False)
        train_steps = int((buf_len - 1) * buf_num * self.repeat_times / self.batch_size)
        for _ in trange(train_steps):
            ids = torch.randint((buf_len - 1) * buf_num, size=(self.batch_size,), requires_grad=False)
            # (B,)
            ids0 = torch.fmod(ids, buf_len - 1)
            # (B,)
            ids1 = torch.div(ids, buf_len - 1, rounding_mode='floor')
            # (B,state_dim)
            states = buf_s[(ids0, ids1)]
            # (B, F)
            masks = buf_m[(ids0, ids1)]
            # (B,1)
            r_sum = buf_r_sum[(ids0, ids1)]
            # (B,1)
            adv_v = buf_adv_v[(ids0, ids1)]
            # (B,1)
            actions = buf_a[(ids0, ids1)]
            # (B,1)
            log_prob = buf_log_prob[(ids0, ids1)]

            '''actor loss'''
            # (B,1), (1,)
            new_log_prob, entropy = self.get_batch_log_prob_entropy(states, masks, actions)
            # (B,1)
            ratio = (new_log_prob - log_prob).exp()
            # (B,1) broadcasting
            surrogate1 = -adv_v * ratio
            surrogate2 = -adv_v * torch.clamp(ratio, min=1 - self.ratio_clip, max=1 + self.ratio_clip)
            surrogate = torch.max(surrogate1, surrogate2).mean()
            entropy_loss = entropy.mean()
            actor_loss = surrogate - self.lambda_entropy * entropy_loss
            '''critic loss'''
            # V(s) (B,1)
            v = self.critic(states)
            critic_loss = self.critic_loss_func(v, r_sum)
            '''update actor critic'''
            loss = actor_loss + critic_loss
            self.optimizer_update(loss)

            record_curve[0] += actor_loss.item()
            record_curve[1] += critic_loss.item()
            record_curve[2] += ((ratio - 1) - ratio.log()).mean().item()
            # print(r_sum.mean(), new_log_prob.mean(), log_prob.mean(), ratio.mean(), record_curve[2])
            record_curve[3] += entropy_loss.item()
        record_curve /= train_steps
        # [average_actor_loss, average_actor_loss, average_entropy_loss, average_approx_kl]
        return record_curve.tolist()

    @staticmethod
    def gumbel_max(dist: Tensor):
        u = torch.rand_like(dist)
        z = dist.log() - (-u.log()).log()
        return z.argmax(dim=-1)

    def optimizer_update(self, loss: Tensor):
        """
        优化Actor Critic
        :param loss: 目标函数
        :return: None
        """
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

    def lr_decay(self, cur_steps):
        lr_new = cfg.rl.lr * (1 - (cur_steps - 1) / cfg.env.max_epoch)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_new

    def save(self):
        """
        保存模型参数

        :return: None
        """
        if not os.path.exists(cfg.save.agent_path):
            os.makedirs(cfg.save.agent_path)

        actor_path = os.path.join(cfg.save.agent_path,
                                  f'{self.dataset_name}_{self.model_name}_actor.pt')
        critic_path = os.path.join(cfg.save.agent_path,
                                   f'{self.dataset_name}_{self.model_name}_critic.pt')
        torch.save(
            {'state_dict': {k: v.to('cpu') for k, v in self.actor.state_dict().items()}},
            actor_path
        )
        torch.save(
            {'state_dict': {k: v.to('cpu') for k, v in self.critic.state_dict().items()}},
            critic_path
        )

    def load(self):
        """
        加载模型

        :return: None
        """
        actor_path = os.path.join(cfg.save.agent_path, f'{self.dataset_name}_{self.model_name}_actor.pt')
        critic_path = os.path.join(cfg.save.agent_path, f'{self.dataset_name}_{self.model_name}_critic.pt')

        actor_dict = torch.load(actor_path, map_location=self.device)
        self.actor.load_state_dict(actor_dict['state_dict'])

        critic_dict = torch.load(critic_path, map_location=self.device)
        self.critic.load_state_dict(critic_dict['state_dict'])


if __name__ == '__main__':
    device = torch.device('cuda:0')
    env = Environment(dataset_name='cora', model_name='gcnii', device=device, render=False)
    agent = PPO(env.dataset_name, env.model_name, input_dims=env.input_dims, output_dims=env.output_dims, device=device)
    start = time.time()
    buffer = agent.explore_vec_env(env, 64, 128)
    end = time.time()
    print(f'time cost: {end - start}')
    print(agent.update_net(buffer))
