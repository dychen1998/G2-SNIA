import os
import argparse

import numpy as np
import torch
from tqdm import trange

from default import adv_samples_path
from config import cfg
from algorithms.tools.environment import Environment
from algorithms.tools.ppo import PPO


def get_misrate(env: Environment, agent: PPO, save: bool = False) -> float:
    """
    test the performance of the agent

    :param env: environment
    :param agent: trained agent
    :param save: save samples
    :return: attack success rate
    """

    nodes = env.attack_index
    env.render = False
    batch_size = cfg.rl.num_envs
    iter_times = nodes.shape[0] // batch_size
    add_times = 0 if nodes.shape[0] % batch_size == 0 else 1
    num_classes = env.dataset.num_classes
    t_mis_num = torch.zeros(num_classes, dtype=torch.float)

    snia_samples_path = os.path.join(adv_samples_path, 'g2_snia', env.dataset_name, env.model_name)
    if not os.path.exists(snia_samples_path):
        os.makedirs(snia_samples_path)

    for c_new in range(num_classes):
        adv_samples = {}
        save_samples_path = os.path.join(snia_samples_path, f'{c_new}.pt')
        cur_index = 0
        for _ in trange(iter_times + add_times):
            nx_index = cur_index + batch_size
            if nx_index < nodes.shape[0]:
                target_node = nodes[cur_index:nx_index]
                num_envs = batch_size
                cur_index = nx_index
            else:
                target_node = nodes[cur_index:]
                num_envs = target_node.shape[0]
            target_label = torch.zeros(target_node.shape[0], dtype=torch.long, device=env.device) + c_new
            s, m = env.reset(target_node, target_label, num_envs=num_envs)
            done = torch.zeros((num_envs, 1), dtype=torch.bool)
            while not torch.all(done):
                a, a_prob = agent.evaluate_policy(s, m)
                nx_s, nx_m, r, done, info = env.step(a, a_prob)
                s = nx_s
                m = nx_m
            t_mis_num[c_new] += env.is_misclassify(record=False)
            if not save:
                continue
            for i, v_t in enumerate(target_node.tolist()):
                adv_samples[str(v_t)] = [env.vec_graph[0][i][-1].to('cpu'), env.vec_graph[-2][i].to('cpu')]
        if not save:
            continue
        torch.save(adv_samples, save_samples_path)

    attack_succ = t_mis_num * 100 / nodes.shape[0]
    average_succ = round(torch.mean(attack_succ).item(), 2)
    attack_succ = np.round(attack_succ.tolist(), decimals=1).tolist()

    result = {
        'attack numbers': nodes.shape[0],
        'attack success rate': attack_succ,
        'average attack success rate': average_succ
    }
    print('g2_snia', env.dataset_name, env.model_name, result)
    return average_succ


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f'cuda:{cfg.attack.gpu}')
    env = Environment(
        dataset_name=cfg.attack.dataset,
        model_name=cfg.attack.model,
        device=device,
        render=False
    )

    agent = PPO(
        dataset_name=env.dataset_name,
        model_name=env.model_name,
        input_dims=env.input_dims,
        output_dims=env.output_dims,
        device=device
    )
    agent.load()
    # 控制显存大小
    if cfg.attack.dataset == 'dblp' and cfg.attack.model == 'tagcn':
        cfg.rl.num_envs = 21
    else:
        cfg.rl.num_envs = 64
    # performance
    mis_rate = get_misrate(env, agent, save=True)
    print("misclassification_rate: ", mis_rate)


def parse_set():
    """
    配置参数

    :return: None
    """
    parser = argparse.ArgumentParser(description='parse set')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    cfg.attack.dataset = args.dataset
    cfg.attack.model = args.model
    cfg.attack.gpu = args.gpu


if __name__ == '__main__':
    parse_set()
    main()
