import os
import argparse

import torch

from config import cfg
from algorithms.tools.environment import Environment
from algorithms.tools.ppo import PPO
from algorithms.utils.visualize import make_writer
from test_agent import get_misrate

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    device = torch.device(f'cuda:{cfg.attack.gpu}')
    env = Environment(dataset_name=cfg.attack.dataset,
                      model_name=cfg.attack.model,
                      device=device,
                      render=False)
    agent = PPO(env.dataset_name, env.model_name, input_dims=env.input_dims, output_dims=env.output_dims, device=device)
    target_step = cfg.rl.target_step
    # 控制显存大小
    if cfg.attack.dataset == 'dblp' and cfg.attack.model == 'tagcn':
        cfg.rl.num_envs = 21
    else:
        cfg.rl.num_envs = 64
    num_envs = cfg.rl.num_envs
    max_epoch = cfg.env.max_epoch
    best_misrate = 0.
    epoch = 0
    early_stopping = cfg.env.patience
    writer = make_writer(env.dataset_name, env.model_name)

    while True:
        epoch += 1
        # 采样
        buffer = agent.explore_vec_env(env, num_envs, target_step)
        r_exp = buffer[4].mean().item()
        # 更新网络
        agent.lr_decay(epoch)
        actor_loss, critic_loss, approx_kl, entropy_loss = agent.update_net(buffer)
        writer.add_scalar(
            tag='Performance/r_exp',
            scalar_value=r_exp,
            global_step=epoch
        )
        writer.add_scalar(
            tag='Loss/actor_loss',
            scalar_value=actor_loss,
            global_step=epoch
        )
        writer.add_scalar(
            tag='Loss/critic_loss',
            scalar_value=critic_loss,
            global_step=epoch
        )
        writer.add_scalar(
            tag='Loss/approx_kl',
            scalar_value=approx_kl,
            global_step=epoch
        )
        writer.add_scalar(
            tag='Loss/entropy_loss',
            scalar_value=entropy_loss,
            global_step=epoch
        )
        if (epoch - 1) % 400 == 0:
            test_misrate = get_misrate(env, agent)
            print(test_misrate)
            agent.last_state = None
            writer.add_scalars(
                main_tag='Performance/misrate',
                tag_scalar_dict={
                    'test': test_misrate
                },
                global_step=epoch
            )
            if test_misrate > best_misrate:
                agent.save()
                best_misrate = test_misrate
                early_stopping = cfg.env.patience
            else:
                early_stopping -= 1

            writer.add_scalars(
                main_tag='Performance/el',
                tag_scalar_dict={
                    'patience': early_stopping,
                },
                global_step=epoch
            )
        if epoch >= max_epoch or early_stopping <= 0:
            break
    writer.close()

    test_misrate = get_misrate(env, agent)
    print("test_misrate: ", test_misrate)


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
