import os
import config
import myenv
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#from tensorboard import SummaryWriter
import time
from tianshou.policy import PPOPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from gen_filter_matrix import gen
from mylogger import MyLogger
from ppo import test_ppo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CircuitDesign')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99) #0.99
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--step-per-epoch', type=int, default=50000)
    parser.add_argument('--episode-per-collect', type=int, default=64)
    parser.add_argument('--repeat-per-collect', type=int, default=2) #2
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=4)  # 20
    parser.add_argument('--test-num', type=int, default=1)  # 100
    parser.add_argument('--logdir', type=str, default='exp')
    parser.add_argument('--render', type=float, default=1/30.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95) #0.95
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--resume', type=bool, default=False)
    #args = parser.parse_known_args()[0]
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    args.resume = False

    #port_nums = [4,5,6,7,8]
    #epoches = [100,200,200,300,400]
    
    port_nums = [4]
    epoches = [5]
    vp = np.array([12,24,36,48])

    

    for p, epoch in zip(port_nums,epoches):
        root_path = f'./test/exp_{p}_port_hard_soft'
        node_num = p * 2 
        connect_num = p * 2
        valid_res = []
        time_res = []
        args.epoch = epoch
        exp_path = f"/port_{p}"
        args.logdir = root_path + exp_path + "/log"
        gen(p)
        print("testing port %d ..." % p)
        time_res.append(time.time())
        test_ppo(args, node_num, connect_num, valid_res, time_res,vp=vp)
        print("\n")
        print("valid_res num:", len(valid_res))
        with open(root_path + exp_path + "/RL_port_%d.csv" % p ,"w") as f:
            for x in valid_res:
                f.write(str(x)[1:-1]+"\n")
        with open(root_path + exp_path + "/time_RL_port_%d.csv" % p ,"w") as f:
            for x in time_res:
                f.write(str(x)+"\n")

