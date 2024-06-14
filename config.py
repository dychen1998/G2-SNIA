import os
from yacs.config import CfgNode as CN

from default import models_path, datasets_path, adv_samples_path

_C = CN()
_C.gpu_num = 0
_C.dataset = ''
_C.model = ''
_C.role = ''
_C.net = CN()
"""GCN"""
_C.net.gcn = CN()
_C.net.gcn.hidden_layers = None
_C.net.gcn.activation = ''
"""GCNII"""
_C.net.gcnii = CN()
_C.net.gcnii.hidden_layers = None
_C.net.gcnii.activation = ''
_C.net.gcnii.lamda = ''
_C.net.gcnii.alpha = ''
"SGCN"
_C.net.sgcn = CN()
_C.net.sgcn.activation = ''
_C.net.sgcn.hidden_layers = None
"TAGCN"
_C.net.tagcn = CN()
_C.net.tagcn.activation = ''
_C.net.tagcn.hidden_layers = None
"""save directory"""
_C.save = CN()
_C.save.datasets_path = datasets_path
_C.save.adv_samples_path = adv_samples_path
_C.save.surrogate_path = os.path.join(models_path, 'surrogate')
_C.save.victim_path = os.path.join(models_path, 'victim')
_C.save.agent_path = os.path.join(models_path, 'agent')
_C.save.defense_path = os.path.join(models_path, 'defense')
"""training settings for GNNs"""
_C.train = CN()
_C.train.max_epoch = 5000
_C.train.patience = 30
_C.train.dropout = 0.1
_C.train.seed = 42
"""optimizer settings"""
_C.train.optimal = CN()
_C.train.optimal.name = 'adam'
_C.train.optimal.lr = 1e-3
_C.train.optimal.weight_decay = 0
"""attack settings"""
_C.attack = CN()
_C.attack.method = ''
_C.attack.name = ''
_C.attack.dataset = 'cora'
_C.attack.model = 'gcn'
_C.attack.gpu = 0
_C.attack.seed = 42
_C.attack.edge_budget = 1
_C.attack.feat_budget = 'max_feat_num'
"""environment settings"""
_C.env = CN()
# agent training epochs
_C.env.max_epoch = 40_000
# early stopping
_C.env.patience = 20
"""reinforcement learning settings"""
_C.rl = CN()
_C.rl.seed = 42
_C.rl.lr = 2e-4
_C.rl.discount = 0.99
_C.rl.repeat_times = 3
_C.rl.ratio_clip = 0.1
_C.rl.lambda_entropy = 0.02
_C.rl.lambda_gae_adv = 0.95
_C.rl.target_step = 64
_C.rl.num_envs = 64
_C.rl.batch_size = 512

cfg = _C
