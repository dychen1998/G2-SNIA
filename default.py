import os

# change this path to your directory.
project_path = '/home/G2-SNIA'
datasets_path = os.path.join(project_path, 'datasets_file')
adv_samples_path = os.path.join(project_path, 'adversarial_samples')
algos_path = os.path.join(project_path, 'algorithms')
baseline_path = os.path.join(project_path, 'baseline')
models_path = os.path.join(project_path, 'models_file')
visualizations_path = os.path.join(project_path, 'visualizations_file')

# all datasets we used.
_ALL_DATASET = {
    'pt': ['citeseer', 'cora', 'dblp'],
}

# all graph neural networks we used.
_ALL_MODEL = {
    'surrogate': ['gcn'],
    'victim': ['gcn', 'sgcn', 'tagcn', 'gcnii'],
}
    
# all attack methods we used.
_ALL_ATTACK = {
    'random': {
        'scenario': 'gia',
        'type': 'black-box'
    },
    'most_attr': {
        'scenario': 'gia',
        'type': 'black-box'
    },
    'afgsm': {
        'scenario': 'gia',
        'type': ['white-box', 'black-box']
    },
    'gb_fgsm': {
        'scenario': 'gia',
        'type': ['white-box', 'black-box']
    },
    'ugba': {
        'scenario': 'gia',
        'type': ['black-box']
    },
    'gafnc': {
        'scenario': 'gia',
        'type': ['black-box']
    },
    'g2_snia': {
        'scenario': 'gia',
        'type': 'black-box'
    },
}

# training settings for GNNs
_SET = {
    'cora': {
        'gcn': {
            'net_set': {
                'hidden_layers': [64],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'surrogate': {
            'net_set': {
                'hidden_layers': [64],
                'activation': ''
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'sgcn': {
            'net_set': {
                'activation': 'tanh',
                'hidden_layers': [64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.6,
                'seed': 42,
                'patience': 500,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'tagcn': {
            'net_set': {
                'activation': 'leak_relu',
                'hidden_layers': [64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'gcnii': {
            'net_set': {
                'activation': 'relu',
                'hidden_layers': [64 for _ in range(5)],
                'lamda': 0.5,
                'alpha': 0.2
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.6,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        }
    },
    'dblp': {
        'gcn': {
            'net_set': {
                'hidden_layers': [64],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'surrogate': {
            'net_set': {
                'hidden_layers': [64],
                'activation': ''
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'sgcn': {
            'net_set': {
                'activation': 'relu',
                'hidden_layers': [64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 500,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'tagcn': {
            'net_set': {
                'activation': 'relu',
                'hidden_layers': [64, 64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.7,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'gcnii': {
            'net_set': {
                'activation': 'relu',
                'hidden_layers': [64 for _ in range(5)],
                'lamda': 0.5,
                'alpha': 0.2
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.6,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
    },
    'citeseer': {
        'gcn': {
            'net_set': {
                'hidden_layers': [64],
                'activation': 'relu'
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 3407,
                'patience': 100,
                'opt': 'adam',
                'lr': 5e-3,
                'weight_decay': 5e-4,
            }
        },
        'surrogate': {
            'net_set': {
                'hidden_layers': [64],
                'activation': ''
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'sgcn': {
            'net_set': {
                'activation': 'tanh',
                'hidden_layers': [64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.5,
                'seed': 42,
                'patience': 500,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'tagcn': {
            'net_set': {
                'activation': 'leak_relu',
                'hidden_layers': [64]
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.7,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
        'gcnii': {
            'net_set': {
                'activation': 'relu',
                'hidden_layers': [64 for _ in range(5)],
                'lamda': 0.6,
                'alpha': 0.2
            },
            'train_set': {
                'max_epoch': 500,
                'dropout': 0.7,
                'seed': 42,
                'patience': 100,
                'opt': 'adam',
                'lr': 1e-2,
                'weight_decay': 5e-4,
            }
        },
    },
}
