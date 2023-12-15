
DATA = {
    'type': ['Mnist', 'Cifar10'][1],
    'class': {
        'Cifar10': {'airplane' : 0, 'automobile' : 1, 'bird' : 2, 'cat' : 3, 'deer' : 4, 'dog' : 5
            , 'frog' : 6, 'horse' : 7, 'ship' : 8, 'truck' : 9},
    },
    'dataset': {
        'resolution': (32, 32),
        # -> transform,
        'path': ['/home/ysj/Desktop/sdsd/mnist', '/home/ysj/Desktop/sdsd/cifar10/'][1],
        'mode': ['train', 'test'][0],
        'batch_size' : [0, 4, 32][2],
    },
}

MODEL = {
    'backbone': {
        'type': ['ResNet50', 'Net_backbone'][1],
        # ...
        },
    'head': {
        'type': ['MlpClassifier', 'Net_head'][1],
        # ...
        },
    'in_channels': [1, 3][1],
    'in_features': [65536][0],
    'out_features': [10][0],
    }

LOSS = {
    'type': ['CrossEntropy', 'MSELoss'][0],
    # ...
    }

OPTIMIZER = {
    'type': ['Adam', 'SGD'][1],
    'learning_rate': [0.1, 0.01, 0.001][2],
    'momentum': [0.9][0]
    # ...
    }

TRAIN = {
    'epochs': [2, 10][1],
}
# ...
