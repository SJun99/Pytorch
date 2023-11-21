
DATA = {
    'type': ['Mnist', 'Cifar10'][1],
    'dataset': {
        'resolution': (32, 32),
        # -> transform,
        'path': ['/home/ysj/Desktop/sdsd/mnist', '/home/ysj/Desktop/sdsd/cifar10'][1],
        'mode': ['train', 'test'][0],
        'batch_size' : [0, 4][1],
    },
}

MODEL = {
    'backbone': {
        'type': 'ResNet50',
        # ...
        },
    'head': {
        'type': 'MlpClassifier',
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
    'type': ['Adam', 'SGD'][0],
    'learning_rate' : [0.1, 0.01, 0.001][0],
    # ...
    }

TRAIN = {
    'epochs': 10
}
# ...
