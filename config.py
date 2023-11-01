
DATA = {
    'type': ['Mnist', 'Cifar10'][1],
    'resolution': (32, 32),
    'PATH': ['/home/ysj/Desktop/sdsd/mnist', '/home/ysj/Desktop/sdsd/cifar10'][1],
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
    'type': 'CrossEntropy',
    # ...
    }

OPTIMIZER = {
    'type': 'Adam',
    # ...
    }

BATCH_SIZE = 4

TRAIN = {
    'epochs': 10
}
# ...
