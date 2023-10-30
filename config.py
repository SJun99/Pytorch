
DATA = {
    'type': ['Mnist', 'Cifar10'][0],
    'resolution': (32, 32),
    'PATH' : ['/home/ysj/Desktop/sdsd/mnist'][0]
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
