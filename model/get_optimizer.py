import torch.optim as optim

def get_optimizer(cfg, model_parameters):
    optimizer_type = cfg['type']
    lr = cfg['learning_rate']

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_parameters, lr=lr, momentum=cfg['momentum'])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model_parameters, lr=lr)

    return optimizer
