import torch.nn as nn


class Trainer(nn.Module):
    def __init__(self, dataloader, model, loss, optimizer, cfg):
        super(Trainer, self).__init__()
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = cfg['epochs']

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0

            for i, data in enumerate(self.dataloader):
                inputs, labels = data['img'], data['label']
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 250 == 249:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                    running_loss = 0.0

        print("Finished Training")