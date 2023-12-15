import torch


class Validater:
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.correct = 0
        self.total = 0

    def eval(self, dataloader):
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                images, labels = data['img'], data['label']
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                self.total += labels.size(0)
                self.correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * self.correct // self.total} %')

