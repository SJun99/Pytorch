import torch


class Validater:
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.correct = 0
        self.total = 0
        self.classes = cfg['class']['Cifar10']
        self.reversed_classes = {v: k for k, v in self.classes.items()}

    def evaluation(self, dataloader):
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                images, labels = data['img'], data['label']
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                self.total += labels.size(0)
                self.correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * self.correct // self.total} %')

    def individual_evaluation(self, dataloader):
        correct_pred = {classname: 0 for classname in self.reversed_classes.values()}
        total_pred = {classname: 0 for classname in self.reversed_classes.values()}

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                images, labels = data['img'], data['label']
                outputs = self.model(images)

                _, predictions = torch.max(outputs.data, 1)

                for label, prediction in zip(labels.tolist(), predictions.tolist()):
                    if label == prediction:
                        correct_pred[self.reversed_classes[label]] += 1
                    total_pred[self.reversed_classes[label]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
