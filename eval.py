import torch


class Eval:
    def __init__(self, testloader, device) -> None:
        self._testloader = testloader
        self.device = device

    def eval(self, model) -> int:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self._testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                for i in range(len(labels)):
                    if torch.max(labels[i], 0)[1] == predicted[i]:
                        correct += 1

        return 100 * correct / total
