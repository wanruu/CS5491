import torch
import numpy as np


class evalMatrix:
    def __init__(self, clses, device, Matrix=None):
        if Matrix is None:
            self.Matrix = torch.from_numpy(np.zeros((clses, clses))).to(device)
        else:
            self.Matrix = Matrix
        self.device = device
        self.clses = clses
        self.acc = -1
        self.recall = -1
        self.precision = -1
        self.f1 = -1
        self.kappa = -1
        # print(self.Matrix)

    def record(self, out, y):
        pred = torch.argmax(out, dim=1)
        y = y.T.squeeze()
        for i in range(len(pred)):
            # print(pred[i],y[i])
            self.Matrix[y[i].item()-1][pred[i].item()-1] += 1

    def clear(self):
        self.Matrix = torch.from_numpy(np.zeros((self.clses, self.clses))).to(self.device)

    def analysis(self):
        all_number = torch.sum(self.Matrix)
        # print(all_number)

        # get acc
        acc = 0
        for i in range(self.clses):
            acc += self.Matrix[i][i]
        acc = acc / all_number
        self.acc = acc.item()
        # print(acc)

        # get recall
        recall = torch.zeros(self.clses).to(self.device)
        for i in range(self.clses):
            recall[i] = self.Matrix[i][i] / torch.sum(self.Matrix, dim=0)[i]
        self.recall = recall.sum().item() / self.clses

        # print(recall)
        precision = torch.zeros(self.clses).to(self.device)
        for i in range(self.clses):
            precision[i] = self.Matrix[i][i] / torch.sum(self.Matrix, dim=1)[i]
        self.precision = precision.sum().item() / self.clses

        # get f1
        f1 = torch.torch.zeros(self.clses).to(self.device)
        for i in range(self.clses):
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        self.f1 = f1.sum().item() / self.clses
        # print(f1)

        # get kappa
        pe = 0
        for i in range(self.clses):
            pe += torch.sum(self.Matrix, dim=1)[i] * torch.sum(self.Matrix, dim=0)[i]
        pe = pe / (torch.sum(self.Matrix) ** 2)
        kappa = (acc - pe) / (1 - pe)
        self.kappa = kappa.item()
        # print(kappa)

        return self.acc, self.recall, self.precision, self.f1, self.kappa
