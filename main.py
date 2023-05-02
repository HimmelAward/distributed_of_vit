import os.path
import random
import torch.nn as nn
import torch
import config
import itertools
import numpy as np
from models import get_model
from dataset import get_dataloader2


class Trainer:
    def __init__(self, models, modelsY, optimizers, critrions, train_loader, test_loader):
        self.models = models
        self.modelsY = modelsY
        self.optimizers = optimizers
        self.critrions = critrions
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrain = True
        self.models_grad, self.modelsY_grad = [[] for n in range(config.NMODELS)], [[] for n in range(config.NMODELS)]
        self.train_accuracy, self.test_accuracy = [], []
        self.train_loss, self.test_loss = [], []
        self.best_acc = 0
        self.fi = open("./res.txt", "w+")

    def accurate(self, pre_data, labels):
        pre, target = pre_data.cpu(), labels.cpu()
        pred = torch.max(pre.data, 1)[1]
        rights = torch.eq(pred, target.data.view_as(pred)).sum()
        return 100 * rights / len(target)

    def get_noise(self,x=1e-3, y=1e-3):
        a = 1.9
        b = c = 1
        for n in range(10000):
            x = a * x * (1 - b * y ** 2)
            y = c * x

        return torch.Tensor(float(random.choice([x, y]))).cuda()

    def update_demo(self, index, models: list, modelsY: list, configure):
        # 拉普拉斯噪声
        # print("进行第{}模型的中间层聚合".format(index))
        inter_layer = models[0].state_dict()
        ar = configure.flush_weight()['ar'][index]
        ac = configure.flush_weight()['ac'][index]
        for name in modelsY[index].state_dict().keys():
            if "heads" not in str(name):
                break

            # X
            for node, rate in ar:
                modelsY[index].state_dict()[name] -= rate * (models[node].state_dict()[name])
            # Y
            for node, rate in ac:
                modelsY[index].state_dict()[name] += rate * (modelsY[node].state_dict()[name])
            #   Y +1                                    X                                     e*Y
            modelsY[index].state_dict()[name] += (models[index].state_dict()[name]) - config.E * (
                (modelsY[index].state_dict()[name]))

        for i, name in enumerate(inter_layer.keys()):
            if "heads" not in str(name):
                break
            inter_value = models[ac[0][0]].state_dict()[name]
            for node, rate in ar[1:]:
                inter_value += rate * (models[node].state_dict()[name])
            #                                             e*y                                       b*gradX
            inter_layer[name] = inter_value + config.E * (modelsY[index].state_dict()[name])

        for name in inter_layer.keys():
            if "heads" not in str(name):
                break
            models[index].state_dict()[name] = inter_layer[name]

    def update_grad_from_loss(self, index, pre, label):
        """
        获得对应loss函数的loss并存到梯度中
        :param index: 第几个模型的loss
        :param pre: 中间层 Xi
        :return:
        """
        update_grads = []
        noise = self.get_noise()
        loss_1 = self.critrions[index](pre + config.Y * noise, label)
        loss_2 = self.critrions[index](pre - config.Y * noise, label)
        for layer_grad in self.models_grad[index]:
            if layer_grad == None:
                break
            # 这里获得损失因为自动变为一维向量，而一维向量无法直接赋值
            loss = (loss_1 - loss_2) * noise * config.Y * config.DIM_NODE
            # 梯度为张量 loss 为数，所以要填充
            updated_grad = layer_grad.fill_(loss[0])
            update_grads.append(updated_grad)
        self.models_grad[index] = update_grads

    def get_grad(self, index):
        for name in self.models[index].parameters():
            self.models_grad[index].append(name.grad)
            if self.pretrain:
                self.modelsY_grad[index].append(name.grad)

    def put_grad(self, index):
        for name, new_grad in itertools.zip_longest(self.models[index].parameters(), self.models_grad[index]):
            name.grad = new_grad

    def _run_batch(self, index, data, label, batch):
        if batch % 200 == 0:
            print("这是第{}个模型：{}批次训练".format(index, batch))
        if self.pretrain:
            pre = self.models[index](data)
            loss = self.critrions[index](pre, label)
            self.optimizers[index].zero_grad()
            loss.backward()
            self.optimizers[index].step()
            self.get_grad(index)
            if index == 9:
                self.pretrain = False
                print("=======预训练结束=======")
        else:
            # self.get_grad(index)
            if batch % 200 == 0:
                for index in random.choices([n for n in range(10)], k=5):
                    print("updating ...............................................")
                    self.fi.write("updating ..............................................." + '\n')
                    self.update_demo(index=index, models=self.models, modelsY=self.modelsY, configure=config.configure)

            pre = self.models[index](data)
            loss = self.critrions[index](pre, label)
            self.update_grad_from_loss(index, pre, label)
            self.optimizers[index].zero_grad()
            loss.backward()
            self.optimizers[index].step()
            self.put_grad(index)
            self.train_accuracy.append(self.accurate(pre, label))
            self.train_loss.append(loss.item())

    def _run_epoch(self, epoch):
        print("这是第{}轮训练".format(epoch))
        for batch, (data, label) in enumerate(self.train_loader):
            data, label = data.cuda(), label.cuda()
            if self.pretrain:
                random_models = [n for n in range(len(self.models))]
            else:
                random_models = random.choices([n for n in range(len(self.models))], k=3)
            for index in random_models:
                self._run_batch(index, data, label, batch)
                if len(self.train_accuracy) != 0 and batch % 200 == 0:
                    msg = "Epoch:{} Model:{} train_accuracy:{}% train_loss:{}"
                    train_accuracy = sum(self.train_accuracy) / len(self.train_accuracy)
                    train_loss = sum(self.train_loss) / len(self.train_loss)
                    print(msg.format(epoch, index, train_accuracy, train_loss))
                    self.fi.write(msg.format(epoch, index, train_accuracy, train_loss) + '\n')
                    if self.best_acc <= train_accuracy:
                        torch.save(self.models[index].state_dict(), "./models/best.pth")

    def train(self):
        if self.pretrain:
            print("=======预训练开始=======")
        for epoch in range(config.EPOECHS):
            self._get_init_weight()
            self._run_epoch(epoch=epoch)
        for index, model in enumerate(self.models):
            self.show_update(index, model)

    def _get_init_weight(self):
        if os.path.exists('./models/best.pth'):
            for index in range(len(self.models)):
                self.models[index].load_state_dict(torch.load('./models/best.pth'), strict=False)

    def show_update(self, index, model):
        """
        展示最终优化结果
        :param index: 模型索引
        :param model: 模型
        :return:
        """
        criterion = self.critrions[index]
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.cuda(), label.cuda()
                res = model(data)
                loss = criterion(res, label)
                res, label = res.cpu(), label.cpu()
                pred = torch.max(res.data, 1)[1]
                rights = torch.eq(pred, label.data.view_as(pred)).sum()
                self.test_loss.append(loss.item())
                self.test_accuracy.append(100 * rights / len(label))
        test_loss = sum(self.test_loss) / len(self.test_loss)
        test_accuracy = sum(self.test_accuracy) / len(self.test_accuracy)
        print("经过参数更新后,##模型{}的 loss: {:.3f} accuracy: {:.3f}%".format(index, test_loss, test_accuracy))


nmodels, nmodelsY = [get_model(10).cuda() for _ in range(config.NMODELS)], \
                    [get_model(10).cuda() for _ in range(config.NMODELS)]
critrions = [nn.CrossEntropyLoss() for _ in range(config.NMODELS)]
optimizers = [torch.optim.Adam(model.parameters(), lr=3e-1) for model in nmodels]
train_loader, test_loader = get_dataloader2()

trainer = Trainer(models=nmodels,
                  modelsY=nmodelsY,
                  optimizers=optimizers,
                  critrions=critrions,
                  train_loader=train_loader,
                  test_loader=test_loader)

trainer.train()
trainer.fi.close()