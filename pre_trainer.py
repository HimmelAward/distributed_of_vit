import torch.nn as nn
import torch.optim as optim
from models import get_model
import config
import os
import itertools
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import copy
from dataset import get_dataloader
from torch.distributed import init_process_group, destroy_process_group
if not os.path.exists(config.IMG_PATH) :
    os.mkdir(config.IMG_PATH)
if (not os.path.exists(config.SAVE_PTH)):
    os.mkdir(config.SAVE_PTH)
init_process_group(backend='nccl')
train,test  = get_dataloader()


def accurate( pre_data, labels):
    pre, target = pre_data.cpu(), labels.cpu()
    pred = torch.max(pre.data, 1)[1]
    rights = torch.eq(pred, target.data.view_as(pred)).sum()
    return 100 * rights / len(target)


def trainer(pre_trained):
    model = get_model(10)
    model = DDP(model.to(torch.device("cuda:{}".format(os.environ["LOCAL_RANK"]))), device_ids=[int(os.environ["LOCAL_RANK"])])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=7e-3, momentum=0.99)
    if pre_trained:
        model.load_state_dict(torch.load("./models/best.pth"),strict=False)

    best_acc = 0
    best_epoch = 0
    for epoch in range(50):
        acc = []
        for data, label in train:
            data,label = data.to(torch.device("cuda:{}".format(os.environ["LOCAL_RANK"]))),label.to(torch.device("cuda:{}".format(os.environ["LOCAL_RANK"])))
            pre = model(data)
            loss = criterion(pre,label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            acc.append(accurate(pre,label))

        accuracy = sum(acc)/len(acc)
        if best_acc < accuracy:
            best_acc = accuracy
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "./models/best.pth")


        print('Epoch:', '{}'.format(epoch + 1), 'acc =', '{:.6f}'.format(accuracy))
    print('best_loss: ', best_acc, '  best_epoch:', best_epoch)


trainer(pre_trained=True)
destroy_process_group()