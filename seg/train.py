import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import seg.Loss as Loss
import seg.Parameter as para
from seg.Dataset import Dataset, RandomSampler

import seg.Network as Network
from seg.Network import kiunet_min, unet, kiunet_org, unet_min, ResUNet

if __name__ == '__main__':

    # 设置环境
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
    cudnn.benchmark = para.cudnn_benchmark

    # 初始化网络
    net = unet(training=True)
    net.apply(Network.init)
    net = net.cuda()
    net.train()
    print('net total parameters:', sum(param.numel() for param in net.parameters()))
    print("Finish prepare Net.")

    # 准备数据集

    train_ds = Dataset(para.train_ct_path, para.train_seg_path)
    train_sampler = RandomSampler(data_source=train_ds, num_samples=200)
    train_dl = DataLoader(train_ds, para.batch_size, shuffle=False, num_workers=para.num_workers,
                          pin_memory=para.pin_memory, sampler=train_sampler)

    test_ds = Dataset(para.test_ct_path, para.test_seg_path)
    test_sampler = RandomSampler(data_source=test_ds, num_samples=200)
    test_dl = DataLoader(test_ds, para.batch_size, False, num_workers=para.num_workers,
                         pin_memory=para.pin_memory, sampler=test_sampler)
    print("Finish prepare Data.")


    # 准备损失函数及优化器
    loss_func = Loss.TverskyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)
    alpha = para.alpha

    train_loss = []
    val_loss = []
    val_loss_min = 1

    print("Begin to Train...")
    start = time()
    for epoch in range(para.Epoch):
        torch.cuda.empty_cache()
        mean_loss = []
        loss = 0

        for step, (ct, seg) in enumerate(train_dl):
            ct = ct.cuda()
            seg = seg.cuda()
            outputs = net(ct)

            loss1 = loss_func(outputs[0], seg)
            loss2 = loss_func(outputs[1], seg)
            loss3 = loss_func(outputs[2], seg)
            loss4 = loss_func(outputs[3], seg)

            loss = (loss1 + loss2 + loss3) * alpha + loss4

            mean_loss.append(loss4.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 5 is 0:
                print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                      .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

        mean_loss = sum(mean_loss) / len(mean_loss)

        print(f"epoch {epoch+1}: begin to test...")
        torch.cuda.empty_cache()
        net.eval()
        test_loss = []
        with torch.no_grad():
            for step, (ct, seg) in enumerate(test_dl):
                ct = ct.cuda()
                seg = seg.cuda()
                output = net(ct)
                test_loss.append(loss_func(output, seg).item())
            test_loss = sum(test_loss) / len(test_loss)
        net.train()
        print(f"Train Loss: {mean_loss}, Val Loss: {test_loss}")
        train_loss.append(mean_loss)
        val_loss.append(test_loss)
        if test_loss < val_loss_min:
            val_loss_min = test_loss
            torch.save(net.state_dict(),
                       'best_epoch_net{}-{:.3f}-{:.3f}-{:.3f}.pth'.format(epoch + 1, loss,
                                                                                                            mean_loss,
                                                                                                            test_loss))

        if (epoch + 1) % 5 is 0:
            torch.save(net.state_dict(), 'net{}-{:.3f}-{:.3f}-{:.3f}.pth'.format(epoch+1, loss, mean_loss, test_loss))

        if (epoch+1) % 40 is 0 and epoch is not 0:
            alpha *= 0.8


        lr_decay.step()

    import pickle
    with open('unet.pkl', 'wb') as f:
        pickle.dump({"train_loss": train_loss, "val_loss": val_loss}, f, pickle.HIGHEST_PROTOCOL)