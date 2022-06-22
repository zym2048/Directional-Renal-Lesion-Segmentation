import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """

    二值交叉熵损失函数
    """

    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        return self.bce_loss(pred, target)


class DiceLoss(nn.Module):
    """

    Dice loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 1)


class ELDiceLoss(nn.Module):
    """
    Exponential Logarithmic Dice loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    """

    Dice loss + BCE loss
    """

    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # dice系数的定义
        dice = 2 * (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离 +　二值化交叉熵损失
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    """

    Jaccard loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # jaccard系数的定义
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / (pred.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                            target.pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        return torch.clamp((1 - dice).mean(), 0, 1)


class SSLoss(nn.Module):
    """

    Sensitivity Specificity loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = pred.squeeze(dim=1)

        smooth = 1

        # jaccard系数的定义
        s1 = ((pred - target).pow(2) * target).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + target.sum(dim=1).sum(dim=1).sum(dim=1))

        s2 = ((pred - target).pow(2) * (1 - target)).sum(dim=1).sum(dim=1).sum(dim=1) / (smooth + (1 - target).sum(dim=1).sum(dim=1).sum(dim=1))

        # 返回的是jaccard距离
        return (0.05 * s1 + 0.95 * s2).mean()


class TverskyLoss(nn.Module):
    """

    Tversky loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(dim=1)

        smooth = 1

        # print("pred", pred.shape)
        # print("target", target.shape)
        # dice系数的定义
        dice = (pred * target).sum(dim=1).sum(dim=1).sum(dim=1) / ((pred * target).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                                   0.3 * (pred * (1 - target)).sum(dim=1).sum(
                    dim=1).sum(dim=1) + 0.7 * ((1 - pred) * target).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        return torch.clamp((1 - dice).mean(), 0, 2)


class WCELoss(nn.Module):
    """

    加权交叉熵损失函数
    统计了一下训练集下的正负样本的比例，接近20:1
    """
    def __init__(self):
        super().__init__()
        weight = torch.FloatTensor([0.05, 1]).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight)

    def forward(self, pred, target):
        pred_ = torch.ones_like(pred) - pred
        pred = torch.cat((pred_, pred), dim=1)

        target = torch.long()

        return self.ce_loss(pred, target)