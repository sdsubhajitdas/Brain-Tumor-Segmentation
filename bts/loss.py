import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Sørensen–Dice coefficient loss to calculate
    the mean loss over a batch of data.This loss mainly
    calculates the similarity between two samples.
    To know more about this loss check this link:
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    def __init__(self):
        """Simple constructor for the class."""
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        """ Method for calculation of loss from sample.
        Parameters:
            predicted(torch.Tensor): Predicted output of the network.
                                    Shape - (Batch Size,Channel,Height,Width)
            target(torch.Tensor): Actual required output for the network
                                    Shape - (Batch Size,Channel,Height,Width)

        Returns:
            The mean dice Loss over the batch size.
        """
        batch = predicted.size()[0]
        batch_loss = 0
        for index in range(batch):
            coefficient = self._dice_coefficient(
                predicted[index], target[index])
            batch_loss += coefficient

        batch_loss = batch_loss / batch

        return 1 - batch_loss

    def _dice_coefficient(self, predicted, target):
        """Calculates the Sørensen–Dice Coefficient for a
        single sample.
        Parameters:
            predicted(torch.Tensor): Predicted single output of the network.
                                    Shape - (Channel,Height,Width)
            target(torch.Tensor): Actual required single output for the network
                                    Shape - (Channel,Height,Width)

        Returns:
            coefficient(torch.Tensor): Dice coefficient for the input sample.
                                        1 represents high similarity and
                                        0 represents low similarity.
        """
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2*intersection + smooth) / \
            (predicted.sum() + target.sum() + smooth)
        return coefficient


class BCEDiceLoss(nn.Module):
    """ Combination of Binary Cross Entropy Loss and Soft Dice Loss.
    This combined loss is used to train the network so that both
    benefits of the loss are leveraged.
    """

    def __init__(self, device):
        """Simple constructor for the class."""
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss().to(device)

    def forward(self, predicted, target):
        """ Method for calculation of combined loss from sample."""
        return F.binary_cross_entropy(predicted, target) \
            + self.dice_loss(predicted, target)
