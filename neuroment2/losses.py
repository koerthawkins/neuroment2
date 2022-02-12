import logging as log
import torch
from torch import nn
import torch.nn.functional as F


class BinaryCrossentropyPerInstrument(nn.Module):
    def __init__(self, weight_per_instrument=None):
        super(BinaryCrossentropyPerInstrument, self).__init__()

        if weight_per_instrument is not None:
            self.weight_per_instrument = weight_per_instrument
        else:
            log.warning(
                "No weights passed to BinaryCrossentropyPerInstrument.__init__()! Setting all instrument weights to 1.0!"
            )
            self.weight_per_instrument = torch.ones(size=[8])

    def forward(self, predictions, labels):
        """Computes and returns the loss.

        Expected input shape(s): [n_batches, n_instruments, n_time_frames]
        """
        total_loss = 0.0

        for i_instrument in range(predictions.shape[1]):
            # compute the BinaryCrossentropy of only the current instrument and multiply it with
            # its weight
            total_loss += self.weight_per_instrument[
                i_instrument
            ] * F.binary_cross_entropy(
                predictions[:, i_instrument, :],
                labels[:, i_instrument, :],
            )

        return total_loss


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super(FrobeniusLoss, self).__init__()

    def forward(self, predictions, labels):
        """Computes and returns the loss.

        Expected input shape(s): [n_batches, n_instruments, n_time_frames]
        """
        # compute difference matrix
        d = predictions - labels

        # compute frobenius norm
        total_loss = 0.0
        for i_batch in range(predictions.shape[0]):
            total_loss += torch.linalg.norm(d[i_batch, :, :], ord="fro")

        return total_loss
