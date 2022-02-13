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
            # also consider numerical instability
            total_loss += self.weight_per_instrument[
                i_instrument
            ] * F.binary_cross_entropy(
                predictions[:, i_instrument, :] + 1e-10,
                labels[:, i_instrument, :] + 1e-10,
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
        difference_matrix = torch.abs(predictions - labels)

        # clamp values s.t. we don't hit numerical stability
        with torch.no_grad():
            difference_matrix = torch.clamp(difference_matrix, min=1e-8)

        # compute frobenius norm of difference matrix per batch
        frobenius_norm = torch.linalg.norm(difference_matrix, dim=(1, 2), ord="fro")

        # return average frobenius norm
        return torch.mean(frobenius_norm)


class CompressedSpectralLoss(nn.Module):
    def __init__(self, loudness_exponent: float = 0.6):
        super(CompressedSpectralLoss, self).__init__()

        self.loudness_exponent = loudness_exponent

    def forward(self, predictions, labels):
        """Computes and returns the loss.

        Expected input shape(s): [n_batches, n_instruments, n_time_frames]
        """
        # compute absolute difference
        p = torch.abs(predictions + 1e-08)
        l = torch.abs(labels + 1e-08)

        # clamp values s.t. we don't hit numerical stability
        with torch.no_grad():
            p = torch.clamp(p, min=1e-08)
            l = torch.clamp(l, min=1e-08)

        p = torch.pow(p, self.loudness_exponent)
        l = torch.pow(l, self.loudness_exponent)

        diff_matrix = torch.pow(torch.abs(p - l), 2.0)

        return torch.mean(diff_matrix)
