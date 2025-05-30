import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    """
    Loss function that combines losses calculated at multiple scales.

    Args:
        loss_criterion (nn.Module): The base loss function to use at each scale
                                     (e.g., nn.BCEWithLogitsLoss, nn.MSELoss).
        weights (list or tuple, optional): Weights to apply to the loss at each scale.
                                           Should have the same length as the number of decoder outputs
                                           (including the final output). Defaults to None (equal weights).
    """

    def __init__(self, loss_criterion, weights=None):
        super().__init__()
        self.loss_criterion = loss_criterion
        self.weights = weights

    def forward(self, outputs, target):
        """
        Calculates the multiscale loss.

        Args:
            outputs (list of torch.Tensor): A list of output tensors from the decoder,
                                            where each tensor represents a different scale.
                                            The last tensor in the list is typically the
                                            finest resolution output.
            target (torch.Tensor): The ground truth target tensor. It will be compared
                                   to each of the scaled outputs.

        Returns:
            torch.Tensor: The combined multiscale loss.
        """
        num_outputs = len(outputs)
        total_loss = 0

        if self.weights is None:
            weights = [1.0] * num_outputs
        else:
            if len(self.weights) != num_outputs:
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match the number of outputs ({num_outputs})."
                )
            weights = self.weights

        for i, output in enumerate(outputs):
            scaled_target = F.interpolate(
                target, size=output.shape[2:], mode="bilinear", align_corners=False
            )

            loss = self.loss_criterion(output, scaled_target)
            total_loss += weights[i] * loss

        return total_loss
