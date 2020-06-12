
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score


def metric_segmentation(outputs, targets, metric, background=None, SMOOTH=1e-6, device='cpu'):
    assert len(outputs.shape) in [3, 4], ('outputs must have shape (batch_size, N_classes, width, length) or (batch_size, width, length).'
                                     'Current Length: {}'.format(outputs.shape))
    if len(outputs.shape) == 4:
        pred_labels = outputs.argmax(1)
    else:
        pred_labels = outputs

    labels = np.unique(targets.cpu())
    if background is not None:
        labels = list(set(labels).difference(background))
    labels.sort()

    metric_labels = []
    for label in labels:
        true = (targets == label).to(device)
        preds = (pred_labels == label).to(device)

        metric_labels.append(
            metric(preds, true).unsqueeze(1)
        )

    return torch.cat(metric_labels, 1).cpu().numpy()


# Taken from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
def iou(outputs_orig, targets_orig, SMOOTH=1e-6,):
    y_true = (targets_orig != 0)
    y_pred = outputs_orig != 0
    intersection = (y_pred & y_true).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (y_pred | y_true).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou.detach().cpu()


def dice(outputs_orig, targets_orig, SMOOTH=1e-6,):

    targets = (targets_orig != 0)
    outputs = (outputs_orig != 0)

    intersection = (outputs & targets).float().sum((1, 2))

    return (
        (2*intersection + SMOOTH) / (targets.sum((1, 2)) + outputs.sum((1, 2)) + SMOOTH)
    ).detach().cpu()


def mcc_quotient(tp, tn, fp, fn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        den = 1
    return (tp * tn - fp * fn) / np.sqrt(den)


def matthewscc_custom(y_pred, y_true):
    """Computes Matthew Correlation Coefficient.

    Args:
        outputs_orig (torch.tensor): arbitrary size. Tensor of 0s and 1s. Predictions.
        targets_orig (torch.tensor): same size as outputs_orig. Tensor of 0s and 1s. True values.

    Returns:
        torch.tensor: tensor size (0) of accuracy.
    """

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return mcc_quotient(tp, tn, fp, fn)



def matthewscc(y_pred, y_true):
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.cpu().numpy()

    if type(y_true) == torch.Tensor:
        y_true = y_true.cpu().numpy()

    if y_pred.max() <= 1 and y_true.max() <= 1:
        return matthewscc_custom(y_pred, y_true)

    return matthews_corrcoef(y_pred, y_true)


def accuracy(outputs_orig, targets_orig):
    """Computes accuracy.

    Args:
        outputs_orig (torch.tensor): arbitrary size. Tensor of 0s and 1s. Predictions.
        targets_orig (torch.tensor): same size as outputs_orig. Tensor of 0s and 1s. True values.

    Returns:
        torch.tensor: tensor size (0) of accuracy.
    """
    y_pred = outputs_orig + 0.
    y_true = targets_orig + 0.
    if type(y_pred) == torch.Tensor:
        return (y_pred - y_true == 0).float().mean().detach().item()
    return np.mean(y_pred - y_true == 0)


def f1score_custom(outputs_orig, targets_orig):
    """Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Args:
        outputs_orig (torch.tensor): arbitrary size. Tensor of 0s and 1s. Predictions.
        targets_orig (torch.tensor): same size as outputs_orig. True values.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    """
    y_true = targets_orig + 0
    y_pred = outputs_orig + 0

    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()

    if type(tp) == torch.Tensor:
        tp = tp.to(torch.float32).detach()
        fp = fp.to(torch.float32).detach()
        fn = fn.to(torch.float32).detach()

    if (tp + fp) * (tp + fn) == 0:
        epsilon = 1e-7
    else:
        epsilon = 0

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1


def f1score(y_pred, y_true, average='macro', *args, **kwargs):

    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.cpu().numpy()

    if type(y_true) == torch.Tensor:
        y_true = y_true.cpu().numpy()

    if y_pred.max() <= 1 and y_true.max() <= 1:
        return f1score_custom(y_pred, y_true)

    return f1_score(y_pred, y_true, average=average, *args, **kwargs)


def metric_binary_thresh(outputs_orig, targets_orig, metric, threshold=.5):
    """
    Thresholds the outputs and applies the metric.

    Args:
        outputs_orig (torch.tensor): Predictions between 0 and 1, not thresholded.
                                     Size: (batch_size, nb_classes).
        targets_orig (torch.tensor): True values: 0 or 1. Same size as outputs_orig.
        metric (function): Metric that accepts as inputs binary tensors.
        threshold (float, optional): Threshold to define positive in binary classification.
                                     Defaults to .5.

    Returns:
        float: returns outputs of metrics for the two tensors, the first one thresholded.
    """

    y_true = (targets_orig + 0).squeeze()
    y_pred = (outputs_orig + 0).squeeze()

    if y_true.ndim == 2:
        y_true = y_true.argmax(1)

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    # print(y_pred)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        # y_pred = y_pred[:, 1]

    elif y_pred.ndim == 1 and not set(np.unique(y_pred)).issubset([0, 1]):
        y_pred = y_pred > threshold

    return metric(y_pred, y_true)


def metric_multiclass(outputs_orig, targets_orig, metric):

    y_true = (targets_orig + 0).squeeze()
    y_pred = (outputs_orig + 0).squeeze()

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(1)

    if y_true.ndim == 2:
        y_true = y_true.argmax(1)

    return metric(y_pred, y_true)
