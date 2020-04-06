
import torch



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
    return (y_pred - y_true == 0).float().mean().detach()


def f1score(outputs_orig, targets_orig):
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

    

        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.detach()


def metric_argmaxer(outputs_orig, targets_orig, metric, threshold=.5):
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

    if y_true.ndim > 1:
        y_true = y_true.argmax(1)

    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    # print(y_pred)
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        # y_pred = y_pred[:, 1]

    elif y_pred.ndim == 1:
        y_pred = y_pred > threshold

    return metric(y_pred, y_true)