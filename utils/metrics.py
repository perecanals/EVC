

def compute_accuracy(pred, label):
    """
    Computes accuracy for a batch of graphs.
    
    Parameters
    ----------
    pred : torch.tensor
        Tensor with predicted labels.
    label : torch.tensor
        Tensor with ground truth labels.

    Returns
    -------
    accuracy : float
        Accuracy for the batch.
    """
    accuracyList = []
    for node, _ in enumerate(pred):
        if pred[node].item() == label[node].item():
            accuracyList.append(1)
        else:
            accuracyList.append(0)
    return sum(accuracyList) / len(accuracyList)