import os

import numpy as np

from extracranial_vessel_labelling.utils.metrics import compute_accuracy

import torch

def run_testing(root, model_name, test_loader, model = None, device = "cpu", fold = None):
    """
    Performs testing of a model over a test set. If the model is not input, it loads the 
    best model (model_best.pth) from the corresponding model dir.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model_name : string
        Name of the model.
    test_loader : torch_geometric.loader.DataLoader
        DataLoader for testing.
    model : torch.nn.Module, optional
        Model to test. The default is None, which means that it will 
        load the `model_best.pth` from the model_path.
    device : string, optional
        Device to use for testing. The default is "cpu".
    fold : int, optional
        Fold number. The default is None.
    """
    def test_step(model, graph):
        # Set model in evaluation mode
        model.eval()
        # In validation we do not keep track of gradients
        with torch.no_grad():
            # Perform inference with single graph
            pred = model(graph.x.to(device), graph.edge_index.to(device)).argmax(dim=1).to("cpu")
        # Get label from graph
        label = graph.y
        # Compute testing accuracy
        acc = compute_accuracy(pred, graph.y)
        return pred, label, acc
    # Define model path
    model_path = os.path.join(root, "models", model_name)
    if fold is not None:
        model_path = os.path.join(model_path, f"fold_{fold}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Performs testing with best model (minimum validation loss)
    if model is None:
        model = torch.load(os.path.join(model_path, "model_best.pth"))
    if not os.path.isdir(os.path.join(model_path, "test")): os.mkdir(os.path.join(model_path, "test"))
    if not os.path.isdir(os.path.join(model_path, "test", "labels")): os.mkdir(os.path.join(model_path, "test", "labels"))
    if not os.path.isdir(os.path.join(model_path, "test", "preds")): os.mkdir(os.path.join(model_path, "test", "preds"))
    preds, labels, accuracy_test = [], [], []
    for graph in test_loader:
        pred, label, acc = test_step(model, graph)
        preds.append(pred)
        labels.append(label)
        accuracy_test.append(acc)
        np.save(os.path.join(model_path, "test", "labels", f"{graph.raw_file_path[0].split('.')[0]}.npy"), label)
        np.save(os.path.join(model_path, "test", "preds", f"{graph.raw_file_path[0].split('.')[0]}.npy"), pred)

    # Saves testing metrics in test directory
    np.savetxt(os.path.join(model_path, "test", "accuracy.out"), accuracy_test)
    np.savetxt(os.path.join(model_path, "test", "accuracy_mean.out"), [np.mean(accuracy_test), np.std(accuracy_test)])

    print(f"Testing accuracy (lowest validation loss) was {np.mean(accuracy_test):.4f}")
    print()