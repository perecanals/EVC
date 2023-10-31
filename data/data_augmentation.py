from torch_geometric.transforms import Compose, RadiusGraph, ToDevice

def get_transforms(device):
    """
    Define transforms for the dataset.

    Paramaters
    ----------
    device : string
        Device to use for training.
    
    Returns
    -------
    pre_transforms : torch_geometric.transforms.Compose
        Pre-transforms to apply to the dataset.
    train_transforms : torch_geometric.transforms.Compose
        Dynamic transforms to apply for data augmentation during training.
    test_transforms : torch_geometric.transforms.Compose
        Dynamic transforms to apply during testing.
    """
    # Define transforms
    pre_transforms = Compose([
        RadiusGraph(r = 0.5, max_num_neighbors = 10),
        ToDevice(device)
    ])
    train_transforms = Compose([
        ToDevice(device)
    ])
    test_transforms = Compose([
        ToDevice(device)
    ])

    return pre_transforms, train_transforms, test_transforms