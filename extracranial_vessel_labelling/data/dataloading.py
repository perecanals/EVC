import os

from extracranial_vessel_labelling.data.dataset import EVCDataset

from sklearn.model_selection import train_test_split, KFold

from torch_geometric.loader import DataLoader

def get_data_loaders(root, args, fold = None, pre_transform = None, train_transform = None, test_transform = None):
    """
    Define train, validation and test data loaders for the dataset.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    args : argparse.Namespace
        Arguments.
    fold : int
        Fold for cross validation. By default is None.
    pre_transform : torch_geometric.transforms.Compose, optional
        Pre-transforms to apply to the dataset. The default is None.
    train_transform : torch_geometric.transforms.Compose, optional
        Dynamic transforms to apply for data augmentation during training, optional
    test_transforms : torch_geometric.transforms.Compose, optional
        Dynamic transforms to apply during testing, optional

    Returns
    -------
    t
    """
    # Define datasets . First make division of train + val and test
    dataset_filenames = [f for f in os.listdir(os.path.join(root, "raw")) if f.endswith(".pickle")]
    if not args.test_size == args.val_size == 0:
        train_val_filenames, test_filenames = train_test_split(dataset_filenames, test_size = args.test_size, train_size = 1 - args.test_size, random_state = args.random_state)
        # Now, depending on the folds being specified or not, perform cross validation (select the k-fold corresponding to "fold", with k being "args.folds") or not
        if fold is not None:
            kf = KFold(n_splits = args.folds)
            folds = list(kf.split(train_val_filenames))
            train_filenames = [train_val_filenames[i] for i in folds[fold][0]]
            val_filenames = [train_val_filenames[i] for i in folds[fold][1]]    
        else:
            train_filenames, val_filenames = train_test_split(train_val_filenames, test_size = args.val_size, train_size = 1 - args.val_size, random_state = args.random_state)
        # Now define dataset classes
        train_dataset = EVCDataset(root, raw_file_names_list = train_filenames, pre_transform = pre_transform, transform = train_transform)
        val_dataset = EVCDataset(root, raw_file_names_list = val_filenames, pre_transform = pre_transform, transform = test_transform)
        test_dataset = EVCDataset(root, raw_file_names_list = test_filenames, pre_transform = pre_transform, transform = test_transform)
    else:
        print("Training with the whole dataset (ignore validation and test results)\n")
        train_dataset = EVCDataset(root, pre_transform = pre_transform, transform = train_transform)
        val_dataset = EVCDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform)
        test_dataset = EVCDataset(root, raw_file_names_list = dataset_filenames[:2], pre_transform = pre_transform, transform = test_transform)

    print("------------------------------------------------ Dataset information")
    print("Total number of samples:        {}".format(len(dataset_filenames)))
    if fold is not None:    
        print(f"Fold:                           {fold}")
    print("Number of training samples:     {} ({:.2f}%)".format(len(train_dataset), 100 * len(train_dataset) / len(dataset_filenames)))
    print("Number of validation samples:   {} ({:.2f}%)".format(len(val_dataset), 100 * len(val_dataset) / len(dataset_filenames)))
    print("Number of testing samples:      {} ({:.2f}%)\n".format(len(test_dataset), 100 * len(test_dataset) / len(dataset_filenames)))

    # Define loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader