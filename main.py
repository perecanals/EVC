
def main(root, args):
    import os, json

    from extracranial_vessel_labelling.data.data_augmentation import get_transforms
    from extracranial_vessel_labelling.data.dataloading import get_data_loaders
    from extracranial_vessel_labelling.models.models import get_model
    from extracranial_vessel_labelling.train.train import run_training
    from extracranial_vessel_labelling.test.test import run_testing

    import torch

    print("------------------------------------------------")
    print("Running training and testing for extracranial vessel labelling\n")

    # Read device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #################################### Dataset organization ############################################
    # Define pre-transforms (applied to the graph before batching, regardless of training or testing)
    pre_transform, train_transform, test_transform = get_transforms(device)
    for fold in (range(args.folds) if args.folds is not None else [None]):
        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(root, args, fold, pre_transform, train_transform, test_transform)

        with open(os.path.join(root, "dataset.json")) as f:
            dataset_description = json.load(f)

        #################################### Model definition ################################################
        model, model_name = get_model(args, dataset_description, device)
        
        if args.train:
            # Define loss function
            if dataset_description["edge_class_frequencies"] is not None:
                loss_function = torch.nn.CrossEntropyLoss(weight = 1 / torch.tensor(dataset_description["edge_class_frequencies"], dtype=torch.float).to(device))
            else:
                loss_function = torch.nn.CrossEntropyLoss()
            # Run training
            run_training(
                root,
                model,
                model_name,
                train_loader,
                val_loader,
                loss_function,
                total_epochs = args.total_epochs,
                learning_rate = args.learning_rate,
                lr_scheduler = args.lr_scheduler,
                device = device,
                fold = fold
                )

        if args.test:
            # Run testing
            run_testing(
                root,
                model_name,
                test_loader,
                model = None,
                device = device,
                fold = fold
            )

if __name__ == "__main__":
    import os, sys
    import argparse
    sys.path.append("/path/to/EVC")

    root = os.environ["EVC_root"]

    # Create argument parser
    parser = argparse.ArgumentParser(description='Train and test the Graph U-Net model.')
    # Add arguments
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, 
        help='Dataset ratio for testing, with respect to the total size of the dataset. Default is 0.2.')
    parser.add_argument('-vs', '--val_size', type=float, default=0.2,
        help='Dataset ratio for validation, with respect to the size of the training dataset (after subtracting testing set)'
        'It will be overrun if args.folds is not None (size will be (1 - test_size) / args.folds). Default is 0.2.')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, 
        help='Batch size for training and validation. Default is 32.')
    parser.add_argument('-bnm', '--base_model_name', type=str, default="GraphUNet", 
        help='Base model name. Default is GraphUNet.')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=64, 
        help='Number of hidden channels. Default is 64.')
    parser.add_argument('-d', '--depth', type=int, default=3, 
        help='Depth of the U-Net. Default is 3.')
    parser.add_argument('-te', '--total_epochs', type=int, default=500, 
        help='Total number of epochs. Default is 500.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, 
        help='Initial learning rate. Default is 0.01.')
    parser.add_argument('-lrs', '--lr_scheduler', type=str, default=True, choices=['True', 'False'],
        help='Learning rate scheduler. Default is True.')
    parser.add_argument('-rs', '--random_state', type=int, default=42,
        help='Random state for splitting the dataset. Default is 42.')
    parser.add_argument('-f', '--folds', type=int, default=None,
        help='Folds number. Default is None.')
    parser.add_argument('-train', '--train', type=str, default=True, choices=['True', 'False'],
        help='Whether to train the model. Default is True.')
    parser.add_argument('-test', '--test', type=str, default=True, choices=['True', 'False'],
        help='Whether to test the model. Default is True.')
    parser.add_argument('-tag', '--tag', type=str, default=None,
        help='Additional tag to add to the model name for identification. Default is None.')

    # Parse arguments
    args = parser.parse_args()
    
    main(root, args)