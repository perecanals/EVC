import torch.nn.functional as F

from torch_geometric.nn import GraphUNet

def get_model(args, dataset_description, device = "cpu"):
    """
    Define model for the dataset.

    Paramaters
    ----------
    args : argparse.Namespace
        Arguments. Contains:
            - base_model_name : string
                Name of the base model to use.
            - batch_size : int
                Batch size.
            - hidden_channels : int
                Number of hidden channels.
            - depth : int
                Depth of the U-Net.
    dataset_description : dict
        Dictionary containing dataset description. Contains:
            - num_edge_features : int
                Number of edge features.
            - num_edge_classes : int
                Number of edge classes.
    device : string, optional
        Device to use for training. The default is "cpu".

    Returns
    -------
    model : torch.nn.Module
        Model to train.
    model_name : string
        String with the model name, where data (train and test) will be 
        in os.path.join(root, "models", model_name).
    """
    model_name = "{}_bs-{}_hc-{}_d-{}_rs-{}".format(args.base_model_name, args.batch_size, args.hidden_channels, args.depth, args.random_state)
    if args.tag is not None:
        model_name += "_{}".format(args.tag)

    print("------------------------------------------------ Model information")
    print(f"Training model:                 {args.base_model_name}")
    print(f"Batch size:                     {args.batch_size}")
    print(f"Hidden channels:                {args.hidden_channels}")
    print(f"Network depth:                  {args.depth} ")

    # Initialize model with the corresponding parameters
    if args.base_model_name == "GraphUNet":
        model = GraphUNet(
            in_channels=dataset_description["num_edge_features"], 
            hidden_channels=args.hidden_channels, 
            out_channels=dataset_description["num_edge_classes"], 
            depth=args.depth,
            pool_ratios=0.5, 
            sum_res=True, 
            act=F.relu).to(device
            )
        
    return model, model_name