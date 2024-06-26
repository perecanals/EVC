import os

import numpy as np

from extracranial_vessel_labelling.train.lr_schedulers import PolyLRScheduler
from extracranial_vessel_labelling.train.utils import make_train_plot
from extracranial_vessel_labelling.utils.metrics import compute_accuracy

import torch

def run_training(root, model, model_name, train_loader, val_loader, loss_function, total_epochs = 500, learning_rate = 0.01, lr_scheduler = True, device = "cpu", fold = None):
    """
    Trainer function for a given model. Thought out for GraphUNet for node classification.

    Parameters
    ----------
    root : string or path-like object
        Path to root folder.
    model : torch.nn.Module
        Model to train.
    model_name : string 
        Name of the model.
    train_loader : torch_geometric.loader.DataLoader
        DataLoader for training.
    val_loader : torch_geometric.loader.DataLoader
        DataLoader for validation.
    loss_function : torch.nn.Module
        Loss function to use.
    total_epochs : int, optional
        Number of epochs to train. The default is 500.
    learning_rate : float, optional
        Initial learning rate. The default is 0.01.
    lr_scheduler : bool, optional
        Whether to use a learning rate scheduler. The default is True.
    device : string, optional
        Device to use for training. The default is "cpu".
    node_class_frequencies : list, optional
        Node class frequencies, that will be used for weighting the loss function. The default is None.
    fold : int, optional
        Fold number. The default is None.
    """
    def train_step(model, batch):
        """
        Performs a training step for a batch of graphs.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        batch : torch_geometric.data.Batch
            Batch of graphs.

        Returns
        -------
        loss : torch.tensor
            Loss for the batch.
        """
        # Set model in training mode
        model.train()
        # Set gradients to 0
        optimizer.zero_grad() 
        # Perform forward pass with batch
        out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)).to(device)
        # Compute loss
        loss = loss_function(out, batch.y).to("cpu")
        # Compute back propagation
        loss.backward() 
        # Update weights with optimizer
        optimizer.step()
        # Compute accuracy for training (could be ommited)
        accuracy = compute_accuracy(out.argmax(dim=1), batch.y)
        return loss, accuracy

    def val_step(model, batch):
        """
        Performs a validation step for a batch of graphs.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.
        batch : torch_geometric.data.Batch
            Batch of graphs.

        Returns
        -------
        loss : torch.tensor
            Loss for the batch.
        accuracy : float
            Accuracy for the batch.
        """
        # Set model in evaluation mode
        model.eval()
        # In validation we do not keep track of gradients
        with torch.no_grad():
            # Perform forward pass with batch
            out = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)).to(device)
            # Compute validation loss
            loss = loss_function(out, batch.y).to("cpu")
            # Compute validation accuracy
            accuracy = compute_accuracy(out.argmax(dim=1), batch.y)
        return loss, accuracy
    
    print("\n------------------------------------------------ Training parameters")
    print(f"Total epochs:                   {total_epochs}")
    print(f"Initial learning rate:          {learning_rate}")
    print(f"Learning rate scheduler:        {lr_scheduler}")
    print(f"Running on device:              {device} \n")

    # Define model path
    model_path = os.path.join(root, "models", model_name)
    if fold is not None:
        model_path = os.path.join(model_path, f"fold_{fold}")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate,
        momentum=0.99,
        weight_decay=1e-03
        )
    if lr_scheduler:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min', 
        #     factor=0.1, 
        #     patience=50, 
        #     threshold=0.001, 
        #     threshold_mode='rel', 
        #     eps=1e-06
        #     )
        scheduler = PolyLRScheduler(
            optimizer, 
            initial_lr=learning_rate,
            max_steps=total_epochs,
            exponent=0.9
        )
        
    # Initializes lists for loss and accuracy evolution during training
    losses_train = []
    losses_val = []
    accuracy_train = []
    accuracy_val = []

    # Starts training
    for epoch in range(0, total_epochs + 1):
        # Initializes in-epoch variables
        total_epoch_loss_train, total_epoch_loss_val = 0, 0
        accuracy_train_epoch_list, accuracy_val_epoch_list = [], []
        num_graphs_train, num_graphs_val = 0, 0

        # Iterates over training DataLoader and performs a training step for each batch
        for batch in train_loader:
            # Performs training step
            loss_train, acc_train = train_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_train += loss_train.detach()
            # Adds accuracy to list for epoch
            accuracy_train_epoch_list.append(acc_train)
            # Update the number of graphs
            num_graphs_train += batch.batch.max().item() + 1

        # Divides epoch accumulated training loss by number of graphs
        total_epoch_loss_train = total_epoch_loss_train / num_graphs_train
        # Appends training loss to tracking
        losses_train.append(total_epoch_loss_train)
        # Computes mean training accuracy across batches
        accuracy_train_epoch = np.mean(accuracy_train_epoch_list)
        # Appends training accuracy to tracking
        accuracy_train.append(accuracy_train_epoch)

        # Iterates over validation DataLoader and performs a training step for each batch
        for batch in val_loader:
            # Performs validation step
            loss_val, acc_val = val_step(model, batch)
            # Adds loss to epoch loss
            total_epoch_loss_val += loss_val
            # Adds accuracy to list for epoch
            accuracy_val_epoch_list.append(acc_val)
            # Update the number of graphs
            num_graphs_val += batch.batch.max().item() + 1

        # Divides epoch accumulated validation loss by number of graphs
        total_epoch_loss_val = total_epoch_loss_val / num_graphs_val
        # Appends validation loss to tracking
        losses_val.append(total_epoch_loss_val)
        # Computes mean validation accuracy across batches
        accuracy_val_epoch = np.mean(accuracy_val_epoch_list)
        # Appends validation accuracy to tracking
        accuracy_val.append(accuracy_val_epoch)

        # Saves best model in terms of validation loss
        if total_epoch_loss_val == np.amin(losses_val):
            torch.save(model, os.path.join(model_path, "model_best.pth"))
        # Saves best model in terms of validation accuracy
        if accuracy_val_epoch == np.max(accuracy_val):
            torch.save(model, os.path.join(model_path, "model_best_acc.pth"))

        # Updates learning rate policy if scheduler is used
        if lr_scheduler:
            scheduler.step(total_epoch_loss_val)
        else:
            pass

        # Prints checkpoint every 100 epochs
        if epoch % 100 == 0:
            torch.save(model, os.path.join(model_path, "model_latest.pth"))
            if fold is not None:
                print(f"Epoch: {epoch:03d} (model {model_name}, fold {fold})")
            else:
                print(f"Epoch: {epoch:03d} (model {model_name})")
            print(f"Training loss:       {total_epoch_loss_train:.4f}")
            print(f"Training accuracy:   {accuracy_train_epoch:.4f}")
            print(f"Validation loss:     {total_epoch_loss_val:.4f}")
            print(f"Validation accuracy: {accuracy_val_epoch:.4f}")
            print()
        
            # Make training plot
            make_train_plot(model_path, losses_train, losses_val, accuracy_train, accuracy_val)
    
    print(f"Training finished.\n")
    
    # Make training plot
    make_train_plot(model_path, losses_train, losses_val, accuracy_train, accuracy_val)
