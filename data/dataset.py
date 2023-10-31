import os, json

import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data

from sklearn.model_selection import train_test_split

from extracranial_vessel_labelling.data.utils import load_pickle, node_transform

class EVCDataset(InMemoryDataset):
    """
    Dataset class for the Extarcranial Vascular Centerline (EVC) dataset. The dataset is
    composed of centerline graphs generated from vascular segmentations from Full head-and-neck 
    CTA images, from which the centerlines of the extracranial vessels were extracted. The original 
    dataset contains centerline graphs from 561 cases. Nodes correspond to bifurcations, while edges
    correspond to vascular segments. The graphs contained manually labelled vessel types for each edge.
    However, we perform a node transform to the graphs, converting edges into nodes and maintaining connectivity 
    by adding edges to all segments immediately in contact in all bifurcation points. This way, we convert 
    the problem to a node classification task.

    All vessels are a type of either of these 14 classes: other, AA, BT, RCCA, LCCA, RSA, LSA, RVA, 
        LVA, RICA, LICA, RECA, LECA, BA.

    The dataset description json file is mandatory for normalizaiton purposes. If it is not found, it will be
    created from the raw data. 

    Parameters
    ----------
    root : str
        Path to the root folder of the dataset. We assume that raw_dir (`raw`) and processed_dir (`processed`) 
        are subfolders of root.
    raw_file_names : list of str, optional
        List of raw file names, by default None. If None, all files in raw_dir are used.
    processed_file_names : list of str, optional
        List of processed file names, by default None. If None, all files in processed_dir are used.
    pre_transform : object, optional
        Pre-transform object, by default None. It should input a Data object and output a Data object.

    """
    def __init__(self, root, raw_file_names_list = None, processed_file_names_list = None, pre_transform = None, transform = None):
        if os.path.exists(os.path.join(root, "dataset.json")):
            pass
        else:
            print("No dataset description file found. Creating dataset description file from raw dir data.")
            generate_EVC_dataset_json(root)
        with open(os.path.join(root, "dataset.json"), "r") as f:
            self.dataset_description = json.load(f)
        self.raw_file_names_list = raw_file_names_list
        self.processed_file_names_list = processed_file_names_list

        super(EVCDataset, self).__init__(root=root, pre_transform=pre_transform, transform=transform)
        self.process()

    @property
    def raw_file_names(self):
        if self.raw_file_names_list is not None:
            return self.raw_file_names_list
        else:
            return [f for f in sorted(os.listdir(self.raw_dir)) if f.endswith(".pickle")]

    @property
    def processed_file_names(self):
        if self.processed_file_names_list is not None:
            return self.processed_file_names_list
        else:
            return [f for f in sorted(os.listdir(self.processed_dir)) if f.endswith(".pt")]

    def process(self): 
        # Here you would read your raw files, create Data objects, and apply any pre-transforms
        data_list = []
        for raw_file in self.raw_file_names:
            graph_nx = node_transform(load_pickle(os.path.join(self.raw_dir, raw_file)))
            # Pytorch data object
            data = Data()
            x, edge_index, y, pos = [], [], [], []
            for node in graph_nx.nodes:
                pos.append(graph_nx.nodes[node]["pos"])
                x.append(graph_nx.nodes[node]["features"])
                y.append(graph_nx.nodes[node]["vessel_type"])
            for n0, n1 in graph_nx.edges:
                edge_index.append(np.array([n0, n1]))
            data.pos = torch.tensor(np.array(pos), dtype=torch.float32)
            data.x = self.normalize_edge_features(torch.tensor(np.array(x), dtype=torch.float32))
            data.edge_index = torch.transpose(torch.tensor(np.array(edge_index), dtype=torch.int64), 1, 0)
            data.y = torch.tensor(np.array(y), dtype=torch.int64)
            data.num_nodes = len(graph_nx.nodes)
            data.num_edges = len(graph_nx.edges)
            data.raw_file_path = raw_file
            data.processed_file_path = os.path.join(self.processed_dir, "{}.pt".format(raw_file.split(".")[0]))
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            torch.save(data, data.processed_file_path)
    
        self.data_list = data_list

    def get_splits(self, train_files=None, test_files=None, train_idx=None, test_idx=None, test_size=0.2, random_state=42):
        """
        Creates train and test splits from the dataset.

        Use cases:
        1. Provide train_files and test_files: train and test splits are created from the provided file lists.
            Example:    
                train_files = ["file1.pickle", "file2.pickle", "file3.pickle"]
                test_files = ["file4.pickle", "file5.pickle", "file6.pickle"]
                train_dataset, test_dataset = dataset.get_splits(train_files=train_files, test_files=test_files)
        2. Provide train_idx and test_idx: train and test splits are created from the provided indices.
            Example:
                train_idx = [0, 1, 2]
                test_idx = [3, 4, 5]
                train_dataset, test_dataset = dataset.get_splits(train_idx=train_idx, test_idx=test_idx)
        3. Provide only test_size: train and test splits are created randomly with the provided test_size. Random state can be provided.
            Example:
                test_size = 0.2
                train_dataset, test_dataset = dataset.get_splits(test_size=test_size)

        Parameters
        ----------
        train_files : list of str, optional
            List of train files, by default None
        test_files : list of str, optional
            List of test files, by default None
        train_idx : list of int, optional
            List of train indices, by default None
        test_idx : list of int, optional
            List of test indices, by default None
        test_size : float, optional
            Test size, by default 0.2
        random_state : int, optional
            Random state, by default 42

        Returns
        -------
        train_dataset : torch_geometric.data.InMemoryDataset
            Train dataset
        test_dataset : torch_geometric.data.InMemoryDataset
            Test dataset
        """
        if train_files is not None and test_files is not None:
            if train_files[0].endswith(".pickle") and train_files[0] in self.raw_file_names:
                train_idx = [self.raw_file_names.index(file) for file in train_files]
                test_idx = [self.raw_file_names.index(file) for file in test_files]
            elif train_files[0].endswith(".pt") and train_files[0] in self.processed_file_names:
                train_idx = [self.processed_file_names.index(file) for file in train_files]
                test_idx = [self.processed_file_names.index(file) for file in test_files]
            else:
                print("Provided file lists are not valid, defaulting to random split (test_size = {:.2f}).".format(test_size))
                pass
        elif train_idx is not None and test_idx is not None:
            pass  # Use the provided indices directly
        else:
            indices = list(range(len(self.raw_file_names)))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, train_size = 1 - test_size, random_state=random_state)
        
        # Get the file names for the train and test splits
        train_files = [self.raw_file_names[i] for i in train_idx]
        test_files = [self.raw_file_names[i] for i in test_idx]

        # Create new dataset instances for the train and test splits
        train_dataset = EVCDataset(self.root, raw_file_names_list=train_files, pre_transform=self.pre_transform)
        test_dataset = EVCDataset(self.root, raw_file_names_list=test_files, pre_transform=self.pre_transform)

        return train_dataset, test_dataset

    def normalize_edge_features(self, edge_features):
        def z_score_normalization(edge_features, mean, std):
            return (edge_features - mean) / std
        def min_max_normalization(edge_features, min, max):
            return (edge_features - min) / (max - min)
        def mean_centering_normalization(edge_features, mean):
            return edge_features - mean
        def ensure_normalized_vectors(edge_features):
            return edge_features / torch.norm(edge_features, dim=1, keepdim=True)
        
        if self.dataset_description is None:
            print("No dataset description file found. This will be an issue for normalization of edge features.")
            return edge_features
        else:
            # Compute min max values after mean centering for landmark positions normalization
            mean_position = self.dataset_description["mean_edge_features"][21:24]
            min_edge_features_mc = self.dataset_description["min_edge_features"].copy()
            max_edge_features_mc = self.dataset_description["max_edge_features"].copy()
            # Mean centering for r coordinates
            for idx in [15, 18, 21]:
                min_edge_features_mc[idx] -= mean_position[0]
                max_edge_features_mc[idx] -= mean_position[0]
            # Mean centering for a coordinates
            for idx in [16, 19, 22]:
                min_edge_features_mc[idx] -= mean_position[1]
                max_edge_features_mc[idx] -= mean_position[1]
            # Mean centering for s coordinates
            for idx in [17, 20, 23]:
                min_edge_features_mc[idx] -= mean_position[2]
                max_edge_features_mc[idx] -= mean_position[2]
            for idx, feature_name in enumerate(self.dataset_description["edge_feature_names"]):
                # For distance or continuous features, we apply z-score normalization
                if feature_name in ["mean radius", "proximal radius", "distal radius", "proximal/distal radius ratio", "minimum radius", "maximum radius", "distance", "relative length"]:
                    edge_features[:, idx] = z_score_normalization(edge_features[:, idx], self.dataset_description["mean_edge_features"][idx], self.dataset_description["std_edge_features"][idx])
                # For number of points, we apply min-max normalization
                elif feature_name in ["number of points"]:
                    edge_features[:, idx] = min_max_normalization(edge_features[:, idx], self.dataset_description["min_edge_features"][idx], self.dataset_description["max_edge_features"][idx])
                # For positional features, we apply mean centering normalization followed by min-max normalization
                elif feature_name in ["proximal bifurcation position r", "proximal bifurcation position a", "proximal bifurcation position s", 
                                      "distal bifurcation position r", "distal bifurcation position a", "distal bifurcation position s", 
                                      "pos r", "pos a", "pos s"]:
                    edge_features[:, idx] = mean_centering_normalization(edge_features[:, idx], self.dataset_description["mean_edge_features"][idx])
                    edge_features[:, idx] = min_max_normalization(edge_features[:, idx], min_edge_features_mc[idx], max_edge_features_mc[idx])
                # For direction and departure angle, we ensure normalized vectors
                edge_features[:, 8:11] = ensure_normalized_vectors(edge_features[:, 8:11]) # Direction
                edge_features[:, 11:14] = ensure_normalized_vectors(edge_features[:, 11:14]) # Departure angle

            return edge_features
        
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
def generate_EVC_dataset_json(
        root, 
        dataset_name = "Extracranial vascular centerlines",
        name_node_features = None,
        name_edge_features = "features",
        name_graph_features = None,
        name_graph_label = None,
        name_node_label = None,
        name_edge_label = "vessel_type",
        graph_labels_dict = None,
        node_labels_dict = None,
        edge_labels_dict = dict(zip([idx for idx in range(14)], ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]))
        ):
    """
    Generates a json file with the dataset information. Saves the file in the root folder:

    >>> os.path.join(os.path.join(root, "dataset.json"))

    Parameters
    ----------
    root : str
        Path to the root folder of the dataset.
    dataset_name : str, optional
        Name of the dataset, by default "Extracranial vascular centerlines"
    name_node_features : str, optional
        Name of the node features, by default None
    name_edge_features : str, optional
        Name of the edge features, by default "features"
    name_graph_features : str, optional
        Name of the graph features, by default None
    name_graph_label : str, optional
        Name of the graph label, by default None
    name_node_label : str, optional
        Name of the node label, by default None
    name_edge_label : str, optional
        Name of the edge label, by default "vessel_type"
    graph_labels_dict : dict, optional
        Dictionary with the graph labels, by default None
    node_labels_dict : dict, optional   
        Dictionary with the node labels, by default None
    edge_labels_dict : dict, optional
        Dictionary with the edge labels, by default 
        dict(zip([idx for idx in range(14)], ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]))
    """
    # Define raw_dir
    raw_dir = os.path.join(root, "raw")
    # Read filenames
    filenames = sorted([f for f in os.listdir(raw_dir) if f.endswith(".pickle")])
    # Load the data
    raw_data_list = [load_pickle(os.path.join(raw_dir, f)) for f in filenames]

    # Some data are just hardcoded
    # dataset_name = "Extracranial vascular centerlines"
    # name_graph_features = None
    # name_node_features = None
    # name_edge_features = "features"
    # name_graph_label = None
    # name_node_label = None
    # name_edge_label = "vessel_type"
    # graph_labels_dict = None
    # node_labels_dict = None
    # edge_labels_dict = dict(zip([idx for idx in range(14)], ["other", "AA", "BT", "RCCA", "LCCA", "RSA", "LSA", "RVA", "LVA", "RICA", "LICA", "RECA", "LECA", "BA"]))
    num_graph_classes = None
    num_node_classes = None
    num_edge_classes = len(edge_labels_dict)
    total_number_of_examples = len(raw_data_list)

    # Others are derived from a single example
    example = raw_data_list[0]
    if name_graph_features is None:
        num_graph_features = 0
        graph_feature_names = None
    else:
        num_graph_features = len(example.graph[name_graph_features])
        if name_graph_features + "_dict" in example.graph:
            graph_feature_names = list(example.graph[name_graph_features + "_dict"].keys())
        else:
            graph_feature_names = None
    if name_node_features is None:
        num_node_features = 0
        node_feature_names = None
    else:
        num_node_features = len(example.nodes[0][name_node_features])
        if name_node_features + "_dict" in example.nodes[0]:
            node_feature_names = list(example.nodes[0][name_node_features + "_dict"].keys())
        else:
            node_feature_names = None
    if name_edge_features is None:
        num_edge_features = 0
        edge_feature_names = None
    else:
        num_edge_features = len(example[0][1][name_edge_features])
        if name_edge_features + "_dict" in example[0][1]:
            edge_feature_names = list(example[0][1][name_edge_features + "_dict"].keys())
        else:
            edge_feature_names = None

    # For the rest, we have to iterate over the dataset
    total_number_of_nodes = 0
    total_number_of_edges = 0

    graph_featues_array = np.zeros((num_graph_features, 0))
    node_featues_array = np.zeros((num_node_features, 0))
    edge_featues_array = np.zeros((num_edge_features, 0))

    for example in raw_data_list:
        total_number_of_nodes += len(example.nodes)
        total_number_of_edges += len(example.edges)
        if name_graph_features is not None:
            graph_featues_array = np.concatenate((graph_featues_array, np.expand_dims(example.graph[name_graph_features], axis = 1)), axis=1)
        if name_node_features is not None:
            for node in example.nodes:
                node_featues_array = np.concatenate((node_featues_array, np.expand_dims(example.nodes[node][name_node_features], axis = 1)), axis=1)
        if name_edge_features is not None:
            for src, dst in example.edges:
                edge_featues_array = np.concatenate((edge_featues_array, np.expand_dims(example[src][dst][name_edge_features], axis = 1)), axis=1)

    average_number_of_nodes = total_number_of_nodes / total_number_of_examples
    average_number_of_edges = total_number_of_edges / total_number_of_examples

    # We can also compute class frequencies for loss weighting
    if name_graph_label is not None:
        graph_class_frequencies = np.zeros(num_graph_classes)
        for example in raw_data_list:
            for src, dst in example.graphs:
                graph_class_frequencies[example[src][dst][name_graph_label]] += 1

        graph_class_frequencies /= np.sum(graph_class_frequencies)
    else:
        graph_class_frequencies = np.zeros(0)
    # For nodes
    if name_node_label is not None:
        node_class_frequencies = np.zeros(num_node_classes)
        for example in raw_data_list:
            for src, dst in example.nodes:
                node_class_frequencies[example[src][dst][name_node_label]] += 1
        node_class_frequencies /= np.sum(node_class_frequencies)
    else:
        node_class_frequencies = np.zeros(0)
    # For edges
    if name_edge_label is not None:
        edge_class_frequencies = np.zeros(num_edge_classes)
        for example in raw_data_list:
            for src, dst in example.edges:
                edge_class_frequencies[example[src][dst][name_edge_label]] += 1
        edge_class_frequencies /= np.sum(edge_class_frequencies)
    else:
        edge_class_frequencies = np.zeros(0)

    # Now, just create the json file
    dataset_dict = {
        "dataset_name": dataset_name,
        "total_number_of_examples": total_number_of_examples,
        "total_number_of_nodes": total_number_of_nodes,
        "total_number_of_edges": total_number_of_edges,
        "average_number_of_nodes": average_number_of_nodes,
        "average_number_of_edges": average_number_of_edges,
        "name_graph_features": name_graph_features,
        "name_node_features": name_node_features,
        "name_edge_features": name_edge_features,
        "num_graph_features": num_graph_features,
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "graph_feature_names": graph_feature_names,
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "name_graph_label": name_graph_label,
        "name_node_label": name_node_label,
        "name_edge_label": name_edge_label,
        "graph_labels_dict": graph_labels_dict,
        "node_labels_dict": node_labels_dict,
        "edge_labels_dict": edge_labels_dict,
        "num_graph_classes": num_graph_classes,
        "num_node_classes": num_node_classes,
        "num_edge_classes": num_edge_classes,
        "graph_class_frequencies": graph_class_frequencies.tolist() if len(graph_featues_array) > 0 else None,
        "node_class_frequencies": node_class_frequencies.tolist() if len(graph_featues_array) > 0 else None,
        "edge_class_frequencies": edge_class_frequencies.tolist() if len(graph_featues_array) > 0 else None,
        "mean_graph_features": np.mean(graph_featues_array, axis=1).tolist() if len(graph_featues_array) > 0 else None,
        "median_graph_features": np.median(graph_featues_array, axis=1).tolist() if len(graph_featues_array) > 0 else None,
        "std_graph_features": np.std(graph_featues_array, axis=1).tolist() if len(graph_featues_array) > 0 else None,
        "min_graph_features": np.min(graph_featues_array, axis=1).tolist() if len(graph_featues_array) > 0 else None,
        "max_graph_features": np.max(graph_featues_array, axis=1).tolist() if len(graph_featues_array) > 0 else None,
        "mean_node_features": np.mean(node_featues_array, axis=1).tolist() if len(node_featues_array) > 0 else None,
        "median_node_features": np.median(node_featues_array, axis=1).tolist() if len(node_featues_array) > 0 else None,
        "std_node_features": np.std(node_featues_array, axis=1).tolist() if len(node_featues_array) > 0 else None,
        "min_node_features": np.min(node_featues_array, axis=1).tolist() if len(node_featues_array) > 0 else None,
        "max_node_features": np.max(node_featues_array, axis=1).tolist() if len(node_featues_array) > 0 else None,
        "mean_edge_features": np.mean(edge_featues_array, axis=1).tolist() if len(edge_featues_array) > 0 else None,
        "median_edge_features": np.median(edge_featues_array, axis=1).tolist() if len(edge_featues_array) > 0 else None,
        "std_edge_features": np.std(edge_featues_array, axis=1).tolist() if len(edge_featues_array) > 0 else None,
        "min_edge_features": np.min(edge_featues_array, axis=1).tolist() if len(edge_featues_array) > 0 else None,
        "max_edge_features": np.max(edge_featues_array, axis=1).tolist() if len(edge_featues_array) > 0 else None
    }

    # Save the json file
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump(dataset_dict, f, indent=4)