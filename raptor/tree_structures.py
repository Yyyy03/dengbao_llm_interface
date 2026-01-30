from typing import Dict, List, Set


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, children: Set[int], embeddings, isleaf = False,global_idx=-1) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings
        self.isleaf = isleaf
        self.keywords=""
        self.keywords_embedding=None
        self.global_index = global_idx


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes: Dict[int, Node], root_nodes, leaf_nodes: Dict[int, Node], num_layers: int, layer_to_nodes: Dict[int, List[Node]]
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

