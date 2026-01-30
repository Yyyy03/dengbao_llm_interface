import logging
import os
from typing import Dict, List, Set
import re
import tiktoken

from .EmbeddingModels import BaseEmbeddingModel, OpenAIEmbeddingModel
from .Retrievers import BaseRetriever
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances,
                    reverse_mapping,get_keywords)
import numpy as np

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from typing import List, Dict, Optional, Set
from collections import defaultdict
import logging
logger = logging.getLogger("uvicorn.error")

def build_parent_map(all_nodes: Dict[int, Node]) -> Dict[Node, Optional[Node]]:
    parent_map: Dict[Node, Optional[Node]] = {}
    for node in all_nodes.values():      # 只改这里
        for child in node.children:
            parent_map[child] = node
    return parent_map


def complete_leaf_nodes(leaf_nodes: List[Node],
                        all_nodes: Dict[int, Node]) -> List[Node]:

    parent_map = build_parent_map(all_nodes)
    #logger.info(parent_map)

    hits = defaultdict(list)
    for leaf in leaf_nodes:          # 遍历值
        #logger.info(leaf)
        p = parent_map.get(leaf.index)
        if p is None:
            continue
        else:
            hits[p].append(leaf)

    completed = set(leaf_nodes)


    for p, hit_list in hits.items():

        all_leaf_children = [n for n in p.children if all_nodes[n].isleaf]
        if len(hit_list) >= len(all_leaf_children) / 2:
            logger.info(all_leaf_children)
            completed.update(all_nodes[n] for n in all_leaf_children)

    return list(completed)

class TreeRetrieverConfig:
    def __init__(
        self,
        tokenizer=None,
        threshold=None,
        top_k=None,
        selection_mode=None,
        context_embedding_model=None,
        embedding_model=None,
        num_layers=None,
        start_layer=None,
    ):
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

        if threshold is None:
            threshold = 0.5
        if not isinstance(threshold, float) or not (0 <= threshold <= 1):
            raise ValueError("threshold must be a float between 0 and 1")
        self.threshold = threshold

        if top_k is None:
            top_k = 5
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an integer and at least 1")
        self.top_k = top_k

        if selection_mode is None:
            selection_mode = "top_k"
        if not isinstance(selection_mode, str) or selection_mode not in [
            "top_k",
            "threshold",
        ]:
            raise ValueError(
                "selection_mode must be a string and either 'top_k' or 'threshold'"
            )
        self.selection_mode = selection_mode

        if context_embedding_model is None:
            context_embedding_model = "OpenAI"
        if not isinstance(context_embedding_model, str):
            raise ValueError("context_embedding_model must be a string")
        self.context_embedding_model = context_embedding_model

        if embedding_model is None:
            embedding_model = OpenAIEmbeddingModel()
        if not isinstance(embedding_model, BaseEmbeddingModel):
            raise ValueError(
                "embedding_model must be an instance of BaseEmbeddingModel"
            )
        self.embedding_model = embedding_model

        if num_layers is not None:
            if not isinstance(num_layers, int) or num_layers < 0:
                raise ValueError("num_layers must be an integer and at least 0")
        self.num_layers = num_layers

        if start_layer is not None:
            if not isinstance(start_layer, int) or start_layer < 0:
                raise ValueError("start_layer must be an integer and at least 0")
        self.start_layer = start_layer

    def log_config(self):
        config_log = """
        TreeRetrieverConfig:
            Tokenizer: {tokenizer}
            Threshold: {threshold}
            Top K: {top_k}
            Selection Mode: {selection_mode}
            Context Embedding Model: {context_embedding_model}
            Embedding Model: {embedding_model}
            Num Layers: {num_layers}
            Start Layer: {start_layer}
        """.format(
            tokenizer=self.tokenizer,
            threshold=self.threshold,
            top_k=self.top_k,
            selection_mode=self.selection_mode,
            context_embedding_model=self.context_embedding_model,
            embedding_model=self.embedding_model,
            num_layers=self.num_layers,
            start_layer=self.start_layer,
        )
        return config_log


class TreeRetriever(BaseRetriever):

    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )

        self.tree = tree
        self.num_layers = (
            config.num_layers if config.num_layers is not None else tree.num_layers + 1
        )
        self.start_layer = (
            config.start_layer if config.start_layer is not None else tree.num_layers
        )

        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        self.tokenizer = config.tokenizer
        self.top_k = config.top_k
        self.threshold = config.threshold
        self.selection_mode = config.selection_mode
        self.embedding_model = config.embedding_model
        self.context_embedding_model = config.context_embedding_model

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_nodes)

        logging.info(
            f"Successfully initialized TreeRetriever with Config {config.log_config()}"
        )

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)

    def complete_leaf_nodes2(self,leaf_nodes: List[Node],
                            all_nodes: Dict[int, Node]) -> List[Node]:
        logger.info("---- 进入 retrieve ----")

        parent_map = build_parent_map(all_nodes)
        # logger.info(parent_map)

        hits = defaultdict(list)
        for leaf in leaf_nodes:  # 遍历值
            # logger.info(leaf)
            p = parent_map.get(leaf.index)
            # logger.info(p,'11111')
            if p is None:
                continue
            else:
                hits[p].append(leaf)

        completed = set(leaf_nodes)

        # logger.info(hits)
        for p, hit_list in hits.items():
            # logger.info(p)
            # logger.info(p.children)
            all_leaf_idx = [n for n in p.children if all_nodes[n].isleaf]
            if len(hit_list) >= len(all_leaf_idx) / 2:
                merged_text = "\n".join(all_nodes[n].text for n in all_leaf_idx)
                # 4.2 生成向量
                merged_emb = self.create_embedding(merged_text)
                # 4.3 造新节点
                new_leaf = Node(
                    text=merged_text,
                    index=-1,  # 用负数或全局唯一 id 生成器，避免冲突
                    children=set(),
                    embeddings=merged_emb,
                    isleaf=True
                )
                # 4.4 从结果池里删掉旧叶子
                old_leaf_nodes = [all_nodes[n] for n in all_leaf_idx]
                completed.difference_update(old_leaf_nodes)
                # 4.5 加入新叶子
                completed.add(new_leaf)

        return list(completed)

    def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_embeddings(node_list, self.context_embedding_model)

        distances = distances_from_embeddings(query_embedding, embeddings)

        # 关键词检索并打分
        keywords_embedding = get_keywords(node_list, self.embedding_model)
        keyword_scores = distances_from_embeddings(query_embedding, keywords_embedding)
        if len(distances) != 0:
            d1 = np.max(distances) - np.min(distances)
            k1 = np.max(keyword_scores) - np.min(keyword_scores)
            if d1 != 0 and k1 != 0:
                distances = (distances - np.min(distances)) / d1
                keyword_scores = (keyword_scores - np.min(keyword_scores)) / k1
                for i in range(0, len(keyword_scores)):
                    distances[i] = distances[i] * 0.8 + keyword_scores[i] * 0.2

        indices = indices_of_nearest_neighbors_from_distances(distances)

        total_tokens = 0
        for idx in indices[:top_k]:

            node = node_list[idx]
            node_tokens = len(self.tokenizer.encode(node.text))

            if total_tokens + node_tokens > max_tokens:
                break

            selected_nodes.append(node)
            total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    def retrieve_information(
        self, current_nodes: List[Node], query: str, num_layers: int,all_nodes: List[Node]
    ):
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = self.create_embedding(query)

        query_route = []

        node_list = current_nodes
        weights = []

        for layer in range(num_layers):
            if node_list == []:
                break

            embeddings = get_embeddings(node_list, self.context_embedding_model)

            distances = distances_from_embeddings(query_embedding, embeddings)
            if weights:
                for i in range(len(distances)):
                    distances[i] = distances[i] * weights[i]

            # print(f"layer: {layer}")
            # print(f"dis of embedding:  {distances}")

            # 关键词检索并打分
            # keywords_embeddings = get_keywords(node_list, self.embedding_model)
            # keyword_scores = distances_from_embeddings(query_embedding, keywords_embeddings)
            # if len(distances) != 0:
            #     d1 = np.max(distances) - np.min(distances)
            #     k1 = np.max(keyword_scores) - np.min(keyword_scores)
            #     if d1 != 0 and k1 != 0:
            #         distances = (distances - np.min(distances)) / d1
            #         keyword_scores = (keyword_scores - np.min(keyword_scores)) / k1
            #         for i in range(0, len(keyword_scores)):
            #             distances[i] = distances[i] * 0.8 + keyword_scores[i] * 0.2
            indices = indices_of_nearest_neighbors_from_distances(distances)
            # print(f"rank of dis: {indices}")

            if self.selection_mode == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.threshold
                ]

            elif self.selection_mode == "top_k":
                best_indices = indices[: self.top_k]

            nodes_to_add = [node_list[idx] for idx in best_indices]

            query_route.append(nodes_to_add)
            weights = []
            first_weight = 0.8
            last_weight = 1
            if layer != num_layers - 1:

                child_nodes = []

                for i, index in enumerate(best_indices):
                    weight = first_weight+(last_weight-first_weight)/(self.top_k-1)*i
                    child_nodes.extend(node_list[index].children)
                    weights.extend([weight for idx in range(len(node_list[index].children))])

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]
        leaf_nodes = []
        for row in query_route:
            for node in row:
                if node.isleaf:
                    leaf_nodes.append(node)

        route_nodes = []
        for route in query_route[0:-1]:
            if route[0].isleaf:
                continue
            route_nodes.append(route[0])
        leaf_nodes = self.complete_leaf_nodes2(leaf_nodes,all_nodes)
        return leaf_nodes, route_nodes

    def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10,
        max_tokens: int = 3500,
        collapse_tree: bool = True,
        return_layer_information: bool = False,
    ):
        """
        Queries the tree and returns the most relevant information.

        Args:
            query (str): The query text.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The result of the query.
        """

        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        start_layer = self.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.tree.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")
        selected_nodes = []
        extra_nodes = []
        if collapse_tree:
            logging.info(f"Using collapsed_tree")
            selected_nodes, context = self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )

        else:
            print(self.start_layer)
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, extra_nodes = self.retrieve_information(
                layer_nodes, query, num_layers,self.tree.all_nodes
            )

        selected_context = []
        for node in selected_nodes:
            cleaned = re.sub(r'\n+', '\n', node.text)
            text = cleaned
            node_dict = {"context":text,"type":"leaf","index":node.index}
            try:
                node_dict["global_index"] = node.global_index
            except AttributeError:
                node_dict["global_index"] = -1
            selected_context.append(node_dict)



        extra_context = []
        for node in extra_nodes:
            cleaned = re.sub(r'\n+', '\n', node.text)
            text = cleaned
            extra_context.append({"context":text,"type":"notleaf","index":node.index})


        if return_layer_information:

            layer_information = []

            for node in (selected_nodes+extra_nodes):
                if node.index>-1:
                    layer_information.append(
                        {
                            "node_index": node.index,
                            "layer_number": self.tree_node_index_to_layer[node.index],
                        }
                    )

            return selected_context, extra_context, layer_information

        return selected_context, extra_context
