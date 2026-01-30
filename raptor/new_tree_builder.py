import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set
from collections import deque
# from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)
import openai


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class NewTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension


    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        """
        return base_summary + cluster_tree_summary


class NewTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)
        # if not isinstance(config, NewTreeConfig):
        #     raise ValueError("config must be an instance of ClusterTreeConfig")
        # self.reduction_dimension = config.reduction_dimension
        #
        # logging.info(
        #     f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        # )

    def add_text_tree(self,text:Dict):
        self.text_tree = text


    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) :
        logging.info("Using New TreeBuilder")
        layer_nodes=dict()
        # 通过markdown文档的层次结构直接构建树形node结构
        q=deque()

        root = Node("",0,set(),None)
        all_tree_nodes.update({root.index:root})

        layer_nodes[0]=[root]
        count=1

        for key in self.text_tree.keys():
            cur_node = Node(key+'\n'+self.text_tree[key]["content"], count,set(),None)
            all_tree_nodes.update({cur_node.index:cur_node})
            root.children.add(cur_node.index)
            count+=1
            self.text_tree[key]["index"]=cur_node.index
            q.append((self.text_tree[key],1))
            if layer_nodes.get(1):
                layer_nodes[1].append(cur_node)
            else:
                layer_nodes[1] = [cur_node]
        while q:
            cur, cur_layer = q.popleft()
            for child in cur["children"]:
                cur_node = Node(all_tree_nodes[cur["index"]].text+'\n'+child["title"]+'\n'+child["content"], count,set(),None)
                all_tree_nodes.update({cur_node.index: cur_node})

                all_tree_nodes[cur["index"]].children.add(cur_node.index)

                # print(cur["index"])
                # print(all_tree_nodes.keys())
                child["index"]=cur_node.index
                q.append((child,cur_layer+1))
                if layer_nodes.get(cur_layer + 1):
                    layer_nodes[cur_layer + 1].append(cur_node)
                else:
                    layer_nodes[cur_layer + 1] = [cur_node]
                count+=1
        self.num_layers = len(layer_nodes.keys())-1
        keys = list(layer_nodes.keys())
        for key in keys:
            layer_to_nodes[self.num_layers-key] = layer_nodes.pop(key)
        for node in layer_to_nodes[0]:
            node.isleaf = True
        self.summarize_tree(root,all_tree_nodes)
        # self.no_summarize(root, all_tree_nodes)
        for node in all_tree_nodes.values():
            node.keywords = self.generate_keyword(node.text)
            node.keywords_embedding = self.create_embedding(node.keywords)
        return root

    def summarize_tree(self,root : Node, all_nodes : Dict[int,Node]):
        context = ""
        titles = root.text.split("\n")
        title = ""
        for t in titles:
            title +=t +'\n'
        for index in root.children:
            tmp = all_nodes[index].text+'\n' + self.summarize_tree(all_nodes[index],all_nodes)
            context += (tmp + "\n\n")
        if root.children:
            summarized_text = title+"\n"+self.summarize(context,150)
        else:
            summarized_text = title
            root.isleaf = True
        root.text = summarized_text
        embeddings = {
            model_name: model.create_embedding(summarized_text)
            for model_name, model in self.embedding_models.items()
        }
        root.embeddings = embeddings
        return summarized_text


    def no_summarize(self,root : Node, all_nodes : Dict[int,Node]):
        for node in all_nodes.values():
            embeddings = {
                model_name: model.create_embedding(node.text)
                for model_name, model in self.embedding_models.items()
            }
            node.embeddings = embeddings

    def generate_keyword(self,context):
        base_url = "https://openapi.seu.edu.cn/v1"
        model_name = "qwen2.5-72b"
        api_key = "39c9632a-f67b-4e7c-96f7-9b89faead338"
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        prompt = "请你将下列内容提取关键内容，用于进行检索知识库内容。请提取对知识描述最有价值的关键词，例如专有名词、专业术语或关键对象，能够帮助精准定位到此文档。关键词应能准确反映问题的核心内容，避免提取泛泛无意义的词。"\
                "##原文本"\
                f"{context}"\
                "##输出"\
                "输出3至7个关键词，关键词之间用空格隔开，无需其他多余文字。"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个问答助手"},
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

if __name__ == "__main__":
    import json
    with open('../tree.json', 'r') as f:
        text = json.load(f)
    a = NewTreeBuilder(None,text)
    nodes = a.construct_tree(None,None,None)
    print(nodes)