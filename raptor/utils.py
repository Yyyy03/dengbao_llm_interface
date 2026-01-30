import logging
import re
from typing import Dict, List, Set
import requests
import numpy as np
import tiktoken
from scipy import spatial
import json
from .tree_structures import Node
import config as cf
import base64


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer: tiktoken.get_encoding("cl100k_base"), max_tokens: int, overlap: int = 0
):
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """
    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            
            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)

## 关键词检索相关
# def extract_keywords_embedding(text, embedding_model) :
#     return embedding_model.create_embedding(text)

def get_keywords(node_list: List[Node], embedding_model) :
    keywords=[]
    for node in node_list:
        keywords.append(node.keywords_embedding)
    return keywords


def parse_markdown_to_tree(markdown_text):
    # 正则表达式匹配 Markdown 标题
    header_pattern = re.compile(r'^(#{1,6})\s*(.*)$', re.MULTILINE)
    # 正则表达式匹配非标题内容
    content_pattern = re.compile(r'^[^#].*$', re.MULTILINE)

    # 提取所有标题和内容
    headers = header_pattern.findall(markdown_text)
    contents = content_pattern.findall(markdown_text)

    # 初始化树形结构
    tree = {}
    stack = []  # 用于存储当前层级的节点路径

    # 当前处理到的 Markdown 内容的索引
    content_index = 0

    for i,header in enumerate(headers):
        level = len(header[0])  # 标题层级
        title = header[1].strip()  # 标题文本

        # 如果当前标题层级小于栈顶层级，需要回退到合适的层级
        while stack and stack[-1][0] >= level:
            stack.pop()

        # 创建当前标题的节点
        node = {"title": title, "children": [], "content": []}

        # 如果栈不为空，将当前节点添加到栈顶节点的子节点中
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            # 否则，当前节点是根节点
            tree[title] = node

        # 将当前节点及其层级压入栈
        stack.append((level, node))

        # 处理当前标题下的内容
        first_index = markdown_text.find(f'{header[0]} {header[1]}')
        if i!=len(headers)-1:
            next_index = markdown_text.find(f'{headers[i+1][0]} {headers[i+1][1]}')
        else:
            next_index = len(headers)-1
        if first_index!=-1 and next_index!=-1:
            text = markdown_text[first_index+len(f'{header[0]} {header[1]}'):next_index]
            node["content"] = text


    return tree


def rerank_docs(query, initial_docs, top_n=5):
    """
    使用bge-reranker-v2-m3对初始文档重排序
    """
    # 构造API请求参数（与OpenAI API风格兼容）

    reranker_model = "bge-reranker-v2-m3"
    reranker_key = cf.reranker_models[reranker_model]
    url = "https://openapi.seu.edu.cn/v1/rerank"  # 模型的API端点
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {reranker_key}"  # API密钥
    }

    # 文档内容通常取text字段，根据实际数据结构调整
    # embedding_list = []
    # response = client.embeddings.create(input=text, model=model)
    documents = [doc for doc in initial_docs]

    data = {
        "model": reranker_model,  # 模型名称
        "query": query,  # 用户查询
        "documents": documents,  # 初始候选文档
        "top_n": top_n  # 希望返回的Top N文档
    }

    # 发送请求
    response = requests.post(
        url=url,
        headers=headers,
        data=json.dumps(data)
    )

    # 解析响应（假设返回格式与OpenAI一致）
    if response.status_code == 200:
        result = response.json()
        # 提取重排序后的文档（包含相关性分数）
        reranked_docs = result["results"]
        # 转换为原始文档格式（保留分数）
        return [
            {
                "content": doc["document"]["text"],
                "score": doc["relevance_score"]
                # **initial_docs[i]  # 保留原始文档的其他字段（如id、来源等）
            }
            for i, doc in enumerate(reranked_docs)
        ]
    else:
        raise Exception(f"Reranker API error: {response.text}")

def image_to_decp(img_bytes):
# 2. 把图片 → base64

    model = "qwen2.5-vl-7b"
    apikey = cf.llm_models[model]
    baseurl = f"{cf.seu_base_url}/chat/completions"

    image_b64 = base64.b64encode(img_bytes).decode()

    # 3. 拼装多模态消息
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apikey}"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文详细描述这张图片，直接输出流程图描述"},
                    {
                        "type": "image_url",
                        "image_url": {
                            # 支持 url 或者 base64：data:image/jpeg;base64,xxxx
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800
    }

    # 4. 调用
    response = requests.post(
        baseurl,
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    # 5. 解析结果
    reply = response.json()["choices"][0]["message"]["content"]
    return reply
    # print("模型返回：", reply)
