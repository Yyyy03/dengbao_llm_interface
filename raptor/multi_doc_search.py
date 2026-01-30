# -*- coding: utf-8 -*-
import json
import openai
from typing import List
import numpy as np
from scipy import spatial
import requests
from importlib import resources
import config as cf


def seu_embedding(model, api_key, text):
    client = openai.OpenAI(
        base_url="https://openapi.seu.edu.cn/v1",
        api_key=api_key
    )
    embedding_list = []
    response = client.embeddings.create(input=text, model=model)
    for d in response.data:
        embedding_list.append(d.embedding)
    return embedding_list


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

    # distance = distance_metrics[distance_metric](query_embedding, embedding)

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances

def openai_api_seu(query,apikey,model):
    baseurl = "https://openapi.seu.edu.cn/v1"
    client = openai.OpenAI(
        base_url=baseurl,
        api_key=apikey
    )


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    # print(completion)
    return completion.choices[0].message.content


def rerank_docs(query, initial_docs, top_n=5):
    """
    使用bge-reranker-v2-m3对初始文档重排序
    """
    # 一次最大支持的文档数量
    batch_size = 32
    # 构造API请求参数（与OpenAI API风格兼容）
    q = query.strip()
    if q == "" or len(q)<2:
        raise ValueError("query不能为空")
    if not initial_docs:
        raise ValueError("文档不能为空")
    reranker_key = "b6da699bdbd94f2e856ffdf70f005018fbcb"
    reranker_model = "bge-reranker-v2-m3"
    url = "https://openapi.seu.edu.cn/v1/rerank"  # 模型的API端点
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {reranker_key}"  # API密钥
    }

    n = len(initial_docs)
    batch_num  = n//batch_size +1
    rerank_doc_list = []
    for i in range(batch_num):
        documents = initial_docs[i*batch_size:(i+1)*batch_size]
        if documents == []:
            continue
        data = {
            "model": reranker_model,  # 模型名称
            "query": q,  # 用户查询
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
            # print(response.text)
            if result.get('error'):
                raise ValueError(result["error"])
            # print(result)
            # 提取重排序后的文档（包含相关性分数）
            reranked_docs = result["results"]
            # print(reranked_docs)
            # 转换为原始文档格式（保留分数）
            batch_res =  [
                {
                    "score": doc["relevance_score"],
                    "index": doc["index"]
                }
                for i, doc in enumerate(reranked_docs[0:top_n])
            ]
            rerank_doc_list.extend(batch_res)
        else:
            raise Exception(f"Reranker API error: {response.text}")
    rerank_doc_list.sort(key=lambda x: x["score"], reverse=True)
    return rerank_doc_list


def select_doc(query, topk, embedding_model, embedding_model_key):
    query_embedding = seu_embedding(embedding_model,embedding_model_key,query)
    with resources.open_text("tree-json","meta_data.json") as f:
    # with open("../tree-json/meta_data.json", encoding="utf8") as f:
        meta_data = json.load(f)
    file_summary_text = []
    for key in meta_data.keys():
        tmp = meta_data[key]["filename"] + meta_data[key]["text"] # + meta_data[key]["children_text"]
        file_summary_text.append(tmp)
    summary_embeddings = seu_embedding(embedding_model,embedding_model_key,file_summary_text)
    dis = []
    dis.extend(distances_from_embeddings(query_embedding[0], summary_embeddings))

    index_dis = np.argsort(dis)
    top_k = 3
    best_files = []
    keys = list(meta_data.keys())
    for i in range(top_k):
        best_files.append(keys[index_dis[i]])
    return best_files
def select_doc_llm(query):
    prompt = f"""我们正在完成一个多标准文档检索查询任务，请根据{query}，判断回答该问题需要在哪个或哪些标准文档里进行检索。文档与文档摘要（内容和适用范围等）如下所述。请输出需要搜索的文档名，用","分隔，不要输出其他内容。输出格式参考：['文档名1', '文档名2', '文档名3', '文档名4']

# 22239
本标准是网络安全等级保护的通用基本要求，定义了第一级至第四级等级保护对象的安全通用要求，以及云计算、移动互联、物联网、工业控制系统等安全扩展要求。内容涉及安全物理环境、安全通信网络、安全管理中心等技术和管理框架。适用于所有行业非涉密对象的安全建设和监督管理，是等级保护的核心标准之一。

# 28448
本标准是网络安全等级保护的通用测评要求，规定了第一级至第四级等级保护对象的安全测评通用要求，以及针对云计算、移动互联、物联网、工业控制系统等新技术的安全测评扩展要求。内容涵盖安全物理环境、安全通信网络、安全区域边界等测评指标。适用于测评服务机构、运营使用单位及主管部门进行安全测评，并提供大数据安全评估方法参考。

# 28449
本标准是网络安全等级保护的测评过程指南，规范了等级测评的工作流程和活动任务，包括测评准备、方案编制、现场测评、报告编制等阶段。内容强调测评风险规避和新技术（如云计算、物联网）测评补充要求。适用于测评机构、主管部门和运营使用单位开展网络安全测试评价工作。

# 25070
本标准是网络安全等级保护的安全设计技术要求标准，规定了第一级至第四级等级保护对象的安全设计通用要求，以及针对云计算、移动互联、物联网、工业控制和大数据等新技术的安全设计扩展要求。内容涵盖安全计算环境、安全区域边界、安全通信网络和安全管理中心的设计策略、目标及具体技术措施。适用于指导运营使用单位、网络安全企业和服务机构进行安全技术方案的设计与实施，也可作为监督部门检查的依据。区别于等级保护基本要求和测评要求标准，本标准专注于技术设计层面。

# 22240
本标准是网络安全等级保护的定级指南，提供了非涉密等级保护对象的安全保护等级定级方法、流程和原则，包括确定定级对象、初步定级、等级确定和变更等环节。适用于所有行业的网络运营者开展等级保护对象定级工作，不涉及具体安全要求或测评内容。

# JRT0060
本标准是证券期货业网络安全等级保护的基本要求标准，定义了总体要求和第一级至第四级的安全通用要求，以及云计算、移动互联、物联网、工业控制系统等安全扩展要求。内容涉及安全物理环境、安全通信网络、安全管理中心等技术和管理措施。适用于证券期货业非涉密对象的安全建设和监督管理。

# JRT0071
本标准是金融行业网络安全等级保护的基本要求标准，定义了网络安全保障框架和第二级至第四级的安全通用要求，并包括云计算、移动互联、物联网等安全扩展要求。内容涉及安全物理环境、安全通信网络、安全管理中心等技术和管理要求。适用于指导金融机构、测评机构和主管部门实施网络安全等级保护工作，包括安全建设和监督管理。

# 网络安全等级保护测评高风险判定实施指引
本标准是网络安全等级保护测评的高风险判定实施指引，提供了安全通用要求和云计算、移动互联、物联网、工业控制系统等扩展要求的高风险判例，包括物理访问控制、边界防护、身份鉴别、数据备份恢复等场景的风险判定。适用于等级保护测评和安全检查活动，用于识别和整改高风险问题。
  
# 测评机构在沪工作指引规范手册
该文档是上海市重要信息系统安全等级保护工作协调小组办公室发布的测评机构工作指引，规范了在上海市范围内开展等级测评活动的机构管理、项目管理和实施流程。内容涵盖机构申请与报备、测评月报管理、项目申请与实施、现场检查、报告评审及附件模板（如承诺书、检查表）。适用于在沪测评机构，基于《网络安全法》等法律法规，提供本地化操作规范和监督检查依据。

# 关于对网络安全等级保护有关工作事项进一步说明的函
该文档是公安部十一局针对网络安全等级保护工作事项的官方说明函，重点细化了系统备案动态更新、第五级网络系统定义、数据资源摸底、风险隐患排查和问题整改等操作细节。附件包含24个问题释疑，涵盖备案流程、定级原则、测评要求、数据调查方法和高风险判定依据等。适用于中央和国家机关各部委、国务院直属机构、事业单位及中央企业的网络安全职能部门，提供具体工作指导和执行参考。

# 网络安全法
该文档是中华人民共和国网络安全法，作为国家法律规定了网络安全的基本框架和要求，内容包括总则、网络安全支持与促进、网络运行安全（含一般规定和关键信息基础设施）、网络信息安全、监测预警与应急处置、法律责任及附则。适用于所有网络运营者，是网络安全等级保护制度的法律基础，强调网络运行安全、关键信息基础设施保护和数据安全等核心义务。"""
    model = "qwen2.5-72b"
    res = openai_api_seu(prompt,cf.llm_models[model],model)
    return res

if __name__ == "__main__":
    res = rerank_docs("111",["111","222","333"])
    print(res)
    # queries = [
    #     "请简述在定级阶段、安全建设阶段、等级测评阶段主要参考的标准和作用是什么？",
    #     "等级保护对二级系统都有哪些关于安全物理环境的基本要求？根据上级部门监管要求，某金融公司的二级系统开展等级保护工作需依据金融行业标准的基本要求，则在安全物理环境方面会增加哪些要求？",
    #     "某三级系统的业务应用系统是使用用户名+口令的方式对登录用户进行身份鉴别的，根据等保相关要求，这是否存在什么问题？如果存在问题的话，一般将此问题的级别判定为高风险、中风险还是低风险？是否存在缓解措施？"
    # ]
    # import ast
    # for q in queries:
    #     res = select_doc_llm(q)
    #     res = ast.literal_eval(res)
    #     if '22239' in res:
    #         res.remove('22239')
    #     print(res)
