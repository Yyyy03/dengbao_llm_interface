import logging
import os

from numba.cuda.printimpl import print_item

import multi_doc

if not os.path.exists('log'):
    os.mkdir('log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./log/run.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
if not os.path.exists('demo/token_database'):
    os.mkdir('demo/token_database')

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import numpy as np
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, utils
from raptor.CustomRA import CustomRA
from example.build_tree import CustomQAModel, CustomEmbeddingModel, CustomSummarizationModel
from raptor.multi_doc_search import rerank_docs, select_doc_llm, openai_api_seu
from bm25 import BM25Retriever, Block
import openai
from typing import List
from scipy import spatial
import config as cf
import ast
from raptor.query_generation import generate_queries
from multi_doc import doc_type






app = FastAPI(title="DengbaoRAG")

# raptor启动
summary_model = "qwen2.5-72b"
summary_model_key = cf.llm_models[summary_model]
qa_model = "qwen3-32b"
qa_model_key = cf.llm_models[qa_model]
embedding_model = "bge-m3"
embedding_model_key = cf.embedding_models[embedding_model]
custom_summarizer = CustomSummarizationModel(summary_model,summary_model_key)
custom_qa = CustomQAModel(qa_model,qa_model_key)
custom_embedding = CustomEmbeddingModel(embedding_model,embedding_model_key)

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding,
    tree_builder_type="markdown"
)

raptor_22239 = RetrievalAugmentation(tree="./demo/22239-keywords", config=custom_config)
raptor_22239_appendix = RetrievalAugmentation(tree = "./demo/22239-appendix",config=custom_config)


markdown_path = "./demo/22239-new.md"
fenceng_rag = CustomRA(docs_path="./demo/22239-fenceng", md_path = markdown_path, config=custom_config, init=False)
files_dict = {'22240': "GBT 22240—2020 信息安全技术 网络安全等级保护定级指南-报批稿（提交版）-最终提交审批中心版本",
             '28448':"GBT 28448-2019 信息安全技术 网络安全等级保护测评要求",
             '28449':"GB∕T 28449-2018 信息安全技术 网络安全等级保护测评过程指南",
             '25070':"GB∕T 25070-2019 信息安全技术 网络安全等级保护安全设计技术要求",
             'JRT0060':"JR∕T 0060-2021 金融行业网络安全等级保护基本要求",
             'JRT0071':"JR∕T 0071.2-2020 金融行业网络安全等级保护实施指引 第2部分：基本要求",
             '关于对网络安全等级保护有关工作事项进一步说明的函':"关于对网络安全等级保护有关工作事项进一步说明的函（公网安〔2025〕1846号）",
             '测评机构在沪工作指引规范手册':"测评机构在沪工作指引规范手册",
             '网络安全法':"网络安全法",
             '网络安全等级保护测评高风险判定实施指引':"网络安全等级保护测评高风险判定实施指引（试行）"
              }

raptor_pool=dict()
for f in files_dict.keys():
    raptor_pool[f] = RetrievalAugmentation(tree=f"./demo/raptor-trees/{f}", config=custom_config)

fenceng_file_list = ['22239','22240', '28448', '28449', '25070', 'JRT0060', 'JRT0071', '关于对网络安全等级保护有关工作事项进一步说明的函', '测评机构在沪工作指引规范手册', '网络安全法', '网络安全等级保护测评高风险判定实施指引']
fenceng_pool=dict()
for f in fenceng_file_list:
    fenceng_pool[f] = CustomRA(docs_path=f"./demo/fenceng/{f}", md_path = f"./demo/mds/{f}.md", config=custom_config, init=False)

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




'''
{
"query": "用户的知识库查询",
}
返回数据格式：
{
    "context":text,
    "type":"leaf",
    "index":node.index
}
'''
@app.post("/raptor/22239")
async def raptor_rag(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")


    print(req_data["query"])
    leaf_retriever, route_retriever, layer_information = raptor_22239.retrieve(req_data["query"], collapse_tree=False, return_layer_information=True, top_k=5)
    leaf_context = ""
    route_context = ""
    for leaf in leaf_retriever:
        leaf_context += leaf["context"].strip()
        leaf_context += "\n\n"
    for route in route_retriever:
        route_context += route["context"].strip()
        route_context += "\n\n"
    logging.info(f"查询22239知识库：{req_data['query']}")
    logging.info(f"查询22239知识库，召回结果{leaf_context+route_context}")
    return {"retriever": leaf_context+route_context}




@app.post("/fenceng")
async def multi_layer_rag(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])
    # context = fenceng_RAG(query=req_data["query"], Doc1=Doc_1, Doc2=Doc_2, Doc3=Doc_3, Doc4=Doc_4, RA=RA, md_header_splits=md_header_splits)
    best_files = select_doc_llm(req_data["query"])
    best_files = ast.literal_eval(best_files)
    if '22239' in best_files:
        best_files.remove('22239')
    print(f"查询到相似文档列表：{best_files}")
    if best_files == []:
        return {"retriever": []}
    retriever = []
    for i,file in enumerate(best_files):
        context = fenceng_pool[file].retrieve(query=req_data["query"], doc_name=file, top_k=10-i*1)
        retriever.extend(context)
    # results = "\n\n".join(retriever)
    # return {"retriever": results}
    res = rerank_docs(req_data["query"], [content["context"] for content in retriever], 5)
    retriever_list = [retriever[content["index"]] for content in res]
    return {"retriever": retriever_list}

@app.get("/hello")
def hello_world():
    return "hello world"




@app.post("/retrieve_new")
async def dev_retrieve(req:Request):
    log_str = ""
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])
    log_str += f"查询多文档知识库：{req_data['query']}\n"
    for i in range(3):
        try:
            queries = generate_queries(req_data["query"])
            if not queries:
                raise ValueError("生成失败")
            print("生成query成功")
            break
        except:
            queries = [req_data["query"]]
            print("生成query失败，继续")
            continue
    multi_doc_retriever = dict()
    retriever = []
    for query in queries:
        log_str += f"生成子查询：{query}\n"

        best_files = select_doc_llm(query)
        best_files = ast.literal_eval(best_files)
        if '22239' in best_files:
            best_files.remove('22239')
        print(f"查询到相似文档列表：{best_files}")
        log_str += f"查询多文档知识库，相关文档{best_files}\n"

        if best_files == []:
            return {"retriever": []}

        with_route = False
        for i,file in enumerate(best_files):
            raptor_key_tocall = file
            if raptor_pool.get(file,"") == "":
                done = False
                for key in raptor_pool.keys():
                    if key in file:
                        raptor_key_tocall = key
                        done = True
                        break
                if not done:
                    continue
            bm25_retriever = BM25Retriever(file_path=f"./demo/raptor-trees/{raptor_key_tocall}", cache_path=f"demo/token_database/{raptor_key_tocall}_cache")
            bm25_idx_results = bm25_retriever.rank(query)
            bm25_idx_results = bm25_idx_results[0:10-i*3]
            bm25_results = [
                {
                    "context": bm25_retriever.processed_docs[item[0]].text,
                    "type": "leaf",
                    "index": item[0]
                } for item in bm25_idx_results
            ]
            leaf_context, route_context, layer_information = raptor_pool[raptor_key_tocall].retrieve(query, collapse_tree=False, return_layer_information=True, top_k=8-i*3)

            all_retrievers = [bm25_results]
            if with_route:
                all_retrievers.append(route_context+leaf_context)
            else:
                all_retrievers.append(leaf_context)
            for item in all_retrievers:
                for node in item:
                    node["document"] = raptor_key_tocall
            doc_retriever = deduplicate(all_retrievers)

            if multi_doc_retriever.get(raptor_key_tocall,[]) == []:
                multi_doc_retriever[raptor_key_tocall] = [doc_retriever]
            else:
                multi_doc_retriever[raptor_key_tocall].append(doc_retriever)

    for k,v in multi_doc_retriever.items():
        print("单个文档的召回结果")
        print(v)
        tmp = deduplicate(v)
        multi_doc_retriever[k] = tmp
        retriever.extend(tmp[0:5])

    # print(retriever)
    res = rerank_docs(req_data["query"], [content["context"] for content in retriever],8)
    retriever_list = [retriever[content["index"]] for content in res]
    for k,v in multi_doc_retriever.items():
        if v[0] not in retriever_list:
            retriever_list.append(v[0])

    # retriever_list = retriever

    log_str += f"查询多文档知识库，召回结果{retriever_list}\n"
    logging.info(log_str)

    return {"retriever": retriever_list}


@app.post("/retrieve")
async def retrieve(req:Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])
    best_files = select_doc_llm(req_data["query"])
    best_files = ast.literal_eval(best_files)
    if '22239' in best_files:
        best_files.remove('22239')
    print(f"查询到相似文档列表：{best_files}")
    if best_files == []:
        return {"retriever": []}
    retriever = []
    with_route = False
    for i,file in enumerate(best_files):
        raptor_key_tocall = file
        if raptor_pool.get(file, "") == "":
            done = False
            for key in raptor_pool.keys():
                if key in file:
                    raptor_key_tocall = key
                    done = True
                    break
            if not done:
                continue
        bm25_retriever = BM25Retriever(file_path=f"./demo/raptor-trees/{raptor_key_tocall}", cache_path=f"demo/token_database/{raptor_key_tocall}_cache")
        bm25_idx_results = bm25_retriever.rank(req_data["query"])
        bm25_idx_results = bm25_idx_results[0:10-i*3]
        bm25_results = [
            {
                "context": bm25_retriever.processed_docs[item[0]].text,
                "type": "leaf",
                "index": item[0]
            } for item in bm25_idx_results
        ]
        leaf_context, route_context, layer_information = raptor_pool[raptor_key_tocall].retrieve(req_data["query"], collapse_tree=False, return_layer_information=True, top_k=8-i*3)
        all_retrievers = [bm25_results]
        if with_route:
            all_retrievers.append(route_context+leaf_context)
        else:
            all_retrievers.append(leaf_context)
        for item in all_retrievers:
            for node in item:
                node["document"] = raptor_key_tocall
        retriever.extend(deduplicate(all_retrievers))
    # print(retriever)
    res = rerank_docs(req_data["query"], [content["context"] for content in retriever],8)
    retriever_list = [retriever[content["index"]] for content in res]
    logging.info(f"查询多文档知识库：{req_data['query']}")
    logging.info(f"查询多文档知识库，相关文档{best_files}")
    logging.info(f"查询多文档知识库，召回结果{retriever_list}")

    return {"retriever": retriever_list}



@app.post("/22239")
async def retrieve_22239(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])

    bm25_retriever = BM25Retriever(file_path="./demo/22239-keywords", cache_path="demo/token_database/22239_cache")
    all_retrievers = []
    query = req_data["query"]
    bm25_idx_results = bm25_retriever.rank(query)
    bm25_idx_results = bm25_idx_results[0:10]
    bm25_results = [
        {
            "context": bm25_retriever.processed_docs[item[0]].text,
            "type": "leaf",
            "index": item[0]
        } for item in bm25_idx_results
    ]
    leaf_retriever, route_retriever, layer_information = raptor_22239.retrieve(req_data["query"], collapse_tree=False,
                                                                           return_layer_information=True, top_k=5)
    all_retrievers.append(bm25_results)
    all_retrievers.append(leaf_retriever+route_retriever)
    # print(queries)
    # print(len(all_retrievers))
    retriever = deduplicate(all_retrievers)
    # print(len(retriever))
    res_rerank = rerank_docs(req_data["query"], [content["context"] for content in retriever],10)
    result = [retriever[content["index"]] for content in res_rerank]
    return {"retriever": result}

# @app.post("/multi-doc")
# async def multi_doc_rag(req: Request):
#     try:
#         req_data = await req.json()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid JSON")
#
#     query = req_data["query"]
#     doc_type_context = dict()
#     # 在每个类别中召回，整理出来
#     for type in doc_type.keys():
#         for idx, doc in enumerate(doc_type[type].keys()):
#             leaf_context, route_context, layer_information = raptor_pool[doc].retrieve(query, collapse_tree=False, return_layer_information=True, top_k=5-idx)
#             if doc_type_context.get(type,"") == "":
#                 doc_type_context[type] = []
#             for item in leaf_context:
#                 item["document"] = doc_type[type][doc]["name"]
#             doc_type_context[type].extend(leaf_context)
#     logging.info(f"每个类别的召回内容：\n{doc_type_context}")
#     type_answer = dict()
#     # 每个类别召回内容重排序，取前5个，然后总结
#     for type in doc_type_context.keys():
#         if len(doc_type_context[type]) > 5:
#             res = rerank_docs(req_data["query"], [content["context"] for content in doc_type_context[type]], 5)
#             retriever_list = [doc_type_context[type][content["index"]] for content in res]
#         else:
#             retriever_list = doc_type_context[type]
#         tmp_prompt = multi_doc.gen_multi_doc_prompt(query,retriever_list)
#         # 总结每个类别召回的内容
#         summary = openai_api_seu(tmp_prompt, cf.llm_models["qwen3-30b-2507"],"qwen3-30b-2507")
#         type_answer[type] = summary
#     logging.info(f"每个类别召回内容的总结：\n{type_answer}")
#     # 整合所有类别的内容，生成总结
#     prompt = multi_doc.gen_multi_doc_prompt(query,type_answer.values())
#     res = openai_api_seu(prompt, cf.llm_models["qwen3-30b-2507"],"qwen3-30b-2507")
#     return {"retriever": res}

@app.post("/fusion/22239")
async def fusion_rag(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    fenceng_context = fenceng_pool["22239"].retrieve(query=req_data["query"], doc_name='22239', top_k=10)
    bm25_retriever = BM25Retriever(file_path="./demo/22239-keywords", cache_path="demo/token_database/22239_cache")
    all_retrievers = []
    query = req_data["query"]
    bm25_idx_results = bm25_retriever.rank(query)
    bm25_idx_results = bm25_idx_results[0:10]
    bm25_results = [
        {
            "context": bm25_retriever.processed_docs[item[0]].text,
            "type": "leaf",
            "index": item[0],
            "global_index" : bm25_retriever.processed_docs[item[0]].global_index,

        } for item in bm25_idx_results
    ]
    leaf_retriever, route_retriever, layer_information = raptor_22239.retrieve(req_data["query"], collapse_tree=False,
                                                                               return_layer_information=True, top_k=5)
    all_retrievers.append(leaf_retriever)
    all_retrievers.append(bm25_results)
    retriever = deduplicate(all_retrievers)
    print(retriever)
    # fenceng 和 raport去重
    retriever_map = dict()
    res = []
    for item in retriever:
        if item["global_index"] == -1:
            res.append(item)
        else:
            if retriever_map.get(item["global_index"]) is None:
                retriever_map[item["global_index"]] = 1
                res.append(item)

    for item in fenceng_context:
        if retriever_map.get(item["index"]) is None:
            retriever_map[item["index"]] = 1
            res.append(item)

    # print(len(retriever))
    res.extend(route_retriever)
    print(res)
    res_rerank = rerank_docs(req_data["query"], [content["context"] for content in res], 15)
    print(res_rerank)
    result = [res[content["index"]] for content in res_rerank]
    result_context = []
    for item in result:
        result_context.append({"context":item["context"]})
    return {"retriever": result_context,"filename":"GB/T 22239-2019《信息安全技术 网络安全等级保护基本要求》"}


@app.post("/fusion/retrieve")
async def retrieve_fusion(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])

    best_files = select_doc_llm(req_data["query"])
    best_files = ast.literal_eval(best_files)
    if '22239' in best_files:
        best_files.remove('22239')
    print(f"查询到相似文档列表：{best_files}")
    if best_files == []:
        return {"retriever": []}
    retriever = []
    with_route = False
    for i,file in enumerate(best_files):
        raptor_key_tocall = file
        if raptor_pool.get(file, "") == "":
            done = False
            for key in raptor_pool.keys():
                if key in file:
                    raptor_key_tocall = key
                    done = True
                    break
            if not done:
                continue
        # bm25检索
        bm25_retriever = BM25Retriever(file_path=f"./demo/raptor-trees/{raptor_key_tocall}", cache_path=f"demo/token_database/{raptor_key_tocall}_cache")
        bm25_idx_results = bm25_retriever.rank(req_data["query"])
        bm25_idx_results = bm25_idx_results[0:10-i*3]
        bm25_results = [
            {
                "context": bm25_retriever.processed_docs[item[0]].text,
                "type": "leaf",
                "index": item[0]
            } for item in bm25_idx_results
        ]
        # raptor检索，并与bm25合并去重
        leaf_context, route_context, layer_information = raptor_pool[raptor_key_tocall].retrieve(req_data["query"], collapse_tree=False, return_layer_information=True, top_k=8-i*3)
        raptor_bm25_retrievers = [bm25_results]
        if with_route:
            raptor_bm25_retrievers.append(route_context+leaf_context)
        else:
            raptor_bm25_retrievers.append(leaf_context)
        for item in raptor_bm25_retrievers:
            for node in item:
                node["document"] = files_dict[raptor_key_tocall]
        raptor_bm25_retrievers = deduplicate(raptor_bm25_retrievers)
        retriever.extend(raptor_bm25_retrievers)

        # 分层rag检索
        fenceng_context = fenceng_pool[raptor_key_tocall].retrieve(query=req_data["query"], doc_name=file, top_k=10 - i * 1)
        # 通过embedding判断来去重
        tmp = []
        for item in fenceng_context:
            ok  = True
            for item2 in raptor_bm25_retrievers:

                embedding = utils.get_embeddings([raptor_pool[raptor_key_tocall].tree.all_nodes[item2["index"]]], 'EMB')[0]
                va, vb = np.asarray(item["embeddings"]), embedding
                dis = 1 - np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                if dis < 0.1:
                    ok = False
                    break
            if ok:
                tmp.append(item)
        for item in tmp:
            item["document"] = files_dict[raptor_key_tocall]
        retriever.extend(tmp)

    # print(retriever)
    res = rerank_docs(req_data["query"], [content["context"] for content in retriever],8)
    retriever_list = [retriever[content["index"]] for content in res]
    context = [{
        "context": item["context"],
        "document": item["document"]
    } for item in retriever_list]

    return {"retriever": context}




def deduplicate(retriever_list:List[List[dict]]):
    result = []
    node_map = dict()
    for retriever in retriever_list:
        for item in retriever:
            if node_map.get(item["index"],None):
                continue
            result.append(item)
            node_map[item["index"]] = 1
    return result


if __name__ == "__main__":
    uvicorn.run(
        "app:app", # 文件名 : fastapi实例名
        host="0.0.0.0",
        port=8000,
        reload=False           # 开发阶段用
    )
