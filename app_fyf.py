from fastapi import FastAPI, Request, HTTPException
import asyncio
import uvicorn
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.common_retriever import CommonRetriever, fenceng_RAG
from example.build_tree import CustomQAModel, CustomEmbeddingModel, CustomSummarizationModel
from example.doc_processing import load_markdown_custom, split_doc

app = FastAPI(title="Async Demo")

# server启动
summary_model = "qwen2.5-72b"
summary_model_key = "39c9632a-f67b-4e7c-96f7-9b89faead338"
qa_model = "qwen3-32b"
qa_model_key = "fef51a476780489d8c1006f2fbd2f6"
embedding_model = "bge-m3"
embedding_model_key = "d6a99ca07de74c73aa8fa53cd1d2826"
custom_summarizer = CustomSummarizationModel(summary_model,summary_model_key)
custom_qa = CustomQAModel(qa_model,qa_model_key)
custom_embedding = CustomEmbeddingModel(embedding_model,embedding_model_key)

# Create a config with your custom models
custom_config = RetrievalAugmentationConfig(
    summarization_model=custom_summarizer,
    qa_model=custom_qa,
    embedding_model=custom_embedding,
)

RA = CommonRetriever(embedding_model=custom_embedding)

markdown_path = "./demo/22239-new.md"
documents = load_markdown_custom(markdown_path)
Doc_1, Doc_2, Doc_3, Doc_4, indexed_splits, md_header_splits = split_doc(documents)
'''
{
"query": "用户的知识库查询",
}
'''
@app.post("/app_fyf")
async def run_task(req: Request):
    try:
        req_data = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    print(req_data["query"])
    context = fenceng_RAG(query=req_data["query"], Doc1=Doc_1, Doc2=Doc_2, Doc3=Doc_3, Doc4=Doc_4, RA=RA, md_header_splits=md_header_splits)
    return {"retriever": context}

if __name__ == "__main__":
    uvicorn.run(
        "app_fyf:app",
        host="0.0.0.0",
        port=8000,
        reload=True           # 开发阶段用
    )