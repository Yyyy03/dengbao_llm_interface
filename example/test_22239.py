from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from example.build_tree import CustomQAModel, CustomEmbeddingModel, CustomSummarizationModel
import pandas as pd
import json

# 整合自定义模型
# Initialize your custom models
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
    tree_builder_type="markdown"
)

RA = RetrievalAugmentation(tree="../demo/22239-new", config=custom_config)

def test(query):
    # query = "定级备案的要求有哪些？"
    ## query改写
    query_prompt = f"##任务"\
    "请你将用户提出的问题进行提取关键内容，用于进行检索知识库内容。请提取对检索召回最有价值的关键词，例如专有名词、专业术语或关键对象，能够帮助精准定位相关文档或知识点。关键词应能准确反映问题的核心内容，避免提取泛泛无意义的词。"\
    "## 用户提问"\
    f"{query}"\
    "##输出"\
    "输出多个关键词或关键句，关键词和关键句之间用空格隔开，无需其他多余文字。"
    new_query = custom_qa.normal_chat("你是一个文字助手",query_prompt)
    new_query = new_query.split("</think>")[-1].strip()
    answer,layer_info, context = RA.answer_question(question=new_query,return_layer_information=True,collapse_tree=False)
    print(new_query)
    print(f"llm回答：\n{answer}")
    # print(f"召回上下文：\n{context}")
    # print(f"层次查找路径：\n{layer_info}")
    return answer,context,layer_info,new_query

if __name__ == "__main__":
    data = pd.read_excel("../testQuestion.xlsx",header=0)
    # print(data["问题"])
    test_data = []
    for q in data["问题"]:
        ans, context, layer,query=test(q)
        tmp=dict()
        tmp["question"] = q
        tmp["query"] = query
        tmp["answer"] = ans
        tmp["retrieve"] = context
        test_data.append(tmp)
    with open("../test_data_nokeywords.json", 'w', encoding="utf8") as f:
        json.dump(test_data,f,ensure_ascii=False)
