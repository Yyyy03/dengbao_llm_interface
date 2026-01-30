import os
from raptor import BaseSummarizationModel
from raptor import BaseQAModel
from raptor import BaseEmbeddingModel
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
import openai

# 自定义总结模型
class CustomSummarizationModel(BaseSummarizationModel):
    def __init__(self,model,api_key):
        # Initialize your model here
        self.model = model
        self.api_key = api_key


    def summarize(self, context, max_tokens=150):
        # Implement your summarization logic here
        # Return the summary as a string
        try:

            baseurl = "https://openapi.seu.edu.cn/v1"
            client = openai.OpenAI(
                base_url=baseurl,
                api_key=self.api_key
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个文本处理助手"},
                    {
                        "role": "user",
                        "content": f"为下面的文字撰写总结，要用简洁明了的语言，用两句话概括所有关键的内容，保"
                                   f"留核心内容，不要加入详细的描述，以下是文本：{context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e

# 自定义问答模型
class CustomQAModel(BaseQAModel):
    def __init__(self,model_name,api_key):
        # Initialize your model here
        baseurl = "https://openapi.seu.edu.cn/v1"
        self.model = model_name
        self.client = openai.OpenAI(
        base_url=baseurl,
        api_key=api_key
    )
    def answer_question(self, context, question):
        # Implement your QA logic here
        # Return the answer as a string
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个问答助手"},
                {
                    "role": "user",
                    "content": f"基于知识库的内容: {context}， 回答问题：{question}；给出最佳的答案，简洁地输出",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    def normal_chat(self, sys_prompt,question):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": question,
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

# 自定义embedding模型
class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self,model,api_key):
        # Initialize your model here
        self.model = model
        self.client = openai.OpenAI(
            base_url="https://openapi.seu.edu.cn/v1",
            api_key=api_key
        )

    def create_embedding(self, text):
        # Implement your embedding logic here
        # Return the embedding as a numpy array or a list of floats
        text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            # print(response)
            embedding = response.data[0].embedding
        except Exception as e:
            print("embedding error!!")
            print(text)
            print(e)
            raise e
        return embedding

if __name__ == "__main__":
    api_key = "your-openai-api-key"
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
    #
    # # Create a config with your custom models
    custom_config = RetrievalAugmentationConfig(
        summarization_model=custom_summarizer,
        qa_model=custom_qa,
        embedding_model=custom_embedding,
        tree_builder_type="markdown"
    )

    # Initialize RAPTOR with your custom config
    RA = RetrievalAugmentation(config=custom_config)

    dirs = os.listdir("../demo/markdowns/JR")
    for dir in dirs:
        with open(f'../demo/markdowns/JR/{dir}', 'r', encoding='utf-8') as file:
            text = file.read()
        RA.add_documents(text)

        # question = "什么是等保"
        # answer = RA.answer_question(question=question)

        SAVE_PATH = f"../demo/JR/{dir[0:-3]}"
        RA.save(SAVE_PATH)
        print(f"保存成功：{SAVE_PATH}")
