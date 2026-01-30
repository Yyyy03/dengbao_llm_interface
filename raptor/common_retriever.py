from typing import Dict, List, Set
from langchain.schema import Document
from .utils import distances_from_embeddings
import tiktoken
from example.doc_processing import build_metadata_text

def get_docs_list(docs):
    embeddings = [doc.embeddings for doc in docs]
    doc_list = [doc.doc.page_content for doc in docs]
    return embeddings, doc_list

class CommonRetriever():
    def __init__(self, embedding_model) -> None:
        self.embedding_model = embedding_model
        tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tokenizer = tokenizer

    def create_embedding(self, text: str) -> List[float]:
        """
        Generates embeddings for the given text using the specified embedding model.

        Args:
            text (str): The text for which to generate embeddings.

        Returns:
            List[float]: The generated embeddings.
        """
        return self.embedding_model.create_embedding(text)

    def retrieve_information(self, query: str, docs: Document, top_k: int=10, max_tokens: int=3000):
        """
        Retrieves the most relevant information based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The retrieved context.
        """

        query_embedding = self.create_embedding(query)

        embeddings, docs_list = get_docs_list(docs)

        distances = distances_from_embeddings(query_embedding, embeddings)

        sorted_indices = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]

        total_tokens = 0

        selected_docs = []

        selected_indices = []

        for idx in sorted_indices[:top_k]:

            context = docs_list[idx]
            # node_tokens = len(self.tokenizer.encode(context))

            # if total_tokens + node_tokens > max_tokens:
            #     break

            selected_docs.append(context)
            selected_indices.append(idx)
            # total_tokens += node_tokens

        return selected_docs, selected_indices

def fenceng_RAG(query, Doc1, Doc2, Doc3, Doc4, RA, md_header_splits):

    metadata_texts = [build_metadata_text(doc.metadata) for doc in md_header_splits]
    metadata_docs = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts)]

    metadata_texts1 = [build_metadata_text(doc.metadata) for doc in Doc1]
    metadata_docs1 = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts1)]

    metadata_texts2 = [build_metadata_text(doc.metadata) for doc in Doc2]
    metadata_docs2 = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts2)]

    metadata_texts3 = [build_metadata_text(doc.metadata) for doc in Doc3]
    metadata_docs3 = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts3)]

    metadata_texts4 = [build_metadata_text(doc.metadata) for doc in Doc4]
    metadata_docs4 = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts4)]

    doc_list = []
    docs0 = []
    docs1 = []
    docs2 = []
    docs3 = []
    docs4 = []

    if "第一级" in query:
        docs1, index1 = RA.retrieve_information(query, metadata_docs1, top_k=10)
        for ind in index1:
            pg_content = Doc1[ind].page_content
            hd = metadata_docs1[ind].page_content
            new_content = hd + '\n' + pg_content
            doc = Document(page_content=new_content)
            doc_list.append(doc)
    if "第二级" in query:
        docs2, index2 = RA.retrieve_information(query, metadata_docs2, top_k=10)
        for ind in index2:
            pg_content = Doc2[ind].page_content
            hd = metadata_docs2[ind].page_content
            new_content = hd + '\n' + pg_content
            doc = Document(page_content=new_content)
            doc_list.append(doc)
    if "第三级" in query:
        docs3, index3 = RA.retrieve_information(query, metadata_docs3, top_k=10)
        for ind in index3:
            pg_content = Doc3[ind].page_content
            hd = metadata_docs3[ind].page_content
            new_content = hd + '\n' + pg_content
            doc = Document(page_content=new_content)
            doc_list.append(doc)
    if "第四级" in query:
        docs4, index4 = RA.retrieve_information(query, metadata_docs4, top_k=10)
        for ind in index4:
            pg_content = Doc4[ind].page_content
            hd = metadata_docs4[ind].page_content
            new_content = hd + '\n' + pg_content
            doc = Document(page_content=new_content)
            doc_list.append(doc)
    if "第一级" not in query and "第二级" not in query and "第三级" not in query and "第四级" not in query:
        docs0, index0 = RA.retrieve_information(query, metadata_docs, top_k=10)
        for ind in index0:
            pg_content = md_header_splits[ind].page_content
            hd = metadata_docs[ind].page_content
            new_content = hd + '\n' + pg_content
            doc = Document(page_content=new_content)
            doc_list.append(doc)

    context = [(doc.page_content) for doc in doc_list]
    print(f"检索到的内容文档：{context}")

    results = "\n\n".join(context)

    return results