from raptor.kb_builder import KBBuilder
from raptor.common_retriever import CommonRetriever
from example.doc_processing import load_markdown_custom, split_doc
import pickle
from langchain.schema import Document

class CustomRA:
    def __init__(self, config=None, docs_path=None, md_path=None, init = False):
        custom_embedding = config.embedding_model
        self.kb_builder = KBBuilder(config)
        self.md_path = md_path
        if init == False:
            docs_path_1 = docs_path + "_1"
            with open(docs_path_1, "rb") as file:
                self.metadata_docs1 = pickle.load(file).all_nodes
            docs_path_2 = docs_path + "_2"
            with open(docs_path_2, "rb") as file:
                self.metadata_docs2 = pickle.load(file).all_nodes
            docs_path_3 = docs_path + "_3"
            with open(docs_path_3, "rb") as file:
                self.metadata_docs3 = pickle.load(file).all_nodes
            docs_path_4 = docs_path + "_4"
            with open(docs_path_4, "rb") as file:
                self.metadata_docs4 = pickle.load(file).all_nodes
            docs_path_0 = docs_path + "_0"
            with open(docs_path_0, "rb") as file:
                self.metadata_docs = pickle.load(file).all_nodes
            self.retriever = CommonRetriever(embedding_model=custom_embedding)
        else:
            self.add_documents(docs_path)

    def add_documents(self, docs_path):
        self.kb_builder.build_kb(docs_path, self.md_path)

    def retrieve(
            self,
            query,
            doc_name,
            top_k: int = 10,
    ):
        if self.retriever is None:
            raise ValueError(
                "The Retriever instance has not been initialized. Call 'add_documents' first."
            )

        documents = load_markdown_custom(self.md_path)
        Doc_1, Doc_2, Doc_3, Doc_4, indexed_splits, md_header_splits = split_doc(documents)

        doc_list = []
        index_list = []
        docs0 = []
        docs1 = []
        docs2 = []
        docs3 = []
        docs4 = []
        selected_context = []
        all_embed = [doc.all_embed for doc in self.metadata_docs]

        if "第一级" in query:
            docs1, index1 = self.retriever.retrieve_information(query, self.metadata_docs1, top_k=top_k)
            for ind in index1:
                chunk_ind = Doc_1[ind].metadata['chunk_index']
                pg_content = Doc_1[ind].page_content
                hd = self.metadata_docs1[ind].doc.page_content
                new_content = hd + '\n' + pg_content
                doc = Document(page_content=new_content)
                doc_list.append(doc)
                index_list.append(ind)
                selected_context.append({"context": new_content, "index": chunk_ind, "document": doc_name, "embeddings": all_embed[chunk_ind]})
        if "第二级" in query:
            docs2, index2 = self.retriever.retrieve_information(query, self.metadata_docs2, top_k=top_k)
            for ind in index2:
                chunk_ind = Doc_2[ind].metadata['chunk_index']
                pg_content = Doc_2[ind].page_content
                hd = self.metadata_docs2[ind].doc.page_content
                new_content = hd + '\n' + pg_content
                doc = Document(page_content=new_content)
                doc_list.append(doc)
                index_list.append(ind)
                selected_context.append({"context": new_content, "index": chunk_ind, "document": doc_name, "embeddings": all_embed[chunk_ind]})
        if "第三级" in query:
            docs3, index3 = self.retriever.retrieve_information(query, self.metadata_docs3, top_k=top_k)
            for ind in index3:
                chunk_ind = Doc_3[ind].metadata['chunk_index']
                pg_content = Doc_3[ind].page_content
                hd = self.metadata_docs3[ind].doc.page_content
                new_content = hd + '\n' + pg_content
                doc = Document(page_content=new_content)
                doc_list.append(doc)
                index_list.append(ind)
                selected_context.append({"context": new_content, "index": chunk_ind, "document": doc_name, "embeddings": all_embed[chunk_ind]})
        if "第四级" in query:
            docs4, index4 = self.retriever.retrieve_information(query, self.metadata_docs4, top_k=top_k)
            for ind in index4:
                chunk_ind = Doc_4[ind].metadata['chunk_index']
                pg_content = Doc_4[ind].page_content
                hd = self.metadata_docs4[ind].doc.page_content
                new_content = hd + '\n' + pg_content
                doc = Document(page_content=new_content)
                doc_list.append(doc)
                index_list.append(ind)
                selected_context.append({"context": new_content, "index": chunk_ind, "document": doc_name, "embeddings": all_embed[chunk_ind]})
        if "第一级" not in query and "第二级" not in query and "第三级" not in query and "第四级" not in query:
            docs0, index0 = self.retriever.retrieve_information(query, self.metadata_docs, top_k=top_k)
            for ind in index0:
                chunk_ind = indexed_splits[ind].metadata['chunk_index']
                pg_content = md_header_splits[ind].page_content
                hd = self.metadata_docs[ind].doc.page_content
                new_content = hd + '\n' + pg_content
                doc = Document(page_content=new_content)
                doc_list.append(doc)
                index_list.append(ind)
                selected_context.append({"context": new_content, "index": chunk_ind, "document": doc_name, "embeddings": all_embed[chunk_ind]})

        context = [(doc.page_content) for doc in doc_list]
        print(f"检索到的内容文档：{context}")

        results = "\n\n".join(context)

        return selected_context