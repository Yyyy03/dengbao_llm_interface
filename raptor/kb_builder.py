from example.doc_processing import load_markdown_custom, split_doc, KnowledgeBase, DocumentNode, build_metadata_text
import pickle
from langchain.schema import Document

class KBBuilder():
    def __init__(self, config):
        self.custom_embedding = config.embedding_model

    def build_single_kb(self, docs):
        metadata_texts = [build_metadata_text(doc.metadata) for doc in docs]
        metadata_docs = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(metadata_texts)]

        KB_all_nodes = [
            DocumentNode(
                doc=doc,
                embeddings=self.custom_embedding.create_embedding(doc.page_content),
                all_embed=self.custom_embedding.create_embedding(doc.page_content+docs[doc.metadata["source"]].page_content)
            )
            for doc in metadata_docs
        ]

        KB = KnowledgeBase(all_nodes=KB_all_nodes)
        return KB

    def save(self, path, kb):
        with open(path, "wb") as file:
            pickle.dump(kb, file)

    def build_kb(self, docs_path, md_path):
        documents = load_markdown_custom(md_path)
        Doc_1, Doc_2, Doc_3, Doc_4, indexed_splits, md_header_splits = split_doc(documents)
        docs_path_1 = docs_path + "_1"
        self.save(path = docs_path_1, kb = self.build_single_kb(Doc_1))
        docs_path_2 = docs_path + "_2"
        self.save(path=docs_path_2, kb=self.build_single_kb(Doc_2))
        docs_path_3 = docs_path + "_3"
        self.save(path=docs_path_3, kb=self.build_single_kb(Doc_3))
        docs_path_4 = docs_path + "_4"
        self.save(path=docs_path_4, kb=self.build_single_kb(Doc_4))
        docs_path_0 = docs_path + "_0"
        self.save(path=docs_path_0, kb=self.build_single_kb(md_header_splits))
