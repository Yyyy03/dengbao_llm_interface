from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

def load_markdown_custom(file_path: str) -> list[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path})]

def build_metadata_text(metadata: dict) -> str:
    h1 = metadata.get("Header 1", "")
    h2 = metadata.get("Header 2", "")
    h3 = metadata.get("Header 3", "")
    h4 = metadata.get("Header 4", "")
    return f"{h1} > {h2} > {h3} > {h4}"

def split_doc(documents):

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(documents[0].page_content)
    Doc_1 = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, doc in enumerate(md_header_splits) if "第一级" in doc.metadata.get("Header 1", "")
    ]

    Doc_2 = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, doc in enumerate(md_header_splits) if "第二级" in doc.metadata.get("Header 1", "")
    ]

    Doc_3 = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, doc in enumerate(md_header_splits) if "第三级" in doc.metadata.get("Header 1", "")
    ]

    Doc_4 = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, doc in enumerate(md_header_splits) if "第四级" in doc.metadata.get("Header 1", "")
    ]

    indexed_splits = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_index": i}
        )
        for i, doc in enumerate(md_header_splits)
    ]

    return Doc_1, Doc_2, Doc_3, Doc_4, indexed_splits, md_header_splits

class DocumentNode:
    def __init__(self, doc: Document, embeddings) -> None:
        self.doc = doc
        self.embeddings = embeddings

class KnowledgeBase:
    def __init__(self, all_nodes) -> None:
        self.all_nodes = all_nodes