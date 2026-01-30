# -*- coding: utf-8 -*-
import math
import re
from collections import defaultdict, Counter
import jieba  # 中文分词库
import jieba.analyse  # 用于关键词提取
import pickle

class Block:
    """
    Represents a node in the bm25 retriever dataset
    """

    def __init__(self, text: str, index: int, tokens, length: int, global_idx=-1) -> None:
        self.text = text
        self.index = index
        self.tokens = tokens
        self.tf = 0
        self.length = length
        self.global_index = global_idx



class BM25Chinese:
    def __init__(self, documents, doc_names, k1=1.5, b=0.75, stopwords=None, cache_path=None):
        """
        初始化中文BM25模型
        :param documents: 中文文档列表，每个文档是一个字符串
        :param k1: 调节词频饱和度的参数
        :param b: 调节文档长度对评分影响的参数
        :param stopwords: 停用词列表， None则使用默认停用词
        """
        self.documents = documents
        self.doc_names = doc_names
        self.k1 = k1
        self.b = b

        # 加载停用词
        self.stopwords = self._load_stopwords() if stopwords is None else stopwords
        self.cache_path = cache_path
        if cache_path and self._load_cache():
            print(f"已从缓存 {cache_path} 加载预处理数据")
        else:
            print("开始预处理文档并缓存...")
            self.process_all_docs()  # 预处理所有文档
            if cache_path:
                self._save_cache()  # 保存到缓存文件
                print(f"预处理数据已保存至 {cache_path}")
        # 预处理文档


        # 计算文档长度


    def _load_stopwords(self):
        """加载默认停用词（可以根据需要扩展）"""
        # 常见中文停用词
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '也', '但', '与', '等', '呢', '啊', '吧', '哦'
        }

    def _preprocess(self, text):
        """中文预处理：去除标点、分词、过滤停用词"""
        # 去除标点和特殊字符
        text = re.sub(r'[^\w\s]', '', text)  # 保留汉字、字母、数字和空格
        text = re.sub(r'\s+', ' ', text).strip()  # 合并空格

        # 分词
        words = jieba.cut(text, cut_all=False)  # 精确模式分词

        # 过滤停用词和空字符串
        filtered_words = [word for word in words if word and word not in self.stopwords]

        return filtered_words
    def process_all_docs(self):
        self.processed_docs = [self._preprocess(doc) for doc in self.documents]
        self.doc_lengths = [len(doc) for doc in self.processed_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        # 计算词频和文档频率
        self._calculate_tf()
        self._calculate_df()

        # 文档总数
        self.num_docs = len(self.documents)


    def _calculate_tf(self):
        """计算每个文档中词的词频"""
        self.tf = []  # tf[i][word] 表示词word在文档i中的词频
        for doc in self.processed_docs:
            counter = Counter(doc)
            self.tf.append(counter)

    def _calculate_df(self):
        """计算每个词的文档频率"""
        self.df = defaultdict(int)  # df[word] 表示包含词word的文档数
        for doc in self.processed_docs:
            unique_words = set(doc)
            for word in unique_words:
                self.df[word] += 1

    def _save_cache(self):
        """将预处理数据保存到本地文件（持久化）"""
        cache_data = {
            'processed_docs': self.processed_docs,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'tf': self.tf,
            'df': self.df,
            'num_docs': self.num_docs,
            'doc_names': self.doc_names
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_cache(self):
        """从本地文件加载预处理数据"""
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            # 恢复缓存数据到实例变量
            self.processed_docs = cache_data['processed_docs']
            self.doc_lengths = cache_data['doc_lengths']
            self.avg_doc_length = cache_data['avg_doc_length']
            self.tf = cache_data['tf']
            self.df = cache_data['df']
            self.num_docs = cache_data['num_docs']
            self.doc_names = cache_data["doc_names"]
            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return False

    def score(self, query_terms, doc_index):
        """计算查询词与指定文档的分数（复用预存数据）"""
        if doc_index < 0 or doc_index >= self.num_docs:
            return 0.0

        doc_tf = self.tf[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0

        for term in query_terms:
            if term not in doc_tf:  # 文档不含该词，直接跳过
                continue

            # 复用预存的DF计算IDF
            df = self.df.get(term, 0)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # 复用预存的TF计算词频项
            tf = doc_tf[term]
            term_freq = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))

            score += idf * term_freq

        return score

    def rank(self, query):
        """对所有文档按与查询的相关性排序"""
        scores = []
        query_items = self._preprocess(query)
        print(query_items)
        for i in range(self.num_docs):
            doc_score = self.score(query_items, i)
            scores.append((i, doc_score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

class BM25Retriever:
    def __init__(self, file_path=None, cache_path=None,k1=1.5, b=0.75,stopwords=None):
        self.k1 = k1
        self.b = b
        self.stopwords = self._load_stopwords() if stopwords is None else stopwords
        self.cache_path = cache_path
        if self.cache_path and self._load_cache():
            print(f"已从缓存 {cache_path} 加载预处理数据")
        else:
            if file_path:
                with open(file_path, 'rb') as f:
                    self.tree = pickle.load(f)
            self.processed_docs = dict()
            for node in self.tree.leaf_nodes.values():
                token_list = self._preprocess(node.text)
                try:
                    global_idx = node.global_index
                except AttributeError:
                    global_idx = -1
                self.processed_docs[node.index] = Block(node.text, node.index, token_list, len(token_list),global_idx)
            self.doc_lengths = [doc.length for doc in self.processed_docs.values()]
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
            self._calculate_tf()
            self._calculate_df()

            self.num_docs = len(self.processed_docs.values())
            if self.cache_path:
                self._save_cache()
                print(f"预处理数据已保存至 {cache_path}")

    def _load_stopwords(self):
        """加载默认停用词（可以根据需要扩展）"""
        # 常见中文停用词
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '也', '但', '与', '等', '呢', '啊', '吧', '哦'
        }

    def _preprocess(self, text):
        """中文预处理：去除标点、分词、过滤停用词"""
        # 去除标点和特殊字符
        text = re.sub(r'[^\w\s]', '', text)  # 保留汉字、字母、数字和空格
        text = re.sub(r'\s+', ' ', text).strip()  # 合并空格

        # 分词
        words = jieba.cut(text, cut_all=False)  # 精确模式分词

        # 过滤停用词和空字符串
        filtered_words = [word for word in words if word and word not in self.stopwords]

        return filtered_words

    def _calculate_tf(self):
        """计算每个文档中词的词频"""
  # tf[i][word] 表示词word在文档i中的词频
        for block in self.processed_docs.values():
            counter = Counter(block.tokens)
            block.tf = counter

    def _calculate_df(self):
        """计算每个词的文档频率"""
        self.df = defaultdict(int)  # df[word] 表示包含词word的文档数
        for block in self.processed_docs.values():
            unique_words = set(block.tokens)
            for word in unique_words:
                self.df[word] += 1

    def _save_cache(self):
        """将预处理数据保存到本地文件（持久化）"""
        cache_data = {
            'processed_docs': self.processed_docs,
            'avg_doc_length': self.avg_doc_length,
            'df': self.df,
            'num_docs': self.num_docs,

        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def _load_cache(self):
        """从本地文件加载预处理数据"""
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            # 恢复缓存数据到实例变量
            self.processed_docs = cache_data['processed_docs']
            self.avg_doc_length = cache_data['avg_doc_length']
            self.df = cache_data['df']
            self.num_docs = cache_data['num_docs']

            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return False

    def score(self, query_terms, doc_index):
        """计算查询词与指定文档的分数（复用预存数据）"""
        if doc_index < 0 or doc_index >= self.num_docs:
            return 0.0

        doc_tf = self.processed_docs[doc_index].tf
        doc_len = self.processed_docs[doc_index].length
        score = 0.0

        for term in query_terms:
            if term not in doc_tf:  # 文档不含该词，直接跳过
                continue

            # 复用预存的DF计算IDF
            df = self.df.get(term, 0)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # 复用预存的TF计算词频项
            tf = doc_tf[term]
            term_freq = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))

            score += idf * term_freq

        return score

    def rank(self, query):
        """对所有文档按与查询的相关性排序"""
        scores = []
        query_items = self._preprocess(query)
        # print(query_items)
        for i in self.processed_docs.keys():
            doc_score = self.score(query_items, i)
            scores.append((i, doc_score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# 示例用法
if __name__ == "__main__":
    # from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
    # from example.build_tree import CustomQAModel, CustomEmbeddingModel, CustomSummarizationModel
    # import config as cf
    # # 示例中文文档集合
    #
    file_list = ['22240', '28448', '28449', 'JRT0060', 'JRT0071', '关于对网络安全等级保护有关工作事项进一步说明的函',
                 '测评机构在沪工作指引规范手册', '网络安全法', '网络安全等级保护测评高风险判定实施指引']
    # summary_model = "qwen2.5-72b"
    # summary_model_key = cf.llm_models[summary_model]
    # qa_model = "qwen3-32b"
    # qa_model_key = cf.llm_models[qa_model]
    # embedding_model = "bge-m3"
    # embedding_model_key = cf.embedding_models[embedding_model]
    # custom_summarizer = CustomSummarizationModel(summary_model, summary_model_key)
    # custom_qa = CustomQAModel(qa_model, qa_model_key)
    # custom_embedding = CustomEmbeddingModel(embedding_model, embedding_model_key)
    # custom_config = RetrievalAugmentationConfig(
    #     summarization_model=custom_summarizer,
    #     qa_model=custom_qa,
    #     embedding_model=custom_embedding,
    #     tree_builder_type="markdown"
    # )
    # documents = []
    # doc_names = []
    # for f in file_list:
    #     t = RetrievalAugmentation(tree=f"./demo/raptor-trees/{f}", config=custom_config)
    #     doc = ""
    #     for node_idx in t.tree.leaf_nodes:
    #         doc +=t.tree.all_nodes[node_idx].text +'\n'
    #     documents.append(doc)
    #     doc_names.append(f)

    # 创建中文BM25模型
    # bm25 = BM25Chinese([],file_list,cache_path="./demo/bm25_cache.pkl")
    # print(len(bm25.processed_docs[0]))
    # # 测试查询
    # queries = [
    #     "请简述在定级阶段、安全建设阶段、等级测评阶段主要参考的标准和作用是什么？",
    #     "等级保护对二级系统都有哪些关于安全物理环境的基本要求？根据上级部门监管要求，某金融公司的二级系统开展等级保护工作需依据金融行业标准的基本要求，则在安全物理环境方面会增加哪些要求？",
    #     "某三级系统的业务应用系统是使用用户名+口令的方式对登录用户进行身份鉴别的，根据等保相关要求，这是否存在什么问题？如果存在问题的话，一般将此问题的级别判定为高风险、中风险还是低风险？是否存在缓解措施？"
    # ]
    #
    # for query in queries:
    #     print(f"\n查询: {query}")
    #     print("排序结果:")
    #     ranked_results = bm25.rank(query)
    #
    #     for idx, score in ranked_results:
    #         if score > 0:  # 只显示有匹配的文档
    #             print(f"文档 {bm25.doc_names[idx]}: 分数 = {score:.4f}")
    retriever = BM25Retriever(file_path="./demo/raptor-trees/22240", cache_path="demo/token_database/22240_cache")
    res = retriever.rank("请简述在定级阶段、安全建设阶段、等级测评阶段主要参考的标准和作用是什么？")
    print(res)