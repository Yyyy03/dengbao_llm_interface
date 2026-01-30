# DengbaoRAG 接口服务

一个面向合规/标准类文档的检索增强服务（FastAPI），支持多文档检索、分层检索、BM25 + 向量融合、重排序与多查询扩展，适合构建企业内部知识库问答或检索型应用。

## 特性概览

- 多文档召回：先选文档，再做细粒度检索
- 分层检索：按目录/层级结构定位上下文
- 融合检索：BM25 与向量检索融合去重
- 重排序：对召回结果进行语义重排
- 多查询扩展：自动生成子查询提高覆盖率
- 结构化返回：接口统一输出 `retriever` 列表

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置模型与 API Key

复制配置模板并填写模型与 Key：

```bash
cp config.py.example config.py
```

> `config.py` 中包含 LLM、Embedding、Reranker 的模型名与 Key。

### 3) 准备索引与数据

服务启动时会读取本地索引与分层数据（默认在 `./demo` 下）：

- 树状索引：`./demo/raptor-trees/<doc>`
- 分层索引：`./demo/fenceng/<doc>`
- Markdown 文档：`./demo/mds/<doc>.md`
- 缓存目录：`./demo/token_database`

如果你需要替换/新增文档，请保证对应目录结构与文件命名一致。

### 4) 启动服务

```bash
python app.py
```

默认监听 `0.0.0.0:8000`。

## 接口说明

> 请求体统一为：`{"query": "你的问题"}`

- `POST /raptor/22239`
  - 单文档检索（叶节点 + 路径节点）
- `POST /fenceng`
  - 多文档分层检索 + 重排序
- `POST /retrieve_new`
  - 多查询扩展 + 多文档融合检索 + 重排序
- `POST /retrieve`
  - 多文档融合检索 + 重排序（简化版）
- `POST /22239`
  - 单文档融合检索（BM25 + 树检索）
- `POST /fusion/22239`
  - 单文档分层 + 融合检索 + 去重
- `POST /fusion/retrieve`
  - 多文档分层 + 融合检索 + 去重
- `GET /hello`
  - 健康检查

返回示例：

```json
{
  "retriever": [
    {"context": "...", "type": "leaf", "index": 12, "document": "..."}
  ]
}
```

## 模型与算法说明（简要）

- 默认使用本地配置的 LLM/Embedding/Reranker 模型
- 支持按目录/层级结构进行检索召回
- 结合稀疏检索（BM25）与向量检索提高召回覆盖
- 最终通过重排序进行结果精排

## 目录结构（核心）

```
.
├── app.py                  # FastAPI 服务入口
├── config.py.example        # 模型配置模板
├── demo/                    # 索引与示例数据
├── example/                 # 自定义模型示例
├── bm25.py                  # BM25 检索实现
├── multi_doc.py             # 多文档检索逻辑
└── requirements.txt
```

## 自定义扩展

如需替换模型或自定义检索流程，可从 `example/` 中的实现入手，并在 `app.py` 中注入对应模型与配置。

---

