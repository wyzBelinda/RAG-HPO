# RAG-HPO (dev)

HPO 表型提取管线 — 使用 LLM + RAG 从临床文本中自动提取 Human Phenotype Ontology 术语。

这是原始 Jupyter Notebook (`RAG-HPO.ipynb`) 的重写版本，拆分为独立 Python 模块，支持 CLI、HTTP API、Docker 三种运行方式。

## 项目结构

```
dev/
├── rag_hpo.py          # 主管线入口 (CLI)，包含所有核心函数
├── config.py           # 集中参数管理 (pydantic-settings)
├── pipeline.py         # Pipeline 类 — 可复用核心逻辑 (CLI + HTTP 共享)
├── server.py           # FastAPI 应用
├── schemas.py          # API 请求/响应 Pydantic 模型
├── build_vector_db.py  # 构建 HPO 向量数据库 (hpo_meta.json + hpo_embedded.npz)
├── postprocess.py      # 后处理工具 (替换废弃 HPO 词条 + enrich SNOMED/UMLS)
├── http_utils.py       # HTTP 重试 + 连接池工具
└── inspect_ontology.py # 随机抽查 HPO 本体 term
```

根目录数据文件（运行时依赖）:

| 文件 | 大小 | 用途 |
|---|---|---|
| `hp.obo` | 10 MB | HPO 本体 OBO 文件 |
| `hpo_meta.json` | 31 MB | HPO term 元数据 (lineage, organ_system, definition) |
| `hpo_embedded.npz` | 61 MB | HPO term 向量 (float16, SapBERT) |
| `hpo_terms_full.csv` | 158 MB | HPO 全量 term 表格 |
| `system_prompts.json` | ~2 KB | LLM 系统提示词 |

## 管线流程

```
输入 (clinical_note)
  │
  ▼
阶段 I — LLM 表型提取 (system_message_I)
  │  每句 note → LLM → [{phrase, category}]
  │  Filter: 仅保留 category=="Abnormal"
  │
  ▼
本地 HPO 匹配 (FAISS 向量检索 + rapidfuzz)
  │  精确匹配 → exact_df
  │  非精确匹配 → non_exact_df
  │
  ▼
阶段 II — LLM HPO 映射 (system_message_II)
  │  仅处理 non_exact 条目
  │  LLM + 本地 fallback (cluster_index + fuzzy)
  │
  ▼
输出 ─── [{patient_id, phrase, category, hpo_id}]
```

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements-docker.txt
# 或
uv pip install -r requirements-docker.txt
```

### 2. 配置凭证

```bash
# .env 文件 (项目根目录)
RAG_HPO_API_KEY=sk-xxx
RAG_HPO_BASE_URL=https://api.deepseek.com
RAG_HPO_MODEL=deepseek-v4-pro
RAG_HPO_TEMPERATURE=0.2
```

所有参数见 `config.py`，可通过环境变量 `RAG_HPO_*` 覆盖。优先级: CLI > .env > 默认值。

### 3. CLI 模式

```bash
# 单文件处理
python dev/rag_hpo.py --csv Test_Cases.csv --output results.csv --output-mode save

# 交互式 (手动输入笔记)
python dev/rag_hpo.py
```

### 4. HTTP API 模式

```bash
uvicorn dev.server:app --host 0.0.0.0 --port 8000
```

| 方法 | 端点 | 说明 |
|---|---|---|
| POST | `/api/v1/extract` | 提交临床笔记，返回 job_id |
| GET | `/api/v1/jobs/{job_id}` | 查询任务状态和结果 |
| GET | `/api/v1/health` | 健康检查 |

```bash
# 提交
curl -X POST http://localhost:8000/api/v1/extract \
  -H 'Content-Type: application/json' \
  -d '{"notes": [{"patient_id": "1", "clinical_note": "A 44-year-old obese man..."}]}'

# 查询结果
curl http://localhost:8000/api/v1/jobs/{job_id}
```

服务启动时一次性加载嵌入模型和 FAISS 索引（~1.5GB 内存），后续请求直接复用。

### 5. Docker 模式

```bash
docker build --platform linux/amd64 -t rag-hpo:latest .
docker run -d --name rag-hpo -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/data \
  rag-hpo:latest
```

镜像预构建了 HPO 向量库和 SapBERT 嵌入模型，不需要额外下载。Dockerfile 使用 CPU-only torch 保持镜像精简。

## 配置参数

全部参数集中在 `config.py`，支持通过 `RAG_HPO_*` 环境变量覆盖。

**LLM**

| 参数 | 默认值 | 环境变量 |
|---|---|---|
| `api_key` | None | `RAG_HPO_API_KEY` |
| `base_url` | `https://api.groq.com/...` | `RAG_HPO_BASE_URL` |
| `model` | `llama3-groq-70b-...` | `RAG_HPO_MODEL` |
| `temperature` | 0.7 | `RAG_HPO_TEMPERATURE` |
| `max_tokens_per_day` | 500000 | `RAG_HPO_MAX_TOKENS_PER_DAY` |
| `max_queries_per_minute` | 30 | `RAG_HPO_MAX_QUERIES_PER_MINUTE` |

**嵌入模型**

| 参数 | 默认值 |
|---|---|
| `embedding_model` | `pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb` |
| `embedding_backend` | `auto` (可选: `sentence-transformers`, `fastembed`) |

**检索**

| 参数 | 默认值 |
|---|---|
| `retrieval_top_k` | 500 |
| `retrieval_similarity_threshold` | 0.35 |
| `fuzzy_match_threshold` | 80 |

**Pipeline 控制**

| 参数 | 默认值 |
|---|---|
| `checkpoint_interval_notes` | 5 |
| `checkpoint_interval_hpo` | 50 |

## 后处理

```bash
# 替换已废弃的 HPO 词条
python dev/postprocess.py replace-obsolete results.csv -o results_updated.csv

# 合并 SNOMED CT / UMLS 信息
python dev/postprocess.py enrich results.csv hpo_terms_full.csv -o results_enriched.csv
```

## 嵌入模型选型

`initialize_embeddings_model()` 不再限制特定模型。设置 `RAG_HPO_EMBEDDING_MODEL` 即可切换:

```bash
# 默认 SapBERT (sentence-transformers 后端)
RAG_HPO_EMBEDDING_MODEL=pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb

# BGE (fastembed 后端, 自动检测)
RAG_HPO_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

## 技术要点

- **FAISS IndexFlatIP**: cosine 相似度检索 (L2 归一化后内积)
- **断点续传**: 每 N 条自动保存 pickle checkpoint 到 `tmp/`
- **LLM fallback**: JSON 解析 → 列表兜底 → 正则提取，三层保护
- **HPO 匹配**: 精确 token 重排 → 子串搜索 → rapidfuzz token_sort_ratio (阈值 80)
- **CPU-only torch**: Docker 镜像刻意使用 CPU 版 torch 保持精简，GPU 切换见下方

### 切换到 GPU

1. Dockerfile 去掉 `--extra-index-url https://download.pytorch.org/whl/cpu`
2. `requirements-docker.txt` 中 `faiss-cpu` → `faiss-gpu`
3. 运行时 `docker run --gpus all`
4. `sentence-transformers` 自动检测 `torch.cuda.is_available()`，无需改代码
