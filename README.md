# LegalMind — 基于 RAG 的法律法规智能检索与问答系统

LegalMind 是一个面向中文法律法规的检索增强生成（RAG）问答系统。它将法律法规解析为结构化的"条"级语料，构建全库统一的向量 + BM25 混合索引，作答时严格依据检索到的条文回答，输出精确到《法名》第X条（含施行日期）的引用与免责声明。

## 功能特性

- 条级引用：答案仅依据检索到的条文生成，标注《法名》第X条与施行日期，不编造法条。
- 混合检索：BM25 关键词（jieba 中文分词）+ 向量语义（DashScope text-embedding-v4）双路召回，可选 LLM 重排进一步提升精度。
- 按法名过滤：可将检索范围限定在某部法规（如"劳动合同法"）内。
- 两种交互方式：Streamlit Web 界面与 click 命令行，复用同一套问答管线。
- 全库统一索引：不按文档分片，检索靠条文自带的元数据（法名、条号、施行日期等）过滤。

## 语料规模

语料来源为 29 部独立 docx 法规 + 1 部 PDF 政策汇编（拆分为多篇）。全库法规元数据共 123 条，去重后实际入库并分块的法规 117 部，合计 6941 个检索块（6599 个条级块 + 342 个散文块）。

## 检索与问答流程

1. 解析：将 data/ 下的 docx 与 PDF 汇编解析为结构化 JSON（编/分编/章/节/条、施行日期、文号等）。
2. 分块：以"条"为基本检索单元，超长条按款或字符窗口切分（子块共享同一条号，引用仍归同一条）。
3. 建库：将全部 chunk 汇总为同序的三份产物——统一语料 corpus.json、BM25 索引 bm25.pkl、向量索引 faiss.index。
4. 检索：查询走 BM25 与向量召回；启用 LLM 重排时，先向量召回候选，再由大模型逐条打分重排取 top_n。
5. 作答：命中条文按 [序号]《法名》第X条（施行日期）拼装为上下文，交由大模型仅依据上下文作答，输出答案、引用条文清单与免责声明。

## 目录结构

```
RAG-legal/
├── app.py                      # Streamlit Web 界面
├── cli.py                      # 命令行入口（ask / batch）
├── requirements.txt
├── data/                       # 法规源文件（29 docx + 1 PDF 汇编）
├── databases/                  # 构建产物（已在 .gitignore 中忽略，需本地生成）
│   ├── parsed_laws/            # 解析后的结构化 JSON
│   ├── chunked_laws/           # 分块后的 chunk JSON
│   ├── laws_metadata.json      # 全库法规元数据
│   └── index/                  # corpus.json / bm25.pkl / faiss.index
└── src/
    ├── legal_parser.py         # 解析（docx + PDF 汇编）
    ├── text_splitter.py        # 按条分块
    ├── ingestion.py            # 全库建库（向量 + BM25）
    ├── retrieval.py            # 检索（BM25 / 向量 / 混合重排）
    ├── questions_processing.py # 问答管线（条级引用）
    ├── prompts.py              # 提示词
    ├── api_requests.py         # 大模型 API 封装
    ├── reranking.py            # LLM 重排
    └── api_request_parallel_processor.py
```

## 环境准备

建议 Python 3.10 及以上。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置 API Key

在项目根目录新建 .env 文件，填入 DashScope API Key（用于通义千问对话与 text-embedding 向量）：

```bash
# .env
DASHSCOPE_API_KEY=你的key
```

也可以不建文件，直接在当前 shell 中导出：`export DASHSCOPE_API_KEY=你的key`。

获取地址：https://dashscope.console.aliyun.com/apiKey

## 构建索引（首次使用或数据更新时）

databases/ 为构建产物，未纳入版本控制。克隆后需从项目根依次执行以下三步重新生成：

```bash
# 1. 解析法规源文件 -> databases/parsed_laws/ + databases/laws_metadata.json
python -m src.legal_parser

# 2. 按条分块 -> databases/chunked_laws/
python -m src.text_splitter

# 3. 建库（BM25 + 向量）-> databases/index/
python -m src.ingestion
```

第 3 步会调用 DashScope 向量接口，需已配置 DASHSCOPE_API_KEY。若只需离线构建 BM25 与语料（不建向量库），可在代码中调用 `build_all(with_vector=False)`。

## 使用

### Web 界面

```bash
streamlit run app.py
```

默认仅本机访问（http://127.0.0.1:8501）。侧栏可切换 LLM 重排、调节返回条文数（top_n）、按法名过滤；主区输入问题即可查看答案、引用条文（含施行日期）、分步推理与免责声明。

### 命令行

单题问答：

```bash
python cli.py ask "劳动合同的试用期最长可以约定多久？"
# 可选：--law-filter 劳动合同法  --top-n 6  --no-rerank  --model qwen-turbo
```

批量问答（输入为 JSON 列表，每项含 question，可选 law_filter）：

```bash
python cli.py batch questions.json -o questions_with_answers.json
# 可选：--top-n  --no-rerank  --parallel 5  --model qwen-turbo
```

## 说明

- 本系统输出内容由大模型基于检索到的法条生成，仅供参考，不构成法律意见；具体问题请咨询专业律师，并以官方发布的现行有效法律法规为准。
- 检索与作答默认使用 DashScope（通义千问 + text-embedding-v4），需自备 API Key。
