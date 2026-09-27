# -*- coding: utf-8 -*-
"""法律法规检索模块（P4）

在 P3 建成的"全库统一"索引上做检索，去掉 RAG-cy"按公司名先选索引"的路由：
读取 databases/index/ 下三份同序产物（corpus.json / bm25.pkl / faiss.index），
命中下标即回填该 chunk 的《法名》第X条、施行日期等引用信息。

- BM25Retriever：关键词检索，查询分词与建库端一致（ingestion.tokenize_zh，jieba 精确模式）。
- VectorRetriever：向量检索，查询端与建库端对齐——同模型 text-embedding-v4、
  text_type="query"、faiss.normalize_L2 后与归一化文档向量做内积（即余弦）。
- HybridRetriever：向量召回 + LLM 重排（沿用 RAG-cy 结构，重排器分数解析已在 reranking.py 修复）。

三个检索器均支持可选按法过滤 law_filter（对 law_name 做子串匹配）；返回结果为对应 corpus
条目并附 distance 字段（向量为余弦相似度、BM25 为 BM25 分数，均越大越相关）。
"""
from __future__ import annotations

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt

# 兼容以脚本或模块方式运行：确保 from src.xxx 的绝对导入可用（与 api_requests.py 一致）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingestion import tokenize_zh, EMBED_MODEL, EMBED_DIM, INDEX_DIR
from src.reranking import LLMReranker


def _load_corpus(index_dir: str) -> List[Dict]:
    """读取全库统一语料（与 bm25/faiss 同序，命中下标即回填引用信息）。"""
    return json.loads((Path(index_dir) / "corpus.json").read_text(encoding="utf-8"))


def _match_law(item: Dict, law_filter: Optional[str]) -> bool:
    """按法过滤：law_filter 为空则不过滤，否则对 law_name 做子串匹配。"""
    if not law_filter:
        return True
    return law_filter in (item.get("law_name") or "")


class BM25Retriever:
    """全库 BM25 关键词检索（jieba 分词，与建库端 ingestion.tokenize_zh 一致）。"""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.corpus = _load_corpus(index_dir)
        with open(Path(index_dir) / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

    def retrieve(self, query: str, top_n: int = 6,
                 law_filter: Optional[str] = None) -> List[Dict]:
        """返回 BM25 分数最高的 top_n 条（可选按法过滤），每条含 distance=BM25 分数。"""
        scores = self.bm25.get_scores(tokenize_zh(query))
        candidates = [i for i in range(len(scores)) if _match_law(self.corpus[i], law_filter)]
        top = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_n]
        return [{**self.corpus[i], "distance": round(float(scores[i]), 4)} for i in top]


class VectorRetriever:
    """全库向量检索（DashScope text-embedding-v4，归一化后内积=余弦）。"""

    def __init__(self, index_dir: str = INDEX_DIR):
        load_dotenv()
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY（可复制 .env.example 为 .env 后填入），无法做向量检索。"
            )
        dashscope.api_key = key
        self.corpus = _load_corpus(index_dir)
        self.index = faiss.read_index(str(Path(index_dir) / "faiss.index"))

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
    def _embed_query(self, text: str) -> List[float]:
        """查询端嵌入：固定 text_type="query"，与建库端 document 非对称但同模型同维。"""
        resp = TextEmbedding.call(
            model=EMBED_MODEL, input=[text],
            dimension=EMBED_DIM, text_type="query",
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope 查询嵌入失败: code={resp.status_code} msg={resp.message}"
            )
        return resp.output["embeddings"][0]["embedding"]

    def retrieve(self, query: str, top_n: int = 6,
                 law_filter: Optional[str] = None) -> List[Dict]:
        """返回余弦相似度最高的 top_n 条（可选按法过滤），每条含 distance=余弦相似度。"""
        arr = np.asarray([self._embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(arr)                      # 与建库端一致，内积即余弦
        # 按法过滤需在全库范围内取，故检索全部再过滤；否则只取 top_n
        k = self.index.ntotal if law_filter else min(top_n, self.index.ntotal)
        scores, idxs = self.index.search(arr, k)
        results: List[Dict] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            item = self.corpus[i]
            if not _match_law(item, law_filter):
                continue
            results.append({**item, "distance": round(float(score), 4)})
            if len(results) >= top_n:
                break
        return results


class HybridRetriever:
    """向量召回 + LLM 重排（沿用 RAG-cy 结构；重排器分数解析已在 reranking.py 修复）。"""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.vector_retriever = VectorRetriever(index_dir)
        self.reranker = LLMReranker()

    def retrieve(self, query: str, llm_reranking_sample_size: int = 20,
                 documents_batch_size: int = 4, top_n: int = 6,
                 llm_weight: float = 0.7, law_filter: Optional[str] = None) -> List[Dict]:
        """先向量召回 llm_reranking_sample_size 条候选，再 LLM 重排取前 top_n。"""
        candidates = self.vector_retriever.retrieve(
            query, top_n=llm_reranking_sample_size, law_filter=law_filter
        )
        reranked = self.reranker.rerank_documents(
            query=query,
            documents=candidates,
            documents_batch_size=documents_batch_size,
            llm_weight=llm_weight,
        )
        return reranked[:top_n]
