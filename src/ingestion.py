# -*- coding: utf-8 -*-
"""法律法规建库模块（P3）

将 databases/chunked_laws 下逐部法规的 chunk 汇总为"全库统一"检索索引：
- 去掉 RAG-cy"一份文档=一公司=一个 sha1 索引"的路由，改为整库一个 FAISS + 一个 BM25，
  检索时靠 chunk 自带的 law_name/article_no 等 metadata 过滤，而非先选索引。
- 三份产物同序（第 i 条语料 <-> FAISS 第 i 行 <-> BM25 第 i 篇），检索命中下标即回该条：
  databases/index/corpus.json  统一语料（chunk 字段 + 施行日期/级别/文号，供引用直接取用）
  databases/index/bm25.pkl     全库 BM25（jieba 中文分词）
  databases/index/faiss.index  全库向量库（DashScope text-embedding-v4，归一化后内积=余弦）

向量与 BM25 分词的"查询端"须与本模块一致：检索用 text_type="query" 且同样 normalize_L2，
BM25 查询用本模块的 tokenize_zh，两端分词器一致才可比。
"""
from __future__ import annotations

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
import jieba
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt
import dashscope
from dashscope import TextEmbedding

CHUNKED_DIR = "databases/chunked_laws"
INDEX_DIR = "databases/index"
EMBED_MODEL = "text-embedding-v4"
EMBED_DIM = 1024                 # v4 默认 1024 维
EMBED_BATCH = 10                 # v4 单次请求最多 10 条
# 断点续跑缓存：逐批嵌入随算随追加落盘，中途失败重跑可跳过已嵌入部分（构建成功后清理）
EMBED_CACHE = ".embed_cache.f32"             # 原始（未归一化）向量，按 float32 行连续存储
EMBED_CACHE_META = ".embed_cache.meta.json"  # 语料指纹与目标条数，用于校验缓存是否可复用
# 从每部法规 metainfo 补进每条 chunk 的字段：引用需"《法名》第X条 + 施行日期"，级别/文号供过滤
_ENRICH_KEYS = ("effective_date", "law_level", "doc_number", "source_file")


def tokenize_zh(text: str) -> List[str]:
    """中文分词（jieba 精确模式），建库与检索共用，保证 BM25 两端分词一致。"""
    return [w for w in jieba.lcut(text) if w.strip()]


def load_corpus(chunked_dir: str = CHUNKED_DIR) -> List[Dict]:
    """汇总 chunked_laws 下所有 chunk 为一个有序语料列表。

    列表顺序即建库顺序：既是 FAISS 行号也是 BM25 文档号，检索回填引用直接按下标取。
    每条在原 chunk 字段上补 metainfo 的施行日期/级别/文号/源文件，使语料自足、检索无需再回查。
    """
    corpus: List[Dict] = []
    for fp in sorted(Path(chunked_dir).glob("*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        meta = data["metainfo"]
        enrich = {k: meta.get(k) for k in _ENRICH_KEYS}
        for c in data["chunks"]:
            item = dict(c)
            item.update(enrich)
            corpus.append(item)
    return corpus


class BM25Ingestor:
    """全库统一 BM25 索引（jieba 中文分词，替换 RAG-cy 的空格分词）。"""

    def build(self, corpus: List[Dict]) -> BM25Okapi:
        tokenized = [tokenize_zh(c["text"]) for c in corpus]
        return BM25Okapi(tokenized)


class VectorDBIngestor:
    """全库统一 FAISS 向量库（DashScope text-embedding-v4）。

    嵌入支持断点续跑：每批向量随算随以原始（未归一化）形式追加落盘到
    index_dir/.embed_cache.f32，并在 .embed_cache.meta.json 记录语料指纹与目标条数。
    中途失败（额度耗尽、网络中断等）后重跑，按指纹校验一致则跳过已嵌入部分、
    只补算缺失批次，避免重复消耗已付出的嵌入调用；构建成功后清理缓存。
    """

    def __init__(self):
        load_dotenv()
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "未配置 DASHSCOPE_API_KEY（可复制 .env.example 为 .env 后填入），"
                "无法调用 text-embedding-v4 构建向量库。"
            )
        dashscope.api_key = key

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(3))
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """嵌入一批文本；建库端固定 text_type="document"。失败即抛出，不做静默降级。"""
        resp = TextEmbedding.call(
            model=EMBED_MODEL, input=batch,
            dimension=EMBED_DIM, text_type="document",
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"DashScope embedding 调用失败: code={resp.status_code} msg={resp.message}"
            )
        # 返回项含 text_index，按其还原批内顺序，避免与输入错位
        embs = sorted(resp.output["embeddings"], key=lambda e: e["text_index"])
        return [e["embedding"] for e in embs]

    @staticmethod
    def _fingerprint(texts: List[str]) -> str:
        """语料指纹：绑定模型/维度/条数与逐条文本，任一变化即判定缓存失效需重建。"""
        h = hashlib.md5()
        h.update(f"{EMBED_MODEL}:{EMBED_DIM}:{len(texts)}".encode("utf-8"))
        for t in texts:
            h.update(t.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    @staticmethod
    def _resume_point(cache: Path, meta_path: Path, fingerprint: str) -> int:
        """校验嵌入缓存并返回可续跑的已完成条数（指纹不符或缓存不全则清理，从 0 开始）。"""
        if not (cache.exists() and meta_path.exists()):
            for p in (cache, meta_path):
                if p.exists():
                    p.unlink()                  # 半份缓存（仅存其一）不可信，清理
            return 0
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            cache.unlink(); meta_path.unlink()   # 语料或模型已变，旧缓存作废
            return 0
        row_bytes = EMBED_DIM * 4                 # 每行 = EMBED_DIM 个 float32
        n_rows = cache.stat().st_size // row_bytes
        with open(cache, "r+b") as fh:            # 丢弃进程中途退出可能残留的半行，按整行对齐
            fh.truncate(n_rows * row_bytes)
        if n_rows:
            print(f"  发现嵌入缓存，续跑：已完成 {n_rows} 条，跳过已嵌入部分")
        return n_rows

    def embed_corpus(self, corpus: List[Dict], index_dir: str = INDEX_DIR) -> np.ndarray:
        """按批嵌入整库文本，返回 L2 归一化后的 float32 矩阵（配合内积索引即余弦）。"""
        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)
        cache = out / EMBED_CACHE
        meta_path = out / EMBED_CACHE_META
        texts = [c["text"] for c in corpus]
        fingerprint = self._fingerprint(texts)
        n_done = self._resume_point(cache, meta_path, fingerprint)   # 校验/清理旧缓存
        meta_path.write_text(json.dumps(
            {"model": EMBED_MODEL, "dim": EMBED_DIM,
             "fingerprint": fingerprint, "n_total": len(texts)},
            ensure_ascii=False), encoding="utf-8")
        with open(cache, "ab") as fh:           # 追加模式，续跑接在已完成行之后
            for i in range(n_done, len(texts), EMBED_BATCH):
                batch = self._embed_batch(texts[i:i + EMBED_BATCH])
                np.asarray(batch, dtype=np.float32).tofile(fh)
                fh.flush()                      # 及时落盘，保证中断时已完成进度不丢
                print(f"  已嵌入 {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
        arr = np.fromfile(cache, dtype=np.float32).reshape(-1, EMBED_DIM)
        faiss.normalize_L2(arr)
        return arr

    def build(self, corpus: List[Dict], index_dir: str = INDEX_DIR) -> faiss.Index:
        arr = self.embed_corpus(corpus, index_dir)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        return index


def build_all(chunked_dir: str = CHUNKED_DIR, index_dir: str = INDEX_DIR,
              with_vector: bool = True) -> int:
    """构建全库统一索引：语料 corpus.json + BM25 bm25.pkl（+ 可选 FAISS faiss.index，同序）。

    with_vector=False 时只建 BM25 与语料（无需 API Key），便于离线先行；返回语料条数。
    """
    corpus = load_corpus(chunked_dir)
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bm25 = BM25Ingestor().build(corpus)
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 索引完成：{len(corpus)} 条 -> {out / 'bm25.pkl'}")

    if with_vector:
        index = VectorDBIngestor().build(corpus, index_dir)
        faiss.write_index(index, str(out / "faiss.index"))
        print(f"FAISS 索引完成：{index.ntotal} 向量（维度 {index.d}）-> {out / 'faiss.index'}")
        for p in (out / EMBED_CACHE, out / EMBED_CACHE_META):
            if p.exists():
                p.unlink()               # 构建成功，清理断点续跑缓存
    else:
        print("跳过向量库构建（with_vector=False，未建 faiss.index）")

    return len(corpus)


if __name__ == "__main__":
    build_all()
