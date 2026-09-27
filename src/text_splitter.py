# -*- coding: utf-8 -*-
"""法律法规文本分块模块（P2）

将 databases/parsed_laws 下的结构化法规 JSON 切分为检索用 chunk：
- 分块单元为"条"：一条=一个 chunk，携带 law_name/article_no 等元数据，便于条级引用
- 超长条再切：docx 条文的款以换行分隔，按款聚合到上限；pdf 条文无款边界，按 token 窗口切
- 散文体（通知/意见/复函）按"节"分块，超长节按 token 窗口切
- 每部法规输出一个 chunk JSON（沿用 parsed_laws 的文件名）；去重篇目已在 P1 排除，无需再判

分块粒度依据实测：条文 token 中位 69、p99 311，仅极少数条/节超 512，故以 512 为切分阈值，
绝大多数条保持整条一块，最利于"《法名》第X条"级别的检索与引用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


class LegalTextSplitter:
    """按条/节分块工具：一条=一块，超长条按款(docx)或字符(pdf)再切，散文按节切。"""

    def __init__(self, max_tokens: int = 512, overlap: int = 50,
                 encoding_name: str = "o200k_base"):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """统计文本 token 数（tiktoken o200k_base，与切分器同一编码，计数一致）。"""
        return len(self._enc.encode(text))

    def _char_split(self, text: str) -> List[str]:
        """无款边界(pdf/散文)或单款超长时，按 token 窗口切分。"""
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.encoding_name,
            chunk_size=self.max_tokens,
            chunk_overlap=self.overlap,
        )
        return splitter.split_text(text)

    def _split_long_text(self, text: str) -> List[str]:
        """超长文本切分：docx 条文含款换行则按款聚合到上限，否则(pdf/散文)按字符窗口切。"""
        paras = [p for p in text.split("\n") if p.strip()]
        if len(paras) <= 1:                       # 无款边界（pdf 条文一行拼接、散文节）
            return self._char_split(text)
        pieces, buf, buf_tok = [], [], 0
        for para in paras:
            pt = self.count_tokens(para)
            if pt > self.max_tokens:              # 单款超长：先冲掉缓冲，再字符切该款
                if buf:
                    pieces.append("\n".join(buf)); buf, buf_tok = [], 0
                pieces.extend(self._char_split(para))
            elif buf_tok + pt > self.max_tokens:  # 加上本款会超限：先出一块，再新起缓冲
                pieces.append("\n".join(buf)); buf, buf_tok = [para], pt
            else:
                buf.append(para); buf_tok += pt
        if buf:
            pieces.append("\n".join(buf))
        return pieces

    def _chunk_texts(self, text: str) -> List[str]:
        """整条/整节不超限则原样一块，超限才切分。"""
        if self.count_tokens(text) <= self.max_tokens:
            return [text]
        return self._split_long_text(text)

    def split_law(self, parsed: Dict) -> Dict:
        """把一部解析后的法规(metainfo + articles/sections)切分为 chunk 列表。

        条文块保留 article_no 与编/分编/章/节层级；散文块保留 section_no。
        超长条/节切出的多个子块共享同一 article_no/section_no，用 part 标序、n_parts 记总数，
        引用时仍归于同一条，检索命中任一子块都能定位到"《法名》第X条"。
        """
        meta = parsed["metainfo"]
        law_name = meta["law_name"]
        chunks, cid = [], 0

        for a in parsed.get("articles", []):
            parts = self._chunk_texts(a["text"])
            for pi, txt in enumerate(parts):
                chunks.append({
                    "id": cid,
                    "type": "article",
                    "law_name": law_name,
                    "article_no": a.get("article_no"),
                    "article_num": a.get("article_num"),
                    "book": a.get("book"),
                    "subbook": a.get("subbook"),
                    "chapter": a.get("chapter"),
                    "section": a.get("section"),
                    "section_no": None,
                    "part": pi,
                    "n_parts": len(parts),
                    "length_tokens": self.count_tokens(txt),
                    "text": txt,
                })
                cid += 1

        for s in parsed.get("sections", []):
            parts = self._chunk_texts(s["text"])
            for pi, txt in enumerate(parts):
                chunks.append({
                    "id": cid,
                    "type": "prose",
                    "law_name": law_name,
                    "article_no": None,
                    "article_num": None,
                    "book": None,
                    "subbook": None,
                    "chapter": None,
                    "section": None,
                    "section_no": s.get("section_no"),
                    "part": pi,
                    "n_parts": len(parts),
                    "length_tokens": self.count_tokens(txt),
                    "text": txt,
                })
                cid += 1

        out_meta = dict(meta)
        out_meta["n_chunks"] = len(chunks)
        return {"metainfo": out_meta, "chunks": chunks}

    def split_all(self, parsed_dir="databases/parsed_laws",
                  output_dir="databases/chunked_laws") -> int:
        """批量切分 parsed_dir 下所有法规 JSON，逐部写出 chunk JSON（沿用原文件名）。

        parsed_laws 已在 P1 排除与 docx 重复的 PDF 篇目，此处直接全量处理即可。
        返回产出的 chunk 总数。
        """
        parsed_dir = Path(parsed_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = sorted(parsed_dir.glob("*.json"))
        total_chunks = 0
        for p in paths:
            parsed = json.loads(p.read_text(encoding="utf-8"))
            result = self.split_law(parsed)
            (output_dir / p.name).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total_chunks += result["metainfo"]["n_chunks"]
        print(f"共分块 {len(paths)} 部法规/篇目，产出 {total_chunks} 个 chunk，写入 {output_dir}")
        return total_chunks


if __name__ == "__main__":
    LegalTextSplitter().split_all()


