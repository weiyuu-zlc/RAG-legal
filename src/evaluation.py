"""P7 评测模块：检索质量评测 + 答案质量评测(LLM-judge)。

评测集由 LLM 从 corpus 条文自动生成，每题标注命中的《法名》第X条。
复用 src/retrieval.py 的三种检索器、src/questions_processing.py 的问答管线、
src/reranking.py 的 _parse_llm_json；LLM 调用走 DashScope qwen。
入口：python -m src.evaluation {gen|retrieval|answer|all}
"""
import os
import json
import time
import random
import argparse
from pathlib import Path

import dashscope
from dotenv import load_dotenv

from src.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from src.questions_processing import QuestionsProcessor
from src.reranking import _parse_llm_json

INDEX_DIR = "databases/index"
EVAL_DIR = "databases/eval"
CORPUS_PATH = str(Path(INDEX_DIR) / "corpus.json")
DEFAULT_GEN_MODEL = "qwen-plus"
DEFAULT_JUDGE_MODEL = "qwen-plus"


def _load_corpus():
    """读取建库产物 corpus.json（检索与金标条文的共同来源）。"""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _dashscope_chat(system_content, human_content, model, temperature=0.1):
    """调用 DashScope qwen 并解析出 JSON；失败即抛错，不静默降级。"""
    load_dotenv()
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法调用 LLM 评测。")
    rsp = dashscope.Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content},
        ],
        temperature=temperature,
        result_format="message",
    )
    if rsp.status_code != 200:
        raise RuntimeError(f"DashScope 调用失败: code={rsp.status_code} msg={rsp.message}")
    content = rsp.output.choices[0].message.content
    return _parse_llm_json(content)


def _hit(chunk, gold_law, gold_article):
    """命中判定：法名一致且条号一致（检索命中与引用命中共用）。"""
    return (chunk.get("law_name") == gold_law
            and str(chunk.get("article_no") or "") == str(gold_article or ""))


_GEN_SYSTEM = (
    "你是使用法律法规检索系统的普通用户。给你一条法律条文，请提出一个你在现实中"
    "真会问的中文问题，该问题的答案正好落在这条条文里。要求：问题自然口语、只问一个点、"
    "不要出现法律名称和条号、不要照抄原文。只输出 JSON：{\"question\": \"...\"}。"
)


def generate_eval_set(n=20, out_path=None, model=DEFAULT_GEN_MODEL, seed=42):
    """从 corpus 条文抽样，用 LLM 生成问题并标注命中条（法名+条号）。"""
    out_path = out_path or str(Path(EVAL_DIR) / "eval_questions.json")
    corpus = _load_corpus()
    # 只从“完整单块条文”抽样（article 型、有条号、未被切成多块），问题更干净
    pool = [c for c in corpus
            if c.get("type") == "article"
            and c.get("article_no")
            and (c.get("n_parts") or 1) == 1
            and len(c.get("text", "")) >= 30]
    random.seed(seed)
    random.shuffle(pool)
    # 优先按法名去重，尽量覆盖更多不同法规；不足 n 再从剩余池补足
    seen_laws, picked = set(), []
    for c in pool:
        if c["law_name"] in seen_laws:
            continue
        seen_laws.add(c["law_name"])
        picked.append(c)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for c in pool:
            if c not in picked:
                picked.append(c)
                if len(picked) >= n:
                    break

    items, failed = [], 0
    for c in picked:
        try:
            data = _dashscope_chat(_GEN_SYSTEM, c["text"], model=model, temperature=0.7)
            question = (data.get("question") or "").strip()
            if not question:
                raise ValueError("LLM 未返回 question 字段")
        except Exception as e:
            failed += 1
            print(f"[生成失败] chunk={c.get('id')} : {e}")
            continue
        items.append({
            "id": len(items) + 1,
            "question": question,
            "gold_law_name": c["law_name"],
            "gold_article_no": c.get("article_no"),
            "gold_article_num": c.get("article_num"),
            "gold_text": c.get("text", ""),
            "source_law_level": c.get("law_level"),
        })
        time.sleep(0.3)

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"已生成评测集 {len(items)} 题（失败 {failed}），写入 {out_path}")
    print("提示：LLM 自动生成，建议人工抽查后再评测。")
    return items


def _load_eval_set(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_hit_rank(results, gold_law, gold_article):
    """返回首个命中的 1-based 排名，未命中返回 None。"""
    for rank, c in enumerate(results, start=1):
        if _hit(c, gold_law, gold_article):
            return rank
    return None


def _print_retrieval_table(report):
    k = report["k"]
    print(f"\n=== 检索质量评测 (n={report['n_questions']}, k={k}) ===")
    print(f"{'检索器':<8}{'Hit@1':>8}{'Hit@3':>8}{('Hit@' + str(k)):>8}{'MRR':>8}")
    for name, m in report["per_retriever"].items():
        print(f"{name:<8}{m['hit@1']:>8}{m['hit@3']:>8}{m['hit@k']:>8}{m['mrr']:>8}")


def evaluate_retrieval(eval_path=None, k=5, out_path=None, include_hybrid=True):
    """对 BM25/向量/Hybrid 三器算 Hit@1/Hit@3/Hit@k、MRR、召回率。"""
    eval_path = eval_path or str(Path(EVAL_DIR) / "eval_questions.json")
    out_path = out_path or str(Path(EVAL_DIR) / "retrieval_report.json")
    questions = _load_eval_set(eval_path)

    retrievers = {"bm25": BM25Retriever(INDEX_DIR), "vector": VectorRetriever(INDEX_DIR)}
    if include_hybrid:
        retrievers["hybrid"] = HybridRetriever(INDEX_DIR)

    report = {"k": k, "n_questions": len(questions), "per_retriever": {}, "details": []}
    for name, r in retrievers.items():
        ranks = []
        for q in questions:
            res = r.retrieve(q["question"], top_n=k)
            rank = _first_hit_rank(res, q["gold_law_name"], q["gold_article_no"])
            ranks.append(rank)
            report["details"].append({
                "retriever": name, "qid": q["id"], "question": q["question"],
                "gold": f"《{q['gold_law_name']}》{q.get('gold_article_no')}",
                "hit_rank": rank,
            })
        n = len(ranks) or 1
        hit_at = lambda kk: round(sum(1 for x in ranks if x is not None and x <= kk) / n, 4)
        report["per_retriever"][name] = {
            "hit@1": hit_at(1),
            "hit@3": hit_at(min(3, k)),
            "hit@k": hit_at(k),
            "mrr": round(sum(1.0 / x for x in ranks if x) / n, 4),
            "recall": hit_at(k),
        }

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _print_retrieval_table(report)
    print(f"检索评测报告写入 {out_path}")
    return report


_JUDGE_SYSTEM = (
    "你是严格的法律问答评审。根据【标准条文】判断【模型答案】是否正确回答了【问题】。"
    "只依据标准条文，不臆断。只输出 JSON："
    "{\"correctness\": 0到1的小数, \"reason\": \"简短理由\"}。"
    "correctness=1 表示答案与标准条文一致且正确，0 表示错误或答非所问。"
)


def _judge_answer(question, gold_text, final_answer, model):
    """LLM-judge：对照标准条文给 final_answer 打 0~1 正确性分。"""
    human = f"【问题】{question}\n\n【标准条文】{gold_text}\n\n【模型答案】{final_answer}"
    data = _dashscope_chat(_JUDGE_SYSTEM, human, model=model, temperature=0.0)
    score = float(data.get("correctness"))
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"correctness 越界: {score}")
    return score, data.get("reason", "")


def evaluate_answers(eval_path=None, out_path=None, llm_reranking=False,
                     top_n=6, judge_model=DEFAULT_JUDGE_MODEL):
    """跑问答管线得到 final_answer，再用 LLM-judge 评正确性；引用命中用程序判定。"""
    eval_path = eval_path or str(Path(EVAL_DIR) / "eval_questions.json")
    out_path = out_path or str(Path(EVAL_DIR) / "answer_report.json")
    questions = _load_eval_set(eval_path)

    processor = QuestionsProcessor(llm_reranking=llm_reranking, top_n_retrieval=top_n)

    details, scores, cite_hits, failed = [], [], 0, 0
    for q in questions:
        try:
            ans = processor.get_answer_for_question(q["question"])
            final_answer = ans.get("final_answer", "")
            refs = ans.get("references", [])
            cite_hit = any(_hit(rf, q["gold_law_name"], q["gold_article_no"]) for rf in refs)
            gold_text = q.get("gold_text", "")
            score, reason = _judge_answer(q["question"], gold_text, final_answer, judge_model)
        except Exception as e:
            failed += 1
            print(f"[答案评测失败] qid={q['id']} : {e}")
            details.append({"qid": q["id"], "status": "error", "error": str(e)})
            continue
        scores.append(score)
        cite_hits += 1 if cite_hit else 0
        details.append({
            "qid": q["id"], "status": "ok", "question": q["question"],
            "gold": f"《{q['gold_law_name']}》{q.get('gold_article_no')}",
            "final_answer": final_answer,
            "citations": [rf.get("citation") for rf in refs],
            "citation_hit": cite_hit, "correctness": score, "judge_reason": reason,
        })
        time.sleep(0.3)

    n_ok = len(scores)
    report = {
        "n_questions": len(questions), "n_ok": n_ok, "n_failed": failed,
        "llm_reranking": llm_reranking, "top_n": top_n, "judge_model": judge_model,
        "avg_correctness": round(sum(scores) / n_ok, 4) if n_ok else None,
        "citation_hit_rate": round(cite_hits / n_ok, 4) if n_ok else None,
        "details": details,
    }
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n=== 答案质量评测 (n_ok={n_ok}/{len(questions)}, 失败 {failed}) ===")
    print(f"平均正确性(LLM-judge): {report['avg_correctness']}")
    print(f"引用命中率(程序判定): {report['citation_hit_rate']}")
    print(f"答案评测报告写入 {out_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="P7 评测：检索质量 + 答案质量(LLM-judge)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("gen", help="LLM 自动生成评测集")
    pg.add_argument("-n", type=int, default=20)
    pg.add_argument("--model", default=DEFAULT_GEN_MODEL)
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("-o", "--out", default=None)

    pr = sub.add_parser("retrieval", help="检索质量评测")
    pr.add_argument("-k", type=int, default=5)
    pr.add_argument("--no-hybrid", action="store_true")
    pr.add_argument("--eval", default=None)
    pr.add_argument("-o", "--out", default=None)

    pa = sub.add_parser("answer", help="答案质量评测(LLM-judge)")
    pa.add_argument("--rerank", action="store_true")
    pa.add_argument("--top-n", type=int, default=6)
    pa.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    pa.add_argument("--eval", default=None)
    pa.add_argument("-o", "--out", default=None)

    pall = sub.add_parser("all", help="生成评测集 + 检索评测 + 答案评测")
    pall.add_argument("-n", type=int, default=20)
    pall.add_argument("-k", type=int, default=5)
    pall.add_argument("--rerank", action="store_true")
    pall.add_argument("--no-hybrid", action="store_true")

    args = parser.parse_args()
    if args.cmd == "gen":
        generate_eval_set(n=args.n, out_path=args.out, model=args.model, seed=args.seed)
    elif args.cmd == "retrieval":
        evaluate_retrieval(eval_path=args.eval, k=args.k, out_path=args.out,
                           include_hybrid=not args.no_hybrid)
    elif args.cmd == "answer":
        evaluate_answers(eval_path=args.eval, out_path=args.out,
                         llm_reranking=args.rerank, top_n=args.top_n,
                         judge_model=args.judge_model)
    elif args.cmd == "all":
        generate_eval_set(n=args.n)
        evaluate_retrieval(k=args.k, include_hybrid=not args.no_hybrid)
        evaluate_answers(llm_reranking=args.rerank)


if __name__ == "__main__":
    main()





