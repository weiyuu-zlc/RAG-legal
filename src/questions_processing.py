# -*- coding: utf-8 -*-
"""法律法规问答处理模块（P5）

在 P4 全库统一检索之上做"条级引用"的问答，沿用 RAG-cy 的批处理管线结构
（批量并行、进度保存、统计），但改造为法律法规场景：
- 去掉公司名抽取、多公司比较、比赛提交(sha1/页码)等年报专用逻辑；
- 检索改用 P4 的 HybridRetriever/VectorRetriever（全库索引 + 可选按法过滤）；
- 引用从"页码"改为"条文"：_format_retrieval_results 按 [序号]《法名》第X条 标注上下文，
  _validate_article_references（改自 RAG-cy 的 _validate_page_references）校验 LLM 引用的
  条文块序号，再由序号回填《法名》第X条 + 施行日期作为 references；
- 答案统一用 string schema（法律解释文本），并为每个回答固定附带免责声明。
"""
import json
import threading
import concurrent.futures
import sys
from pathlib import Path
from typing import Union, List, Optional

from tqdm import tqdm

# 兼容以脚本或模块方式运行：确保 from src.xxx 的绝对导入可用（与 api_requests.py 一致）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingestion import INDEX_DIR
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor

# 免责声明：每个法律问答回答固定附带，避免被当作正式法律意见
LEGAL_DISCLAIMER = (
    "本回答依据检索到的法律法规条文由模型自动生成，仅供参考，不构成正式法律意见；"
    "请以官方发布的现行有效法律法规原文为准，具体情形建议咨询专业律师。"
)


class QuestionsProcessor:
    """法律问答批处理器：检索 -> 拼上下文 -> LLM 作答 -> 校验条文引用 -> 回填引用与免责声明。"""

    def __init__(
        self,
        index_dir: Union[str, Path] = INDEX_DIR,
        questions_file_path: Optional[Union[str, Path]] = None,
        llm_reranking: bool = True,
        llm_reranking_sample_size: int = 20,
        top_n_retrieval: int = 6,
        parallel_requests: int = 5,
        api_provider: str = "dashscope",
        answering_model: str = "qwen-turbo",
    ):
        # 初始化：加载问题、初始化检索器（一次复用）与 LLM 处理器
        self.questions = self._load_questions(questions_file_path)
        self.index_dir = index_dir
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.answering_model = answering_model
        self.api_processor = APIProcessor(provider=api_provider)
        # 全库统一索引，检索器初始化一次即可跨问题复用（加载 corpus/faiss 较重，避免每题重载）
        self.retriever = (HybridRetriever(index_dir) if llm_reranking
                          else VectorRetriever(index_dir))
        self.answer_details = []
        self._lock = threading.Lock()
        self.response_data = None

    def _load_questions(self, questions_file_path) -> List[dict]:
        # 加载问题文件，返回问题列表（每项含 question，可选 law_filter）
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索到的条文格式化为 RAG 上下文；每块以 [序号]《法名》第X条（施行日期）开头。"""
        if not retrieval_results:
            return ""
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            law_name = result.get("law_name") or "未知法规"
            article_no = result.get("article_no") or ""
            effective_date = result.get("effective_date")
            header = f"[{i}] 《{law_name}》{article_no}".rstrip()
            if effective_date:
                header += f"（施行日期：{effective_date}）"
            context_parts.append(f'{header}\n"""\n{result["text"]}\n"""')
        return "\n\n---\n\n".join(context_parts)
    def _validate_article_references(self, claimed_indices, retrieval_results,
                                     min_refs: int = 1, max_refs: int = 8) -> List[int]:
        """校验 LLM 引用的条文块序号（1-based，对应上下文块）：去幻觉、补足到 min、截断到 max。

        改自 RAG-cy 的 _validate_page_references：键由"页码"换成"条文块序号"，过滤/补足/截断逻辑一致。
        """
        if claimed_indices is None:
            claimed_indices = []
        available = list(range(1, len(retrieval_results) + 1))
        validated = [i for i in claimed_indices if i in available]

        if len(validated) < len(claimed_indices):
            removed = [i for i in claimed_indices if i not in available]
            print(f"警告：剔除 {len(removed)} 个不存在的条文引用: {removed}")

        # 不足 min 时按检索顺序补足（检索结果本身即相关条文）
        if len(validated) < min_refs:
            for i in available:
                if i not in validated:
                    validated.append(i)
                if len(validated) >= min_refs:
                    break

        # 超过 max 时截断
        if len(validated) > max_refs:
            print(f"引用条文过多，从 {len(validated)} 截断为 {max_refs}")
            validated = validated[:max_refs]

        return validated
    def _build_references(self, indices, retrieval_results) -> List[dict]:
        """由条文块序号回填引用信息：《法名》第X条 + 施行日期。"""
        references = []
        for i in indices:
            result = retrieval_results[i - 1]
            law_name = result.get("law_name") or "未知法规"
            article_no = result.get("article_no") or ""
            references.append({
                "law_name": result.get("law_name"),
                "article_no": result.get("article_no"),
                "article_num": result.get("article_num"),
                "effective_date": result.get("effective_date"),
                "citation": f"《{law_name}》{article_no}".rstrip(),
            })
        return references

    def get_answer_for_question(self, question: str, law_filter: Optional[str] = None) -> dict:
        """检索相关条文 -> 拼上下文 -> LLM 作答 -> 校验条文引用 -> 回填引用与免责声明。"""
        if self.llm_reranking:
            retrieval_results = self.retriever.retrieve(
                question,
                llm_reranking_sample_size=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                law_filter=law_filter,
            )
        else:
            retrieval_results = self.retriever.retrieve(
                question, top_n=self.top_n_retrieval, law_filter=law_filter
            )
        if not retrieval_results:
            raise ValueError("未检索到相关法规条文")

        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.api_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema="string",
            model=self.answering_model,
        )
        self.response_data = self.api_processor.response_data

        claimed = answer_dict.get("relevant_articles", [])
        validated = self._validate_article_references(claimed, retrieval_results)
        answer_dict["relevant_articles"] = validated
        answer_dict["references"] = self._build_references(validated, retrieval_results)
        answer_dict["disclaimer"] = LEGAL_DISCLAIMER
        return answer_dict
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情引用ID，并存储详细内容（分步推理/摘要/引用条文/用量）。"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict.get("step_by_step_analysis"),
                "reasoning_summary": answer_dict.get("reasoning_summary"),
                "relevant_articles": answer_dict.get("relevant_articles"),
                "references": answer_dict.get("references"),
                "response_data": self.response_data,
                "self": ref_id,
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果：总数、错误数、N/A 数、成功数。"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if q.get("answer") == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print("\n处理统计：")
            print(f"总问题数: {total_questions}")
            if total_questions:
                print(f"错误: {error_count} ({error_count / total_questions * 100:.1f}%)")
                print(f"N/A: {na_count} ({na_count / total_questions * 100:.1f}%)")
                print(f"成功: {success_count} ({success_count / total_questions * 100:.1f}%)\n")
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count,
        }
    def _process_single_question(self, question_data: dict) -> dict:
        # 处理单个问题项：读取 question/law_filter，作答并组织结果
        question_index = question_data.get("_question_index", 0)
        question_text = question_data.get("question")
        law_filter = question_data.get("law_filter")
        try:
            answer_dict = self.get_answer_for_question(question_text, law_filter=law_filter)
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question": question_text,
                "answer": answer_dict.get("final_answer"),
                "references": answer_dict.get("references", []),
                "disclaimer": answer_dict.get("disclaimer"),
                "answer_details": {"$ref": detail_ref},
            }
        except Exception as err:
            return self._handle_processing_error(question_text, err, question_index)

    def _handle_processing_error(self, question_text: str, err: Exception, question_index: int) -> dict:
        """记录问题处理异常并返回带错误信息的结果字典（不静默吞掉，保留 traceback）。"""
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {"error_traceback": tb, "self": error_ref}
        print(f"处理问题出错: {question_text}")
        print(f"错误类型: {type(err).__name__}")
        print(f"错误信息: {error_message}")
        return {
            "question": question_text,
            "answer": None,
            "references": [],
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }
    def process_questions_list(self, questions_list: List[dict],
                               output_path: Optional[str] = None) -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回结果与统计
        total_questions = len(questions_list)
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(questions_with_index, desc="处理问题中"):
                processed_questions.append(self._process_single_question(question_data))
                if output_path:
                    self._save_progress(processed_questions, output_path)
        else:
            with tqdm(total=total_questions, desc="处理问题中") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i:i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    if output_path:
                        self._save_progress(processed_questions, output_path)
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics,
        }
    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str]):
        # 保存处理进度到 JSON（含问题结果、答案详情、统计）
        if not output_path:
            return
        statistics = self._calculate_statistics(processed_questions)
        result = {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics,
        }
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json') -> dict:
        # 处理初始化时加载的全部问题
        return self.process_questions_list(self.questions, output_path)
