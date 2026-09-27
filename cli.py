# -*- coding: utf-8 -*-
"""LegalMind 法律法规智能问答命令行（P6）

基于 click，复用 P5 的 QuestionsProcessor（全库统一索引 + 条级引用）。
- ask：单题问答，打印答案、引用条文（《法名》第X条 + 施行日期）与免责声明。
- batch：批量处理 JSON 问题文件（每项 {"question": ..., 可选 "law_filter": ...}），写出结果 JSON。

需在 .env 中配置 DASHSCOPE_API_KEY。
"""
import click

from src.questions_processing import QuestionsProcessor


def _echo_answer(question, result):
    """将单题结果格式化打印到终端。"""
    click.echo(f"\n问题：{question}\n")
    click.echo(f"答案：{result.get('final_answer') or '未获得答案'}\n")

    references = result.get("references") or []
    click.echo("引用条文：")
    if references:
        for ref in references:
            citation = ref.get("citation") or (ref.get("law_name") or "未知法规")
            effective_date = ref.get("effective_date")
            suffix = f"（施行日期：{effective_date}）" if effective_date else ""
            click.echo(f"  - {citation}{suffix}")
    else:
        click.echo("  无")

    disclaimer = result.get("disclaimer")
    if disclaimer:
        click.echo(f"\n{disclaimer}")


@click.group()
def cli():
    """LegalMind 法律法规智能问答命令行。"""


@cli.command()
@click.argument("question")
@click.option("--law-filter", default=None, help="按《法名》子串过滤检索范围，例如：劳动合同法")
@click.option("--rerank/--no-rerank", default=True, help="是否启用 LLM 重排（默认启用）")
@click.option("--top-n", default=6, show_default=True, help="返回的相关条文数量")
@click.option("--model", default="qwen-turbo", show_default=True, help="作答模型")
def ask(question, law_filter, rerank, top_n, model):
    """就单个法律问题作答。"""
    processor = QuestionsProcessor(
        llm_reranking=rerank, top_n_retrieval=top_n, answering_model=model,
    )
    result = processor.get_answer_for_question(question, law_filter=law_filter)
    _echo_answer(question, result)


@cli.command()
@click.argument("questions_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default="questions_with_answers.json",
              show_default=True, help="结果 JSON 输出路径")
@click.option("--rerank/--no-rerank", default=True, help="是否启用 LLM 重排（默认启用）")
@click.option("--top-n", default=6, show_default=True, help="返回的相关条文数量")
@click.option("--parallel", default=5, show_default=True, help="并行处理线程数")
@click.option("--model", default="qwen-turbo", show_default=True, help="作答模型")
def batch(questions_file, output, rerank, top_n, parallel, model):
    """批量处理问题文件（JSON 列表）。"""
    processor = QuestionsProcessor(
        questions_file_path=questions_file,
        llm_reranking=rerank, top_n_retrieval=top_n,
        parallel_requests=parallel, answering_model=model,
    )
    stats = processor.process_all_questions(output_path=output).get("statistics", {})
    click.echo(
        f"已写出结果到 {output}（共 {stats.get('total_questions', 0)} 题，"
        f"成功 {stats.get('success_count', 0)}，错误 {stats.get('error_count', 0)}，"
        f"N/A {stats.get('na_count', 0)}）"
    )


if __name__ == "__main__":
    cli()
