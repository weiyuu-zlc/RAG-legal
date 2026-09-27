from pydantic import BaseModel, Field
from typing import Literal, List, Union
import inspect
import re


def build_system_prompt(instruction: str="", example: str="", pydantic_schema: str="") -> str:
    delimiter = "\n\n---\n\n"
    schema = f"你的回答必须是JSON，并严格遵循如下Schema，字段顺序需保持一致：\n```\n{pydantic_schema}\n```"
    if example:
        example = delimiter + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt

class AnswerWithRAGContextSharedPrompt:
    instruction = """
你是一个法律法规智能问答系统。
你的任务是仅基于检索到的法律法规条文（上下文）回答用户的法律问题。

上下文中每个条文块以 [序号] 《法名》第X条（施行日期）开头，正文在三引号内。
在给出最终答案前，请详细分步思考：
- 只依据上下文中的条文作答，不得编造条文，也不得引入上下文之外的法律知识。
- 答案中引用具体规定时，需指明依据的《法名》第X条。
- 若上下文中没有可回答该问题的条文，请如实说明未检索到相关规定，不要臆测。
"""

    user_prompt = """
以下是上下文:
\"\"\"
{context}
\"\"\"

---

以下是问题：
"{question}"
"""

class AnswerSchemaFixPrompt:
    system_prompt = """
你是一个JSON格式化助手。
你的任务是将大模型输出的原始内容格式化为合法的JSON对象。
你的回答必须以"{"开头，以"}"结尾。
你的回答只能包含JSON字符串，不要有任何前言、注释或三引号。
"""

    user_prompt = """
下面是定义JSON对象Schema和示例的系统提示词:
\"\"\"
{system_prompt}
\"\"\"

---

下面是需要你格式化为合法JSON的LLM原始输出：
\"\"\"
{response}
\"\"\"
"""




class RerankingPrompt:
    system_prompt_rerank_single_block = """
你是一个RAG检索重排专家。
你将收到一个查询和一个检索到的文本块，请根据其与查询的相关性进行评分。

评分说明：
1. 推理：分析文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

    system_prompt_rerank_multiple_blocks = """
你是一个RAG检索重排专家。
你将收到一个查询和若干检索到的文本块，请分别对每个块进行相关性评分。

评分说明：
1. 推理：分析每个文本块与查询的关系，简要说明理由。
2. 相关性分数（0-1，步长0.1）：
   0 = 完全无关
   0.1 = 极弱相关
   0.2 = 很弱相关
   0.3 = 略有相关
   0.4 = 部分相关
   0.5 = 一般相关
   0.6 = 较为相关
   0.7 = 相关
   0.8 = 很相关
   0.9 = 高度相关
   1 = 完全匹配
3. 只基于内容客观评价，不做假设。
"""

class RetrievalRankingSingleBlock(BaseModel):
    """对检索到的单个文本块与查询的相关性进行评分。"""
    reasoning: str = Field(description="分析该文本块，指出其关键信息及与查询的关系")
    relevance_score: float = Field(description="相关性分数，取值范围0到1，0表示完全无关，1表示完全相关")

class RetrievalRankingMultipleBlocks(BaseModel):
    """对检索到的多个文本块与查询的相关性进行评分。"""
    block_rankings: List[RetrievalRankingSingleBlock] = Field(
        description="文本块及其相关性分数的列表。"
    )

class AnswerWithRAGContextStringPrompt:
    instruction = AnswerWithRAGContextSharedPrompt.instruction
    user_prompt = AnswerWithRAGContextSharedPrompt.user_prompt

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(description="""
详细分步推理过程，至少5步，150字以上。请结合上下文信息，逐步分析并归纳答案。
""")
        reasoning_summary: str = Field(description="简要总结分步推理过程，约50字。")
        relevant_articles: List[int] = Field(description="""
仅包含直接用于回答问题的条文块序号（即上下文中每块开头 [序号] 的数字）。只包括：
- 直接包含答案或明确规定的条文块
- 强有力支持答案的关键条文块
不要包含仅与答案弱相关或间接相关的条文块。
列表中至少应有一个条文块序号。
""")
        final_answer: str = Field(description="""
最终答案为一段完整、连贯的中文文本，需仅依据上下文中的条文作答，并在引用具体规定时指明依据的《法名》第X条。
如上下文无相关条文，请如实说明未检索到相关规定，不要臆测。
""")

    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)

    example = r'''
示例：
问题：
"劳动合同的试用期最长可以约定多久？"

答案：
```
{
  "step_by_step_analysis": "1. 问题询问劳动合同试用期的最长期限。\n2. 上下文[1]为《中华人民共和国劳动合同法》第十九条，规定试用期与劳动合同期限挂钩：合同期限三个月以上不满一年的，试用期不得超过一个月；一年以上不满三年的，不得超过二个月；三年以上固定期限和无固定期限的，不得超过六个月。\n3. 据此，试用期最长不得超过六个月，且仅在三年以上或无固定期限劳动合同中方可约定。\n4. 上下文[2]为《中华人民共和国劳动法》第二十一条，同样规定试用期最长不得超过六个月。\n5. 综合两条规定，试用期上限为六个月。",
  "reasoning_summary": "依据劳动合同法第十九条与劳动法第二十一条，试用期最长不得超过六个月。",
  "relevant_articles": [1, 2],
  "final_answer": "劳动合同的试用期最长不得超过六个月。根据《中华人民共和国劳动合同法》第十九条，试用期长短与劳动合同期限挂钩：劳动合同期限三个月以上不满一年的，试用期不得超过一个月；一年以上不满三年的，不得超过二个月；三年以上固定期限和无固定期限劳动合同的，不得超过六个月；《中华人民共和国劳动法》第二十一条同样规定试用期最长不得超过六个月。"
}
```
'''

    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)
