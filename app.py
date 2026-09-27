# -*- coding: utf-8 -*-
"""LegalMind 法律法规智能问答 Web 界面（P6）

基于 Streamlit，复用 P5 的 QuestionsProcessor（全库统一索引 + 条级引用）。
单题问答：输入问题，可选按《法名》过滤、调节检索参数，展示答案、引用条文
（《法名》第X条 + 施行日期）、分步推理与免责声明。

本地运行：streamlit run app.py（默认仅本机 localhost 访问）。
需在 .env 中配置 DASHSCOPE_API_KEY。
"""
import streamlit as st

from src.questions_processing import QuestionsProcessor, LEGAL_DISCLAIMER


st.set_page_config(page_title="LegalMind 法律法规智能问答", layout="wide")


@st.cache_resource(show_spinner="正在加载全库索引与检索器……")
def get_processor(llm_reranking, api_provider, answering_model):
    """按检索器类型缓存 QuestionsProcessor（加载 corpus/faiss 较重，避免每次查询重建）。

    llm_reranking 决定使用 HybridRetriever 还是 VectorRetriever，在 __init__ 内初始化，
    故作为缓存键；top_n_retrieval 等每次查询参数在调用前覆盖实例属性即可。
    """
    return QuestionsProcessor(
        llm_reranking=llm_reranking,
        api_provider=api_provider,
        answering_model=answering_model,
    )


def render_result(result):
    """展示单题作答结果：答案、引用条文、分步推理、免责声明。"""
    st.subheader("答案")
    st.write(result.get("final_answer") or "未获得答案")

    references = result.get("references") or []
    st.subheader("引用条文")
    if references:
        for ref in references:
            citation = ref.get("citation") or (ref.get("law_name") or "未知法规")
            effective_date = ref.get("effective_date")
            suffix = f"（施行日期：{effective_date}）" if effective_date else ""
            st.markdown(f"- {citation}{suffix}")
    else:
        st.write("无")

    analysis = result.get("step_by_step_analysis")
    summary = result.get("reasoning_summary")
    if analysis or summary:
        with st.expander("推理过程"):
            if summary:
                st.markdown("**推理摘要**")
                st.write(summary)
            if analysis:
                st.markdown("**分步推理**")
                st.write(analysis)

    disclaimer = result.get("disclaimer") or LEGAL_DISCLAIMER
    st.caption(disclaimer)


# 侧栏：检索参数
st.sidebar.header("检索参数")
llm_reranking = st.sidebar.checkbox(
    "启用 LLM 重排", value=True,
    help="开启后先向量召回再由大模型逐条重排，检索更准但更慢",
)
top_n = st.sidebar.slider("返回相关条文数（top_n）", min_value=1, max_value=12, value=6)
law_filter = st.sidebar.text_input(
    "按《法名》过滤（可留空）",
    help="填写法规名称子串，仅在该法规范围内检索，例如：劳动合同法",
)

# 主区：单题问答
st.title("LegalMind 法律法规智能问答")
st.write("仅依据检索到的法律法规条文作答，并标注《法名》第X条与施行日期。")

question = st.text_area(
    "请输入法律问题", height=120,
    placeholder="例如：劳动合同的试用期最长可以约定多久？",
)

if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("请先输入问题。")
    else:
        try:
            processor = get_processor(llm_reranking, "dashscope", "qwen-turbo")
            processor.top_n_retrieval = top_n  # 每次查询覆盖 top_n（实例属性）
            with st.spinner("检索并作答中……"):
                result = processor.get_answer_for_question(
                    question.strip(), law_filter=law_filter.strip() or None
                )
            render_result(result)
        except Exception as err:
            st.error(f"处理失败：{type(err).__name__}: {err}")
