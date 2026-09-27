# -*- coding: utf-8 -*-
"""法律法规文档解析模块（P1）

将 data/ 下的法规文档解析为结构化 JSON：
- docx 全部为 Normal 样式，正文层级只能靠正则识别：编 / 分编 / 章 / 节 / 条
- 兼容司法解释体例（用"一、二、…"分节，并带"法释〔〕号"文号）
- 跳过开头目录、规范化全角空格、把"第X条 + 后续款段"聚成一条
- 每部法规输出一个 JSON，并汇总产出全库法规注册表 laws_metadata.json

本文件仅负责解析（P1a：docx）。PDF 汇编切分（P1b）与去重（P1c）后续补充。
"""
from __future__ import annotations

import re
import json
import glob
from pathlib import Path

import docx
import pdfplumber

# ============================ 中文数字转换 ============================
# 用于把"第一百二十三条"里的"一百二十三"转成整数 123，供排序、去重、引用校验使用
_CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def chinese_to_int(s: str):
    """把中文数字字符串转为整数；无法解析返回 None。

    支持：十一 -> 11、二十 -> 20、一百二十三 -> 123、一千二百六十 -> 1260、一百零五 -> 105
    """
    if not s:
        return None
    result = 0  # 已累计的结果
    tmp = 0     # 当前处理的个位数
    for ch in s:
        if ch in _CN_DIGIT:
            tmp = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            # "十X" 开头时十前面没有数字，按 1 处理（如"十一"）
            if tmp == 0:
                tmp = 1
            result += tmp * unit
            tmp = 0
        else:
            return None
    return result + tmp


# ============================ 结构层级正则 ============================
# 说明：docx 正文以行首"第X编/分编/章/节/条"标识层级；先匹配分编再匹配编，避免"第一分编"被当成"编"
_CN = "零〇一二三四五六七八九十百千两"
RE_SUBBOOK = re.compile(rf"^第([{_CN}]+)分编")          # 第一分编（民法典特有，位于编与章之间）
RE_BOOK = re.compile(rf"^第([{_CN}]+)编(?!里|外|内)")   # 第一编
RE_CHAPTER = re.compile(rf"^第([{_CN}]+)章")            # 第一章
RE_SECTION = re.compile(rf"^第([{_CN}]+)节")            # 第一节
RE_ARTICLE = re.compile(rf"^第([{_CN}]+)条")            # 第一条（条号后可能紧跟全角空格或正文）
# 司法解释用"一、二、"作为分节标题（"、"后紧跟简短标题，不含句末标点）
RE_JS_SECTION = re.compile(rf"^([{_CN}]+)、(?![。，、；：])")


def normalize(text: str) -> str:
    """规范化：全角空格转普通空格，去除首尾空白，合并连续空白。"""
    if not text:
        return ""
    text = text.replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _heading_title(text: str) -> str:
    """从"第一章 总 则"这类标题里取纯标题（规范化后已是"第一章 总 则"）。"""
    return normalize(text)


# ============================ 元数据抽取 ============================
# PDF 文本层常在数字/汉字间插入零散空格（宽字距伪影），匹配日期/文号前先去空格
def _despace(text: str) -> str:
    return text.replace(" ", "") if text else ""


def extract_effective_date(intro: str):
    """从开头括号信息里抽取施行日期。

    仅在文本显式写明时抽取，抽不到返回 None（不臆测，避免用"通过日期"冒充"施行日期"）。
    支持："自2023年12月5日起施行"、"自公布之日起施行"。
    """
    if not intro:
        return None
    s = _despace(intro)  # 去掉 PDF 数字间空格，如"自 2021 年 1 月 1 日"
    m = re.search(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起施行", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if "自公布之日起施行" in s or "自发布之日起施行" in s:
        return "自公布之日起施行"
    return None


# 文号：机关简称(+发/函/字)〔年〕序号号；兼容半角括号、国务院令、主席令
RE_DOCNUM = re.compile(
    r"[一-鿿]{2,12}[发函字]?[〔\[]\d{4}[〕\]]\d{1,4}号"
    r"|国务院令第\d+号"
    r"|中华人民共和国主席令第[一-鿿\d]+号"
)
# PDF 封面"…》的通知/复函"后紧跟文号，去空格后机关简称会误粘上"的通知"等文种前缀
_DOCNUM_PREFIX_RE = re.compile(
    r"^的?(?:通知|公告|通告|决定|命令|意见|批复|复函|函|规定|办法|规则|细则|方案|指引)"
)


def extract_doc_number(intro: str):
    """抽取文号，如"法释〔2023〕13号""国务院令第535号""人社部发〔2021〕56号"；抽不到返回 None。"""
    if not intro:
        return None
    m = RE_DOCNUM.search(_despace(intro))
    if m:
        return _DOCNUM_PREFIX_RE.sub("", m.group(0))   # 剥掉封面文种残留前缀
    return None


def detect_law_level(law_name: str, doc_number) -> str:
    """按名称/文号粗判效力级别；可在注册表中人工修正。"""
    if (doc_number and doc_number.startswith("法释")) or "解释" in law_name or "规定" in law_name:
        return "司法解释"
    if law_name.endswith("条例") or law_name.endswith("办法") or (doc_number and "国务院令" in doc_number):
        return "行政法规"
    if law_name.endswith("法") or law_name.endswith("典"):
        return "法律"
    return "其他"


def law_name_from_filename(path: Path) -> str:
    """从文件名去掉尾部日期与扩展名得到法规名；去除书名号方便统一引用。"""
    stem = path.stem
    stem = re.sub(r"_\d{8}$", "", stem)          # 去掉 _20200528 这类日期后缀
    stem = stem.replace("《", "").replace("》", "")
    return stem.strip()


def version_date_from_filename(path: Path):
    """文件名尾部日期为最新通过/修订日期（非施行日期），转 YYYY-MM-DD。"""
    m = re.search(r"_(\d{4})(\d{2})(\d{2})$", path.stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


# 条号（含"第X条之一"这类修正条），group(1)=完整条号标签，group(2)=用于取整的中文数字
RE_ARTICLE_FULL = re.compile(rf"^(第([{_CN}]+)条(?:之[{_CN}]+)?)")


def _extract_articles(paras, law_level, joiner: str = "\n", strict_header: bool = False) -> list:
    """从规范化段落列表中提取条文（编/分编/章/节 状态机）。

    docx 与 PDF 汇编内的结构化法规共用此逻辑；输入为已规范化、已去空段的段落列表，
    返回条文列表（每条含 article_no/article_num/所属层级/text）。

    joiner 为条内续段的连接符：docx 一段=一款，用换行 "\\n"；PDF 每行是物理硬换行
    （中文常在句中断行），用空串 "" 直接拼接以还原完整句子。
    strict_header 为 True 时（PDF），仅当"第X条"后接空格或行尾才视为条文起始，
    避免正文中"第六条、第七条"这类跨条引用因硬换行落到行首被误判为新条。
    """
    articles = []
    cur = {"book": None, "subbook": None, "chapter": None, "section": None}
    current = None  # 正在累计的条

    def flush():
        nonlocal current
        if current is not None:
            articles.append(current)
            current = None

    for p in paras:
        # 层级标题：分编须在编之前判断；章重置节，编重置分编/章/节
        if RE_SUBBOOK.match(p):
            flush(); cur.update(subbook=p, chapter=None, section=None); continue
        if RE_BOOK.match(p):
            flush(); cur.update(book=p, subbook=None, chapter=None, section=None); continue
        if RE_CHAPTER.match(p):
            flush(); cur.update(chapter=p, section=None); continue
        if RE_SECTION.match(p):
            flush(); cur.update(section=p); continue
        # 司法解释以"一、xxx"分节（短标题、无句末标点），仅在非条文累计时识别
        if law_level == "司法解释" and RE_JS_SECTION.match(p) and len(p) <= 20:
            flush(); cur.update(chapter=p, section=None); continue
        m = RE_ARTICLE_FULL.match(p)
        # PDF 严格模式：条号后须为空格或行尾，否则视作跨条引用（并入当前条正文）
        if m and (not strict_header or p[m.end():len(p)][:1] in ("", " ")):
            flush()
            current = {
                "article_no": m.group(1),
                "article_num": chinese_to_int(m.group(2)),
                "book": cur["book"], "subbook": cur["subbook"],
                "chapter": cur["chapter"], "section": cur["section"],
                "text": p,
            }
            continue
        # 其余为条文续段（款/项），并入当前条；条文开始前的标题/引言忽略
        if current is not None:
            current["text"] += joiner + p
    flush()
    return articles


def parse_docx(path) -> dict:
    """解析单部法规 docx，返回结构化 dict（metainfo + articles）。"""
    path = Path(path)
    doc = docx.Document(str(path))
    # 规范化并去掉空段
    paras = [normalize(p.text) for p in doc.paragraphs]
    paras = [p for p in paras if p]

    law_name = law_name_from_filename(path)
    version_date = version_date_from_filename(path)

    # 开头括号信息（施行/通过/文号），取前 15 段里第一个以括号开头的段
    intro = ""
    for p in paras[:15]:
        if p.startswith("（") or p.startswith("("):
            intro = p
            break
    effective_date = extract_effective_date(intro)
    doc_number = extract_doc_number(intro)
    if not doc_number:  # 文号也可能单独成段（如"法释〔2023〕13号"）
        for p in paras[:15]:
            dn = extract_doc_number(p)
            if dn:
                doc_number = dn
                break
    law_level = detect_law_level(law_name, doc_number)

    articles = _extract_articles(paras, law_level)

    # 施行日期常写在最后一条（如民法典"本法自2021年1月1日起施行"）；开头括号未写明时再从末条抽取
    if effective_date is None and articles:
        for a in articles[-2:]:
            ed = extract_effective_date(a["text"])
            if ed:
                effective_date = ed
                break

    return {
        "metainfo": {
            "law_name": law_name,
            "doc_number": doc_number,
            "effective_date": effective_date,
            "version_date": version_date,
            "law_level": law_level,
            "source_file": path.name,
            "source_type": "docx",
            "n_articles": len(articles),
        },
        "articles": articles,
    }


# ============================ PDF 汇编切分（P1b）============================
# 汇编《劳动关系相关政策法规汇编.pdf》：首页封面 + 目录(PDF页2-8) + 正文(94篇)。
# 已校验目录页码=打印页码=pdfplumber 1-based 页号（偏移 0）。
_TOC_CAT_RE  = re.compile(r"^[一二三四五六七八九十]+、")   # 目录类别行：一、二、…（仅中文数字）
_TOC_PM_RE   = re.compile(r"^—\s*[\d\s]+—$")             # 目录/正文页脚：— N —
# 目录行不变式：页码是行尾整数，其前至少一个点/空格分隔符（引导点可能多点/单点/纯空格）
_TOC_LINE_RE = re.compile(r"^(.*?)[.\s·．]+(\d{1,4})$")
# 正文页脚：整行只有破折号/空格/数字（如 — 28 3 —、958、45 7），且至少含一位数字
_FOOTER_RE   = re.compile(r"^[—\-\s\d]+$")


def parse_toc(pdf, toc_pages) -> list:
    """解析汇编目录，返回按目录顺序的 [(category, title, start_page), ...]。

    标题折行（无行尾页码的行）累积到下一行；类别行(一、二、…)只更新当前类别不入表。
    """
    result, pending, cur_cat = [], "", None
    for pno in toc_pages:
        text = pdf.pages[pno].extract_text() or ""
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.replace(" ", "") == "目录" or _TOC_PM_RE.match(line):
                continue
            m = _TOC_LINE_RE.match(line)
            if m:
                title = (pending + m.group(1)).strip(); pending = ""
                page = int(m.group(2))
                if _TOC_CAT_RE.match(title):
                    cur_cat = title
                else:
                    result.append((cur_cat, title, page))
            else:
                pending += line   # 标题折行，累积到下一有页码的行
    return result


def _clean_pdf_lines(pdf, p_start, p_end) -> list:
    """抽取打印页 [p_start, p_end] 的正文，去页脚并规范化，返回非空行列表。"""
    lines = []
    for pno in range(p_start, p_end + 1):
        if pno < 1 or pno > len(pdf.pages):
            continue
        text = pdf.pages[pno - 1].extract_text() or ""   # 打印页 -> 0-based
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if _FOOTER_RE.match(line) and re.search(r"\d", line):
                continue                                   # 丢弃页脚页码行
            line = normalize(line)
            if line:
                lines.append(line)
    return lines


def _title_sig(title: str) -> str:
    """标题签名：去空格与各类括号后取前 12 字，用于在正文中定位标题行（含折行）。"""
    s = re.sub(r"[《》〈〉（）()\[\]〔〕\s]", "", title)
    return s[:12]


def _find_title_line(lines, sig, start=0):
    """在 lines[start:] 中定位以 sig(标题签名)开头的行下标；找不到返回 None。

    严格前缀匹配（去括号空格后 startswith sig），避免"最高人民法院"等署名行
    被误判为下一篇标题首行而截断正文。
    """
    if not sig:
        return None
    for i in range(start, len(lines)):
        ls = re.sub(r"[《》〈〉（）()\[\]〔〕\s]", "", lines[i])
        if ls.startswith(sig):
            return i
    return None


# 发文封面标题的文种词，用于把"…印发《X》的通知"类封面行识别为下一篇起始
_COVER_DOCTYPE = "通知|公告|决定|命令|批复|复函|函|意见|规定|办法|规则|细则|方案|指引"


def _find_next_doc_start(lines, next_title, start=0):
    """在 lines[start:] 中定位下一篇正文的起始行下标；找不到返回 None。

    两种起始形态：
    1. 裸标题行——行首即下一篇标题（去括注后前缀匹配加长签名 long_sig）；
    2. 发文封面行——"[机关]关于(印发/发布…)《下一篇标题…》的(通知/公告…)"。目录常把
       此类篇目登记为所附文件名（如"技能人才最低工资分类参考指引"），而正文首行是完整
       通知标题；封面标题可能跨行折断，故合并相邻行后匹配。要求"《标题…》的+文种"结构，
       以区别于正文里"根据《某规则》第N条"这类句中引用（其后接"第"而非"的+文种"）。
    返回两种形态命中的较靠前者，用于把误纳入本篇尾部的下一篇封面页裁掉。
    """
    # 裸标题匹配须"确是下一篇标题"而非本篇内容，两道防误判：
    #   1) 跳过以《起首的行——本篇正文对法规/文件的引用（如"《X》已于…施行/现予公布"），其去
    #      括注后与下一篇标题前缀同形而易被误判（如"解释（一）"引用自身撞上下一篇"解释（二）"）；
    #   2) 命中行(可折行，合并相邻行)去括注后须匹配加长签名 long_sig（20 字），避免仅与本篇落款
    #      机关名同形的 12 字短前缀造成误截断（如落款"人力资源社会保障部办公厅"恰为下一篇发文机关）。
    long_sig = re.sub(r"[《》〈〉（）()\[\]〔〕\s]", "", next_title)[:20]
    if not long_sig:
        return None
    bare = None
    for i in range(start, len(lines)):
        if lines[i].lstrip().startswith("《"):
            continue
        joined = re.sub(r"[《》〈〉（）()\[\]〔〕\s]", "", "".join(lines[i:i + 3]))
        if joined.startswith(long_sig):
            bare = i
            break
    head = re.sub(r"[《》〈〉（）()\[\]〔〕\s]", "", next_title)[:6]
    # 封面首行须含"关于…《标题头"（封面起始行），再由相邻行合并确认"《标题…》的+文种"，
    # 这样命中的下标锚定在封面起始行本身，而非被合并进来的上一条尾行
    open_re = re.compile(rf"关于[^《》]{{0,20}}《{re.escape(head)}")
    full_re = re.compile(rf"《{re.escape(head)}[^》]*》的(?:{_COVER_DOCTYPE})")
    cover = None
    for i in range(start, len(lines)):
        if open_re.search(lines[i]) and full_re.search("".join(lines[i:i + 3])):
            cover = i
            break
    if bare is None:
        return cover
    if cover is None:
        return bare
    return min(bare, cover)


_PROSE_SEC_RE = re.compile(r"^[一二三四五六七八九十百]+、")   # 散文体一级分节：一、二、…


def _split_prose(lines) -> list:
    """将散文体（通知/意见/复函）正文按一级"一、二、"分节。

    首个"一、"之前为前言(section_no=None)；无"一、"结构时整体作为一节。
    节内各行按硬换行以空串拼接还原完整句子。
    """
    sections, cur_no, buf = [], None, []

    def flush():
        if buf:
            sections.append({"section_no": cur_no, "text": "".join(buf)})

    for line in lines:
        m = _PROSE_SEC_RE.match(line)
        if m:
            flush(); buf = []
            cur_no = m.group(0)[:-1]        # 去掉尾部"、"
        buf.append(line)
    flush()
    return sections


def _dedup_key(law_name: str) -> str:
    """去重键：去《》括注与（节录）等修饰及空格，用于 PDF 篇目与 docx 法规判重。"""
    s = re.sub(r"[《》\s]", "", law_name)
    s = re.sub(r"[（(](节录|试行)[)）]", "", s)
    return s


def parse_pdf_compilation(pdf_path, toc_pages=range(1, 8), docx_names=None) -> list:
    """切分《…汇编.pdf》为逐篇结构化 dict 列表。

    每篇：结构化(有第X条)走 _extract_articles 产出 articles；散文体产出 sections。
    与 docx_names 判重命中的篇目标记 is_duplicate=True（正文以 docx 为准，不入库）。
    """
    pdf_path = Path(pdf_path)
    docx_keys = {_dedup_key(n) for n in (docx_names or [])}
    docs = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        toc = parse_toc(pdf, toc_pages)
        total = len(pdf.pages)
        for i, (category, title, start) in enumerate(toc):
            nxt = toc[i + 1][2] if i + 1 < len(toc) else total + 1
            if nxt > start:
                # 常规：下一篇在下一页顶部，按页边界切分，避免误纳下一篇正文；
                # 但下一篇若带独立通知封面页（封面在其目录页前一页），会落到本篇尾部，
                # 故再按下一篇标题/封面裁掉尾部串味内容。
                p_end = min(nxt - 1, total)
                lines = _clean_pdf_lines(pdf, start, p_end)
                s_idx = _find_title_line(lines, _title_sig(title)) or 0
                e_idx = _find_next_doc_start(lines, toc[i + 1][1], s_idx + 1) \
                    if i + 1 < len(toc) else None
                body = lines[s_idx: e_idx if e_idx is not None else len(lines)]
            else:
                # 同页起始：目录页码不可靠，窗口延伸到再下一篇，靠标题裁切首尾
                win_end = toc[i + 2][2] if i + 2 < len(toc) else total + 1
                lines = _clean_pdf_lines(pdf, start, min(win_end, total))
                s_idx = _find_title_line(lines, _title_sig(title)) or 0
                e_idx = _find_next_doc_start(lines, toc[i + 1][1], s_idx + 1)
                body = lines[s_idx: e_idx if e_idx is not None else len(lines)]
                p_end = start

            intro = "".join(body[:8])                 # 文号/日期常在标题下数行内
            doc_number = extract_doc_number(intro)
            law_level = detect_law_level(title, doc_number)
            articles = _extract_articles(body, law_level, joiner="", strict_header=True)
            # 散文体(通知/意见/复函)可能行首引用"第X条"被误当条文；正文条文成序列(>=3条)才判为结构化
            if len(articles) >= 3:
                structure_type, sections = "articles", []
            else:
                structure_type, articles, sections = "prose", [], _split_prose(body)

            effective_date = extract_effective_date(intro)
            if effective_date is None:               # 施行日期也可能在文末
                effective_date = extract_effective_date("".join(body[-6:]))

            docs.append({
                "metainfo": {
                    "law_name": title,
                    "category": category,
                    "doc_number": doc_number,
                    "effective_date": effective_date,
                    "version_date": None,
                    "law_level": law_level,
                    "source_file": pdf_path.name,
                    "source_type": "pdf",
                    "structure_type": structure_type,
                    "page_start": start,
                    "page_end": p_end,
                    "n_articles": len(articles),
                    "n_sections": len(sections),
                    "is_duplicate": _dedup_key(title) in docx_keys,
                },
                "articles": articles,
                "sections": sections,
            })
    return docs


# ============================ 批量解析入口 ============================
def parse_all_docx(data_dir="data", output_dir="databases/parsed_laws",
                   write_registry=True):
    """解析 data_dir 下所有 docx，逐部写出 JSON，并汇总法规注册表。

    write_registry 为 True 时单独写出 laws_metadata.json（单独运行 docx 解析用）；
    parse_all 汇总 docx 与 PDF 时传 False，由其统一写注册表，避免被 PDF 篇目覆盖。
    返回注册表列表（每部法规一条元数据）。
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = []
    for path in sorted(data_dir.glob("*.docx")):
        result = parse_docx(path)
        out_path = output_dir / (path.stem + ".json")
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        meta = dict(result["metainfo"])
        meta["parsed_json"] = str(out_path)
        registry.append(meta)
        print(f"[OK] {meta['law_name']:24} 条数={meta['n_articles']:>4} "
              f"级别={meta['law_level']} 施行={meta['effective_date']}")

    if write_registry:
        registry_path = output_dir.parent / "laws_metadata.json"
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n共解析 {len(registry)} 部法规，注册表写入 {registry_path}")
    return registry


# 文件名安全化：去掉书名号/路径分隔等文件系统不友好字符，PDF 篇名较长时截断
_UNSAFE_FN_RE = re.compile(r"[《》〈〉/\\:：*?\"<>|\s]")


def _safe_filename(name: str, maxlen: int = 40) -> str:
    """把法规/篇目名转为安全文件名片段：去书名号与路径分隔符、合并空白、截断。"""
    s = _UNSAFE_FN_RE.sub("", name)
    return s[:maxlen]


def parse_all(data_dir="data", output_dir="databases/parsed_laws",
              pdf_name="劳动关系相关政策法规汇编.pdf"):
    """解析 data 下全部 docx 与 PDF 汇编，逐篇写 JSON，汇总统一注册表。

    去重：PDF 篇目与 docx 法规判重命中者（is_duplicate=True）以 docx 为准，
    仅登记不写 JSON、不入检索库（parsed_json=None）。
    返回统一注册表列表（docx 在前、PDF 篇目在后）。
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # docx：复用 parse_all_docx，但注册表交由本函数统一写出（避免被覆盖）
    registry = parse_all_docx(data_dir, output_dir, write_registry=False)
    docx_names = [m["law_name"] for m in registry]

    # PDF 汇编：逐篇切分；非重复篇写 JSON，重复篇仅登记
    pdf_path = data_dir / pdf_name
    docs = parse_pdf_compilation(pdf_path, docx_names=docx_names)
    n_write = n_dup = 0
    for d in docs:
        meta = dict(d["metainfo"])
        if meta["is_duplicate"]:
            meta["parsed_json"] = None            # 与 docx 重复，仅登记不写库
            n_dup += 1
        else:
            fname = f"汇编_{meta['page_start']:04d}_{_safe_filename(meta['law_name'])}.json"
            out_path = output_dir / fname
            out_path.write_text(
                json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            meta["parsed_json"] = str(out_path)
            n_write += 1
        registry.append(meta)
    print(f"[PDF] 汇编切分 {len(docs)} 篇：写出 {n_write}，去重跳过 {n_dup}")

    registry_path = output_dir.parent / "laws_metadata.json"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共登记 {len(registry)} 部法规/篇目（docx {len(docx_names)} + PDF {len(docs)}），"
          f"注册表写入 {registry_path}")
    return registry


if __name__ == "__main__":
    parse_all()
