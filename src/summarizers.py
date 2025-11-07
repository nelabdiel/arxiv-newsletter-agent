# src/summarizers.py

from typing import List, Tuple
import re
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

def make_llm(base_url: str, model: str) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.15,
        num_predict=200,
        repeat_penalty=1.08,
    )

def _only_first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip()

def _strip_preamble(line: str) -> str:
    line = re.sub(r'^[-\\*\\s]*here\\s+are.*?:\\s*', '', line, flags=re.I)
    line = re.sub(r'^(options?:|title:|topic:|label:|suggested:?\\s*)', '', line, flags=re.I)
    line = re.sub(r'^[\\-\\*>\\s]+', '', line)
    return line.strip()

def topic_label(llm: ChatOllama, titles: List[str]) -> str:
    prompt = (
        "Return ONLY a concise topic label (3–6 words), no punctuation, no quotes, no markdown, "
        "no options list, no commentary. Use shared concepts across these titles.\\n"
        + "\\n".join(f"- {t}" for t in titles[:6])
    )
    msg = llm.invoke([SystemMessage(content="You are a scientific copy editor."), HumanMessage(content=prompt)])
    line = _only_first_line(msg.content)
    line = _strip_preamble(line)
    line = re.sub(r'[.:;!?,]+$', '', line)
    return line

def _citations(papers: List[dict]) -> str:
    return ", ".join(f"[{p['id']}]({p['links']['abs']})" for p in papers)

def _sanitize_bullets(text: str) -> str:
    lines = []
    for raw in text.strip().splitlines():
        s = raw.strip()
        # strip any inline '\\n' literals that models sometimes print
        s = s.replace('\\\\n', ' ')
        if not s:
            continue
        if s[0] in '*•-':
            s = re.sub(r'^[*•-]\\s*', '- ', s)
        else:
            s = '- ' + s
        lines.append(s)
        if len(lines) == 3:
            break
    return "\\n".join(lines)

def summarize_science(llm: ChatOllama, cluster: List[dict], max_items: int = 5) -> Tuple[str, str]:
    items = cluster[:max_items]
    ids = ", ".join(f"[{p['id']}]" for p in items)
    bullets = "\\n".join(f"- [{p['id']}] {p['title']} — {', '.join(p['authors'])}" for p in items)
    prompt = (
        "You are a careful scientist. Using ONLY the evidence provided, produce EXACTLY 3 bullet points "
        "(each ≤ 18 words). Be precise, neutral, avoid hype. "
        f"Use bracketed IDs to cite: {ids}. "
        "Output MUST be only the three bullets, nothing else.\\n\\n"
        "Evidence (titles & authors):\\n" + bullets + "\\n\\n"
        "Abstract snippets:\\n" + "\\n".join(f"[{p['id']}] {p['summary']}" for p in items)
    )
    out = llm.invoke([SystemMessage(content="Follow instructions exactly."), HumanMessage(content=prompt)]).content
    return _sanitize_bullets(out), _citations(items)

def summarize_comms(llm: ChatOllama, cluster: List[dict], max_items: int = 5) -> Tuple[str, str]:
    items = cluster[:max_items]
    ids = ", ".join(f"[{p['id']}]" for p in items)
    prompt = (
        "Write ONE paragraph (≤ 80 words) explaining the big picture for a technically literate audience. "
        "Add a final sentence starting with 'Why it matters:' (≤ 16 words). "
        f"Append citations at the end as {ids}. "
        "No headings, lists, questions, or preambles.\\n\\n"
        "Titles + abstracts:\\n" + "\\n".join(f"[{p['id']}] {p['title']}. {p['summary']}" for p in items)
    )
    out = llm.invoke([SystemMessage(content="Be concise and non-hyped."), HumanMessage(content=prompt)]).content.strip()
    out = re.sub(r'\\\\n', ' ', out)
    out = re.sub(r'\\s+', ' ', out).strip()
    return out, _citations(items)
