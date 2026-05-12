"""
DSA Mentor Backend
Stack: LangChain (RAG pipeline) + LangGraph (agentic chat) + FAISS (vector store)
"""

import os
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, Optional, List

from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from bs4 import BeautifulSoup

# ── LangChain core ──
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# ── LangChain Google Gemini (LLM only) ──
from langchain_google_genai import ChatGoogleGenerativeAI

# ── HuggingFace Embeddings (local, free, no API needed) ──
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# ── LangChain FAISS vector store ──
from langchain_community.vectorstores import FAISS

# ── LangGraph ──
from langgraph.graph import StateGraph, END



# ============================================================
# LLM + Embeddings (LangChain wrappers around Gemini)
# ============================================================

import time

def get_llm(temperature: float = 0.3):
    endpoint = HuggingFaceEndpoint(
        repo_id=os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        task="text-generation",
        max_new_tokens=1500,
        temperature=temperature,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
    )
    return ChatHuggingFace(llm=endpoint)


def llm_invoke_with_retry(chain, inputs: dict, max_retries: int = 3) -> str:
    """Invoke a LangChain chain with automatic retry on 429 rate limit errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                # Parse retry delay from error if available, else use backoff
                wait = 60 * (attempt + 1)
                try:
                    import re
                    match = re.search(r"retryDelay.*?(\d+)s", err)
                    if match:
                        wait = int(match.group(1)) + 5
                except Exception:
                    pass
                print(f"Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")

def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings — free, no API key needed."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ============================================================
# Global State
# ============================================================

sheet_data: List[dict] = []
topics_list: List[str] = []
topic_subtopics: dict = {}
faiss_store: Optional[FAISS] = None
question_context_store: dict = {}


def clear_state():
    global sheet_data, topics_list, topic_subtopics, faiss_store, question_context_store
    sheet_data = []
    topics_list = []
    topic_subtopics = {}
    faiss_store = None
    question_context_store = {}


# ============================================================
# Source Fetchers (parallel with ThreadPoolExecutor)
# ============================================================

def fetch_handwritten_notes(q: dict) -> str:
    return (q.get("notes") or "").strip()


def fetch_strivers_blog(url: str) -> str:
    if not url or not url.strip():
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url.strip(), headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = (
            soup.find("article") or soup.find("main") or
            soup.find("div", class_=lambda c: c and any(k in c.lower() for k in ["content", "post", "entry"])) or
            soup.find("div", id=lambda i: i and any(k in i.lower() for k in ["content", "post", "main"]))
        )
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
        return "\n".join(lines)[:2000]
    except Exception:
        return ""


def fetch_google_content(problem_name: str, subtopic: str) -> str:
    serpapi_key = os.environ.get("SERPAPI_KEY", "")
    query = f"{problem_name} {subtopic} DSA solution approach explanation"
    if serpapi_key:
        try:
            resp = requests.get("https://serpapi.com/search",
                                params={"q": query, "api_key": serpapi_key, "num": 3}, timeout=10)
            snippets = [r.get("snippet", "") for r in resp.json().get("organic_results", [])[:3]]
            return "\n".join(snippets)[:1200]
        except Exception:
            return ""
    else:
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            snippets = [s.get_text(strip=True) for s in soup.find_all("a", class_="result__snippet")[:4]]
            return "\n".join(snippets)[:1200]
        except Exception:
            return ""


def build_knowledge_context(q: dict) -> tuple[str, str]:
    """
    Fetch all 3 sources in parallel.
    Priority: Striver Blog (base) → Handwritten Notes (highest trust) → Google (fallback)
    """
    notes = fetch_handwritten_notes(q)
    blog_url = (q.get("blog_link") or "").strip()

    blog_content = ""
    google_content = ""

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if blog_url:
            futures["blog"] = executor.submit(fetch_strivers_blog, blog_url)
        futures["google"] = executor.submit(
            fetch_google_content, q.get("problem_name", ""), q.get("subtopic", "")
        )
        for key, future in futures.items():
            try:
                result = future.result(timeout=12)
                if key == "blog":
                    blog_content = result
                elif key == "google":
                    google_content = result
            except Exception:
                pass

    has_notes  = len(notes) > 50
    has_blog   = len(blog_content) > 80
    has_google = len(google_content) > 80

    sections = []
    sources_used = []

    if has_blog:
        sections.append(
            "=== [🌐 STRIVER'S BLOG — MAIN EXPLANATION BASE] ===\n"
            "Use this as the primary source for building the full explanation.\n\n"
            + blog_content[:2000]
        )
        sources_used.append("🌐 Striver's Blog")

    if has_notes:
        sections.append(
            "=== [⭐ STUDENT'S HANDWRITTEN NOTES — HIGHEST TRUST] ===\n"
            "IMPORTANT: These notes were written by the student themselves during learning.\n"
            "Prioritize these over blog content where they overlap.\n\n"
            + notes
        )
        sources_used.append("📝 Handwritten Notes (priority override)")

    if has_google and not has_blog and not has_notes:
        sections.append("=== [🔍 WEB SEARCH — FALLBACK ONLY] ===\n" + google_content[:1200])
        sources_used.append("🔍 Web Search (fallback)")
    elif has_google and not has_blog and has_notes:
        sections.append("=== [🔍 WEB SEARCH — SUPPLEMENT] ===\n" + google_content[:800])
        sources_used.append("🔍 Web Search (supplement)")

    if not sections:
        return "No notes, blog, or web results found. Use general DSA knowledge.", "⚠️ No sources found"

    return "\n\n".join(sections), " + ".join(sources_used)


# ============================================================
# LangChain RAG: Upload & Index into FAISS
# ============================================================

def process_upload(df: pd.DataFrame) -> tuple[bool, str]:
    global sheet_data, topics_list, topic_subtopics, faiss_store

    clear_state()

    required_cols = ['Topic Name', 'Sub Topic Name', 'Problem Name', 'Difficulty',
                     'Handwritten Typed Notes', 'Strivers Blog Links', 'Question Number']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}. Found: {list(df.columns)}"

    df = df.fillna('')
    rows = []
    for idx, row in df.iterrows():
        qn = row['Question Number']
        try:
            qn = int(qn)
        except (ValueError, TypeError):
            qn = 0
        rows.append({
            "id": f"row_{idx}",
            "topic": str(row['Topic Name']),
            "subtopic": str(row['Sub Topic Name']),
            "problem_name": str(row['Problem Name']),
            "difficulty": str(row['Difficulty']),
            "question_number": qn,
            "notes": str(row['Handwritten Typed Notes']),
            "blog_link": str(row['Strivers Blog Links']),
            "youtube_link": str(row['YouTube Video Link']) if 'YouTube Video Link' in row else "",
        })

    sheet_data = rows

    topics_set = set()
    topic_subtopics_map = {}
    for r in rows:
        topics_set.add(r["topic"])
        topic_subtopics_map.setdefault(r["topic"], set()).add(r["subtopic"])
    topics_list = sorted(list(topics_set))
    topic_subtopics = {t: sorted(list(s)) for t, s in topic_subtopics_map.items()}

    # Build LangChain Documents
    documents = []
    for r in rows:
        content = (
            f"Topic: {r['topic']} | Sub-topic: {r['subtopic']} | "
            f"Problem: {r['problem_name']} | Difficulty: {r['difficulty']} | "
            f"Notes: {r['notes']} | Blog: {r['blog_link']}"
        )
        doc = Document(
            page_content=content,
            metadata={
                "id": r["id"],
                "topic": r["topic"],
                "subtopic": r["subtopic"],
                "problem_name": r["problem_name"],
                "difficulty": r["difficulty"],
                "question_number": r["question_number"],
            }
        )
        documents.append(doc)

    # Index into FAISS using Gemini embeddings
    embeddings = get_embeddings()
    faiss_store = FAISS.from_documents(documents, embeddings)

    return True, f"Uploaded and indexed {len(rows)} questions."


# ============================================================
# LangChain LCEL Chain: Explanation Generator
# ============================================================

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — 2 approaches:
Input: Topic=Strings, Subtopic=Strings 1, Problem=Rotate String, Difficulty=Easy

🔗 Why This Problem Belongs to 'Strings 1'
This problem involves manipulating and comparing strings using substring operations.
It specifically relies on understanding how string rotations work and how substrings can represent different arrangements of the same characters.

💡 Approaches

### 🐢 Brute Force
We generate all possible rotations of the given string by shifting characters one by one. For each rotation, we compare it with the target string to check for a match. This ensures we explicitly test every possible rotated version.
Time Complexity: O(n²) (n rotations × O(n) comparison)
Space Complexity: O(n) (for storing rotated strings)
Why move to next: Generating every rotation is inefficient and repetitive.

### 🚀 Optimal
Instead of generating rotations, we concatenate the string with itself. This doubled string naturally contains all possible rotations as substrings. We then simply check if the target string exists inside it.
Time Complexity: O(n) (substring search)
Space Complexity: O(n) (for concatenated string)

Source used: Striver Blog

---
EXAMPLE 2 — 3 approaches:
Input: Topic=Dynamic Programming, Subtopic=on Subsequences, Problem=Subset sum equal to target (DP-14), Difficulty=Hard

🔗 Why This Problem Belongs to 'on Subsequences'
This problem is about deciding whether a subset (subsequence) of elements can form a given target sum.
It follows the classic pick/not-pick pattern, which is a key property of DP on subsequences where decisions are made at each index.

💡 Approaches

### 🐢 Brute Force
We generate all possible subsequences using recursion (pick and not pick) and check if any subsequence gives the target sum. At each index, we either include the element in the subset or exclude it, exploring all combinations.
Time Complexity: O(2ⁿ)
Space Complexity: O(n) (recursion stack)
Why move to next: Many overlapping subproblems are recomputed multiple times.

### ⚡ Better Approach
We apply memoization to store results of subproblems using a DP table dp[index][target]. Before computing a state, we check if it has already been solved. This avoids recomputation and significantly improves efficiency.
Time Complexity: O(n × target)
Space Complexity: O(n × target) + O(n) (stack space)
Why move to next: Still uses recursion stack and extra memory.

### 🚀 Optimal
We use tabulation (bottom-up DP) to build a 2D table where dp[i][j] tells whether a subset from index 0 to i can form sum j. We initialize base cases and iteratively fill the table using previous results. The final answer is stored at dp[n-1][target].
Time Complexity: O(n × target)
Space Complexity: O(n × target) → optimized to O(target)

Source used: Striver Blog
"""

EXPLANATION_SYSTEM = """You are an expert DSA mentor teaching a student about '{subtopic}'.

Follow this OUTPUT FORMAT exactly:

🔗 Why This Problem Belongs to '{subtopic}'
[Line 1: specific property/pattern that places it here]
[Line 2: how the core concept applies]

💡 Approaches

[1 APPROACH:]
### Approach: [Name]
[3-4 lines explanation]
Time Complexity: O(?)
Space Complexity: O(?)

[2 APPROACHES:]
### 🐢 Brute Force
[3-4 lines]
Time Complexity: O(?)
Space Complexity: O(?)
Why move to next: [one line]

### 🚀 Optimal
[3-4 lines]
Time Complexity: O(?)
Space Complexity: O(?)

[3 APPROACHES:]
### 🐢 Brute Force
[3-4 lines]
Time Complexity: O(?)
Space Complexity: O(?)
Why move to next: [one line]

### ⚡ Better Approach
[3-4 lines]
Time Complexity: O(?)
Space Complexity: O(?)
Why move to next: [one line]

### 🚀 Optimal
[3-4 lines]
Time Complexity: O(?)
Space Complexity: O(?)

Source used: {source_label}

RULES: Exactly 2 lines for subtopic section. 3-4 lines prose per approach. No extra sections.

FEW-SHOT EXAMPLES:
{few_shot_examples}
"""

EXPLANATION_HUMAN = """Problem: {problem_name}
Topic: {topic}
Subtopic: {subtopic}
Difficulty: {difficulty}

Knowledge Context:
{context_text}

{extra_context_section}

Determine approaches count from context (1, 2, or 3).
"""


def generate_single_explanation(q: dict, extra_context: str, subtopic: str) -> tuple[str, str, str]:
    """LangChain LCEL chain for single question explanation."""
    context_text, source_label = build_knowledge_context(q)

    llm = get_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXPLANATION_SYSTEM),
        ("human", EXPLANATION_HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    answer_text = llm_invoke_with_retry(chain, {
        "subtopic": subtopic,
        "source_label": source_label,
        "few_shot_examples": FEW_SHOT_EXAMPLES,
        "problem_name": q["problem_name"],
        "topic": q["topic"],
        "difficulty": q["difficulty"],
        "context_text": context_text if context_text else "No context. Use general DSA knowledge.",
        "extra_context_section": f"Additional context:\n{extra_context}" if extra_context else "",
    })

    return answer_text, context_text, source_label


def generate_similarity(answers: list, subtopic: str) -> str:
    """LangChain chain for similarity/grouping analysis. Runs once after all questions."""
    llm = get_llm(temperature=0.3)

    problem_list = "\n".join(
        [f"{i+1}. {a['question']} (Difficulty: {a['difficulty']})" for i, a in enumerate(answers)]
    )
    problem_summaries = "\n\n".join(
        [f"Q{i+1}. {a['question']}:\n{a['answer_text'][:200]}..." for i, a in enumerate(answers)]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a DSA mentor analyzing problems from subtopic '{subtopic}'.
Find which questions are LOGICALLY SIMILAR or EXTENSIONS of each other.

OUTPUT FORMAT:

## 🔁 Similarity Analysis

### Group [X]: [Shared logic name]
**Shared Logic**: (1 sentence)
**Problems**:
- Q[n]: [Name] — (base / similar / extension of Q?)

### Unique Problems
- Q[n]: [Name]

## 💡 Key Insight
(2-3 lines on what this tells the student about this subtopic)
"""),
        ("human", "Subtopic: {subtopic}\n\nProblems:\n{problem_list}\n\nSummaries:\n{problem_summaries}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return llm_invoke_with_retry(chain, {
        "subtopic": subtopic,
        "problem_list": problem_list,
        "problem_summaries": problem_summaries,
    })


# ============================================================
# LangGraph: Agentic Chat Graph
# ============================================================

class ChatState(TypedDict):
    subtopic: str
    user_query: str
    conversation_history: List[dict]
    matched_question: Optional[dict]
    context_text: str
    question_focus: str
    response: str


def node_detect_question(state: ChatState) -> ChatState:
    """Node 1: Detect which question the user is asking about."""
    query_lower = state["user_query"].lower()
    matched = None

    for store_key, ctx in question_context_store.items():
        if ctx.get("subtopic", "").lower() != state["subtopic"].lower():
            continue
        significant_words = [w for w in store_key.split() if len(w) > 3]
        if any(w in query_lower for w in significant_words):
            matched = ctx
            break

    return {**state, "matched_question": matched}


def node_fetch_context(state: ChatState) -> ChatState:
    """Node 2: Build LLM context from matched question or FAISS fallback."""
    context_sections = []
    question_focus = f"the subtopic '{state['subtopic']}'"

    if state["matched_question"]:
        mq = state["matched_question"]
        context_sections.append(
            f"=== GENERATED EXPLANATION FOR '{mq['problem_name']}' ===\n\n{mq['explanation']}"
        )
        if mq.get("source_context"):
            context_sections.append(
                f"=== RAW SOURCE MATERIAL ===\nSource: {mq['source_label']}\n\n{mq['source_context'][:1500]}"
            )
        question_focus = mq["problem_name"]

    elif faiss_store is not None:
        try:
            docs = faiss_store.similarity_search(state["user_query"], k=3)
            if docs:
                context_sections.append(
                    "=== RELEVANT CONTENT FROM YOUR NOTES ===\n"
                    + "\n---\n".join([d.page_content for d in docs])
                )
        except Exception:
            pass

    context_text = "\n\n".join(context_sections) if context_sections else "No specific context found."
    return {**state, "context_text": context_text, "question_focus": question_focus}


def node_generate_response(state: ChatState) -> ChatState:
    """Node 3: Generate chat response using LangChain with full conversation memory."""
    llm = get_llm(temperature=0.4)

    # Build message list with history (last 6 turns)
    recent_history = state["conversation_history"][-6:]
    messages = [SystemMessage(content=f"""You are an expert DSA mentor. Student is studying '{state["subtopic"]}'.

Answer accurately using the context. If asked for pseudocode give clean pseudocode.
If asked to re-explain, explain differently. Use conversation history for follow-ups.

Asking about: {state["question_focus"]}

=== KNOWLEDGE CONTEXT ===
{state["context_text"]}
""")]

    for turn in recent_history:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=state["user_query"]))

    result = llm.invoke(messages)
    response = result.content if hasattr(result, "content") else str(result)

    return {**state, "response": response}


def build_chat_graph():
    """Build and compile LangGraph StateGraph: detect → fetch → generate."""
    graph = StateGraph(ChatState)
    graph.add_node("detect_question", node_detect_question)
    graph.add_node("fetch_context", node_fetch_context)
    graph.add_node("generate_response", node_generate_response)
    graph.set_entry_point("detect_question")
    graph.add_edge("detect_question", "fetch_context")
    graph.add_edge("fetch_context", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()


# Compile once at import time
chat_graph = build_chat_graph()


# ============================================================
# Public API (called by Streamlit)
# ============================================================

def api_upload(df: pd.DataFrame) -> tuple[bool, str]:
    return process_upload(df)

def api_get_topics() -> List[str]:
    return topics_list

def api_get_subtopics(topic: str) -> List[str]:
    return topic_subtopics.get(topic, [])

def api_generate(topic: str, subtopic: str, progress_callback=None) -> dict:
    """
    Optimised generation:
    1. Scrape ALL sources in parallel first (biggest bottleneck)
    2. Call LLM sequentially with small delay (respect rate limits)
    3. Call progress_callback(current, total, question_name) for live UI updates
    """
    if not sheet_data or faiss_store is None:
        return {"error": "No data uploaded yet"}

    all_questions = [r for r in sheet_data if r["topic"] == topic and r["subtopic"] == subtopic]
    all_questions.sort(key=lambda x: (x["question_number"] == 0, x["question_number"]))

    if not all_questions:
        return {"error": "No questions found for this subtopic"}

    total = len(all_questions)

    def retrieve_context(query_text: str, exclude_problem: str, n: int = 2) -> str:
        try:
            docs = faiss_store.similarity_search(query_text, k=n + 1)
            filtered = [d.page_content for d in docs
                        if d.metadata.get("problem_name", "") != exclude_problem]
            return "\n---\n".join(filtered[:n])
        except Exception:
            return ""

    # ── STEP 1: Scrape ALL sources in parallel ──
    # This is the biggest time sink — blog scraping per question takes 3-8s each
    # Running all in parallel brings total scrape time = max(single scrape) instead of sum
    def scrape_one(q: dict) -> tuple[str, str, str]:
        """Returns (problem_name, context_text, source_label)"""
        context_text, source_label = build_knowledge_context(q)
        return q["problem_name"], context_text, source_label

    scraped = {}  # problem_name -> (context_text, source_label)
    with ThreadPoolExecutor(max_workers=min(total, 5)) as executor:
        futures = {executor.submit(scrape_one, q): q for q in all_questions}
        for future in futures:
            try:
                name, ctx, label = future.result(timeout=20)
                scraped[name] = (ctx, label)
            except Exception:
                q = futures[future]
                scraped[q["problem_name"]] = ("", "⚠️ Scrape failed")

    # ── STEP 2: LLM calls sequentially with rate-limit-safe delay ──
    # Free tier: ~15 RPM → 4s between calls is safe
    DELAY_BETWEEN_CALLS = int(os.environ.get("QUESTION_DELAY", 4))

    answers = []
    for idx, q in enumerate(all_questions):
        if progress_callback:
            progress_callback(idx, total, q["problem_name"])

        query_text = f"{q['problem_name']} {q['subtopic']} {q['difficulty']}"
        extra_context = retrieve_context(query_text, q["problem_name"])

        context_text, source_label = scraped.get(q["problem_name"], ("", "⚠️ No source"))

        # Build and invoke the LangChain LCEL chain directly
        # (reuse already-scraped context instead of re-scraping)
        try:
            llm = get_llm(temperature=0.3)
            prompt = ChatPromptTemplate.from_messages([
                ("system", EXPLANATION_SYSTEM),
                ("human", EXPLANATION_HUMAN),
            ])
            chain = prompt | llm | StrOutputParser()

            answer_text = llm_invoke_with_retry(chain, {
                "subtopic": subtopic,
                "source_label": source_label,
                "few_shot_examples": FEW_SHOT_EXAMPLES,
                "problem_name": q["problem_name"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "context_text": context_text if context_text else "No context. Use general DSA knowledge.",
                "extra_context_section": f"Additional context:\n{extra_context}" if extra_context else "",
            })
        except Exception as e:
            answer_text = f"⚠️ Error: {str(e)}"
            source_context = ""
            traceback.print_exc()

        store_key = q["problem_name"].lower().strip()
        question_context_store[store_key] = {
            "problem_name": q["problem_name"],
            "topic": q["topic"],
            "subtopic": subtopic,
            "difficulty": q["difficulty"],
            "explanation": answer_text,
            "source_context": context_text,
            "source_label": source_label,
            "notes": q.get("notes", ""),
            "blog_link": q.get("blog_link", ""),
        }

        answers.append({
            "question": q["problem_name"],
            "difficulty": q["difficulty"],
            "answer_text": answer_text,
            "blog_link": q.get("blog_link", ""),
            "youtube_link": q.get("youtube_link", ""),
        })

        # Respect free tier rate limit between LLM calls
        if idx < total - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    if progress_callback:
        progress_callback(total, total, "Done")

    similarity_result = None
    if len(answers) > 1:
        try:
            similarity_result = generate_similarity(answers, subtopic)
        except Exception as e:
            similarity_result = f"⚠️ Similarity error: {str(e)}"

    return {
        "answers": answers,
        "similarity": similarity_result,
        "total": len(answers),
        "message": f"All {len(answers)} questions explained!",
    }


def api_chat(subtopic: str, user_query: str, conversation_history: List[dict]) -> dict:
    if not sheet_data or faiss_store is None:
        return {"error": "No data uploaded yet"}

    result = chat_graph.invoke({
        "subtopic": subtopic,
        "user_query": user_query,
        "conversation_history": conversation_history,
        "matched_question": None,
        "context_text": "",
        "question_focus": "",
        "response": "",
    })

    return {
        "response": result["response"],
        "matched_question": result["matched_question"]["problem_name"] if result.get("matched_question") else None,
    }
