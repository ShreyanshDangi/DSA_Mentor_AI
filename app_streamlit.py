"""
DSA Mentor – Phase 1 Frontend
Stack: Streamlit
Calls backend.py (LangChain + LangGraph) directly — no HTTP layer needed.
"""

import streamlit as st
import pandas as pd
import tempfile
import os

# Import backend functions directly
from backend import (
    api_upload,
    api_get_topics,
    api_get_subtopics,
    api_generate,
    api_chat,
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="DSA Mentor RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp { background-color: #0f1117; color: #e8eaf6; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2e3250;
}
[data-testid="stSidebar"] .stMarkdown { color: #e8eaf6; }

/* Headers */
h1 { background: linear-gradient(90deg, #6c63ff, #00d2ff);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 700 !important; }
h2, h3 { color: #e8eaf6 !important; }

/* Cards / expanders */
[data-testid="stExpander"] {
    background: #1a1d27;
    border: 1px solid #2e3250;
    border-radius: 12px;
    margin-bottom: 12px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Generate button — green */
.generate-btn > button {
    background: linear-gradient(135deg, #00c896, #00b4d8) !important;
    color: #0f1117 !important;
}

/* Success / error boxes */
.stSuccess { background: rgba(0,200,150,0.1) !important; border-color: rgba(0,200,150,0.3) !important; }
.stError   { background: rgba(255,95,126,0.1) !important; border-color: rgba(255,95,126,0.2) !important; }
.stInfo    { background: rgba(108,99,255,0.1) !important; border-color: rgba(108,99,255,0.2) !important; }

/* Select boxes */
.stSelectbox [data-baseweb="select"] {
    background: #22263a !important;
    border-color: #2e3250 !important;
    color: #e8eaf6 !important;
    border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1d27 !important;
    border: 2px dashed #2e3250 !important;
    border-radius: 12px !important;
}

/* Difficulty badges */
.badge-easy   { background: rgba(0,200,150,0.15); color: #00c896; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
.badge-medium { background: rgba(255,181,71,0.15); color: #ffb547; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
.badge-hard   { background: rgba(255,95,126,0.15); color: #ff5f7e; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; }

/* Chat message bubbles */
.chat-user {
    background: linear-gradient(135deg, #6c63ff, #8b5cf6);
    color: white;
    padding: 10px 14px;
    border-radius: 14px 14px 3px 14px;
    margin: 6px 0 6px 60px;
    font-size: 13px;
    line-height: 1.5;
}
.chat-bot {
    background: #22263a;
    border: 1px solid #2e3250;
    color: #e8eaf6;
    padding: 10px 14px;
    border-radius: 3px 14px 14px 14px;
    margin: 6px 60px 6px 0;
    font-size: 13px;
    line-height: 1.6;
}
.matched-tag {
    background: rgba(0,210,255,0.1);
    color: #00d2ff;
    border: 1px solid rgba(0,210,255,0.2);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin-bottom: 6px;
    display: inline-block;
}
.similarity-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.08), rgba(0,210,255,0.05));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 12px;
    padding: 20px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialisation
# ============================================================
def init_session():
    defaults = {
        "uploaded": False,
        "topics": [],
        "selected_topic": None,
        "selected_subtopic": None,
        "answers": [],
        "similarity": None,
        "conversation_history": [],   # [{role, content}]
        "current_subtopic": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ============================================================
# Sidebar — Upload + Topic/Subtopic Selection
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 DSA Mentor RAG")
    st.markdown("*Phase 1 · LangChain + LangGraph*")
    st.divider()

    # ── Upload ──
    st.markdown("### 📁 Upload DSA Sheet")
    uploaded_file = st.file_uploader(
        "Choose your Excel file (.xlsx)",
        type=["xlsx", "xls"],
        help="Must have columns: Topic Name, Sub Topic Name, Problem Name, Difficulty, Handwritten Typed Notes, Strivers Blog Links, Question Number"
    )

    if uploaded_file is not None:
        if st.button("⬆️ Upload & Index", use_container_width=True):
            with st.spinner("Indexing into FAISS with Gemini embeddings..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    df = pd.read_excel(tmp_path, engine='openpyxl')
                    os.unlink(tmp_path)

                    success, message = api_upload(df)
                    if success:
                        st.session_state.uploaded = True
                        st.session_state.topics = api_get_topics()
                        st.session_state.answers = []
                        st.session_state.similarity = None
                        st.session_state.conversation_history = []
                        st.session_state.current_subtopic = None
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.divider()

    # ── Topic / Subtopic selectors ──
    if st.session_state.uploaded and st.session_state.topics:
        st.markdown("### 🗂️ Select Topic")
        selected_topic = st.selectbox(
            "Topic",
            options=st.session_state.topics,
            label_visibility="collapsed",
            key="topic_select"
        )
        st.session_state.selected_topic = selected_topic

        subtopics = api_get_subtopics(selected_topic) if selected_topic else []
        st.markdown("### 📂 Select Subtopic")
        selected_subtopic = st.selectbox(
            "Subtopic",
            options=subtopics,
            label_visibility="collapsed",
            key="subtopic_select"
        )
        st.session_state.selected_subtopic = selected_subtopic

        st.divider()

        # ── Generate button ──
        with st.container():
            if st.button("✨ Generate Explanations", use_container_width=True, type="primary"):
                if not selected_topic or not selected_subtopic:
                    st.warning("Please select both a topic and subtopic.")
                else:
                    st.session_state.answers = []
                    st.session_state.similarity = None
                    st.session_state.conversation_history = []
                    st.session_state.current_subtopic = selected_subtopic

                    # Live progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(current, total, question_name):
                        pct = int((current / total) * 100) if total > 0 else 0
                        progress_bar.progress(pct)
                        if current < total:
                            status_text.caption(f"⚙️ Generating {current+1}/{total}: **{question_name}**")
                        else:
                            status_text.caption(f"✅ All {total} questions done!")

                    result = api_generate(selected_topic, selected_subtopic, progress_callback)

                    progress_bar.empty()
                    status_text.empty()

                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state.answers = result.get("answers", [])
                        st.session_state.similarity = result.get("similarity")
                        st.success(f"✅ {result['message']}")

    elif not st.session_state.uploaded:
        st.info("Upload your Excel sheet to get started.")


# ============================================================
# Main Content Area
# ============================================================
st.title("🧠 DSA Mentor RAG System")
st.caption("Phase 1 · LangChain + LangGraph + FAISS + Gemini · AI-powered DSA study companion")

# ── No data yet ──
if not st.session_state.uploaded:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #8b90b0;">
        <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
        <h3 style="color: #e8eaf6;">Upload your DSA sheet to begin</h3>
        <p>Use the sidebar to upload your Excel file and start generating explanations.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Results ──
if st.session_state.answers:
    subtopic_label = st.session_state.current_subtopic or ""
    st.markdown(f"### 📚 Explanations for **{subtopic_label}**")
    st.caption(f"{len(st.session_state.answers)} questions explained")

    def fix_formatting(text: str) -> str:
        import re
        # Hard replace every occurrence of these labels with a guaranteed
        # double-newline before them so markdown renders them as new paragraphs.
        # Works whether they appear inline or already on a new line.
        labels = [
            'Time Complexity:',
            'Space Complexity:',
            'Why move to next:',
            'Why move ahead:',
            'Source used:',
        ]
        for label in labels:
            # Remove any existing leading newlines around the label first,
            # then add exactly two newlines before it — clean and consistent
            text = re.sub(r'\n*(' + re.escape(label) + r')', r'\n\n**' + label + r'**', text)
        return text

    for idx, ans in enumerate(st.session_state.answers):
        diff = (ans.get("difficulty") or "").lower()
        if "easy" in diff:
            badge_html = '<span class="badge-easy">Easy</span>'
            diff_emoji = "🟢"
        elif "hard" in diff:
            badge_html = '<span class="badge-hard">Hard</span>'
            diff_emoji = "🔴"
        else:
            badge_html = '<span class="badge-medium">Medium</span>'
            diff_emoji = "🟡"

        with st.expander(f"Q{idx+1}. {ans['question']}  {diff_emoji}", expanded=(idx == 0)):
            # ── Three info boxes: Difficulty | Blog | YouTube ──
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"<div style='background:#1a1d27;border:1px solid #2e3250;border-radius:8px;"
                    f"padding:8px 12px;text-align:center'>"
                    f"<div style='font-size:10px;color:#8b90b0;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>Difficulty</div>"
                    f"<div>{badge_html}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col2:
                blog = (ans.get("blog_link") or "").strip()
                if blog and blog != "nan":
                    st.markdown(
                        f"<div style='background:#1a1d27;border:1px solid #2e3250;border-radius:8px;"
                        f"padding:8px 12px;text-align:center'>"
                        f"<div style='font-size:10px;color:#8b90b0;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>Online Blog</div>"
                        f"<a href='{blog}' target='_blank' style='color:#00d2ff;font-size:12px;text-decoration:none'>📑 Read Blog ↗</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:#1a1d27;border:1px solid #2e3250;border-radius:8px;"
                        f"padding:8px 12px;text-align:center'>"
                        f"<div style='font-size:10px;color:#8b90b0;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>Online Blog</div>"
                        f"<span style='color:#8b90b0;font-size:12px'>Not available</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            with col3:
                yt = (ans.get("youtube_link") or "").strip()
                if yt and yt != "nan":
                    st.markdown(
                        f"<div style='background:#1a1d27;border:1px solid #2e3250;border-radius:8px;"
                        f"padding:8px 12px;text-align:center'>"
                        f"<div style='font-size:10px;color:#8b90b0;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>YouTube</div>"
                        f"<a href='{yt}' target='_blank' style='color:#ff5f7e;font-size:12px;text-decoration:none'>▶️ Watch Video ↗</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background:#1a1d27;border:1px solid #2e3250;border-radius:8px;"
                        f"padding:8px 12px;text-align:center'>"
                        f"<div style='font-size:10px;color:#8b90b0;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px'>YouTube</div>"
                        f"<span style='color:#8b90b0;font-size:12px'>Not available</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            st.markdown(fix_formatting(ans["answer_text"]))

    # Similarity Section
    if st.session_state.similarity:
        st.markdown("---")
        st.markdown(
            f'<div class="similarity-box">'
            f'<h3 style="color:#6c63ff; margin-top:0">🔁 Similarity Analysis</h3>'
            f'{st.session_state.similarity}'
            f'</div>',
            unsafe_allow_html=True
        )

elif st.session_state.uploaded and st.session_state.selected_subtopic:
    st.info("Select a topic and subtopic from the sidebar, then click **Generate Explanations**.")
else:
    st.info("Select a topic and subtopic from the sidebar to generate explanations.")


# ============================================================
# Chat Section
# ============================================================
st.markdown("---")
st.markdown("### 💬 Ask DSA Mentor")

if not st.session_state.current_subtopic:
    st.info("Generate explanations for a subtopic first, then ask questions here.")
else:
    st.caption(
        f"🎯 Focused on: **{st.session_state.current_subtopic}** · "
        f"Ask: *'pseudocode for X'*, *'explain brute force of X differently'*, *'TC of X?'*"
    )

    # ── Display conversation history ──
    for turn in st.session_state.conversation_history:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{turn["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            if turn.get("matched_question"):
                st.markdown(
                    f'<div class="matched-tag">📌 Context: {turn["matched_question"]}</div>',
                    unsafe_allow_html=True
                )
            st.markdown(turn["content"])

    # ── Input area — st.chat_input auto-clears after every submit ──
    user_input = st.chat_input("Ask about any generated question... e.g. 'pseudocode for Two Sum'")

    if user_input and user_input.strip():
        query = user_input.strip()

        st.markdown(
            f'<div class="chat-user">{query}</div>',
            unsafe_allow_html=True
        )

        st.session_state.conversation_history.append({
            "role": "user",
            "content": query
        })

        with st.spinner("Thinking..."):
            result = api_chat(
                subtopic=st.session_state.current_subtopic,
                user_query=query,
                conversation_history=st.session_state.conversation_history,
            )

        if "error" in result:
            st.error(result["error"])
            full_response = f"⚠️ {result['error']}"
            matched_question = None
        else:
            full_response = result["response"]
            matched_question = result.get("matched_question")

            if matched_question:
                st.markdown(
                    f'<div class="matched-tag">📌 Context: {matched_question}</div>',
                    unsafe_allow_html=True
                )
            st.markdown(full_response)

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": full_response,
            "matched_question": matched_question,
        })

        st.rerun()

    # ── Clear chat ──
    if st.session_state.conversation_history:
        st.markdown("")
        if st.button("🗑️ Clear Chat"):
            st.session_state.conversation_history = []
            st.rerun()
