# DSA_Mentor_AI

## 🚀 Overview

DSA_Mentor_AI is a RAG-based (Retrieval-Augmented Generation) intelligent DSA revision assistant built to solve a very common problem faced during coding interview preparation:

> After completing hundreds of DSA questions from sheets like Striver’s SDE Sheet, we slowly start forgetting important concepts, approaches, optimizations, and patterns.

Re-solving every problem again is time-consuming and inefficient.

This project helps in smart revision by combining:

- 📄 Structured DSA question sheets (Excel format)
- 📝 Personal handwritten notes
- 🤖 AI-generated explanations and summaries
- 🔍 Retrieval-Augmented Generation (RAG)
- 💬 Context-aware chatbot support

The system allows users to revise an entire topic/subtopic in a clean and structured way without manually opening multiple tabs, notes, blogs, and videos.

---

# ✨ Features

## 📚 Smart Topic-wise Revision

Upload a structured Excel file containing DSA questions categorized by:

- Topic
- Subtopic
- Question Name
- Links
- Notes references

Then revise any topic or subtopic instantly.

---

## 🧠 RAG-Based Retrieval System

The application retrieves:

- Relevant questions
- Related handwritten notes
- Online explanation resources
- Contextual information

and combines them to generate intelligent revision summaries.

---

## 📝 AI-Generated Structured Explanations

For every question, the system can generate:

- Problem intuition
- Brute force approach
- Better approach
- Optimal approach
- Time complexity
- Space complexity
- Important observations
- Pattern recognition insights

---

## 🔗 Resource Integration

The generated revision output can include:

- Blog references
- YouTube explanations
- Online resources
- Personal notes context

This creates a centralized revision workflow.

---

## 💬 Context-Aware Chatbot

The chatbot layer allows users to ask follow-up questions such as:

- “Explain this DP transition again”
- “Give pseudocode for this problem”
- “Why is greedy not working here?”
- “Compare memoization vs tabulation”

The chatbot maintains conversational context and answers using the retrieved revision material.

---

# 🏗️ Tech Stack

## Frontend
- Streamlit

## Backend
- Python

## AI / RAG Stack
- LangChain
- LangGraph
- FAISS
- Sentence Transformers
- HuggingFace Models

## Data Processing
- Pandas
- BeautifulSoup

---

# 📂 Project Structure

```bash
DSA_Mentor_AI/
│
├── app_streamlit.py          # Streamlit frontend application
├── backend.py                # Backend logic and RAG pipeline
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables (local only)
├── README.md                 # Project documentation
│
├── example.xlsx              # Example Excel formatting file
│
└── dsa_env/                  # Virtual environment (generated locally)
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ShreyanshDangi/DSA_Mentor_AI.git
```

---

## 2️⃣ Move into the Project Directory

```bash
cd DSA_Mentor_AI
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv dsa_env
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
dsa_env\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Run the Streamlit Application

```bash
streamlit run app_streamlit.py
```

---

# 📄 Excel File Format

The system expects a structured Excel file.

An example file is provided:

```bash
example.xlsx
```

Use this file as a reference for formatting your own DSA sheets.

---

## ✅ Recommended Columns

The uploaded Excel file should follow the structure below:

| Topic Number | Sub Topic Number | Question Number | Topic Name | Sub Topic Name | Problem Name | Difficulty | YouTube Video Link | Strivers Blog Links | Handwritten Typed Notes |
|------|------|------|------|------|------|------|------|------|------|
| 8 | 8.3 | 8.3.3 | Bit Manipulation | Advanced Maths | Count primes in range L to R | Medium | YouTube Link | Blog Link | Important observations and intuition |

You can modify the Excel data as needed, but the column names must remain unchanged for the system to work correctly.

---

# 🔐 Environment Variables

Create a `.env` file in the root directory.

Example:

```env
GEMINI_API_KEY=your_api_key_here
QUESTION_DELAY=2
GEMINI_MODEL=gemini-2.0-flash-lite

HF_TOKEN=your_api_key_here
HF_MODEL=Qwen/Qwen2.5-72B-Instruct
```

---

# 🧪 Example Workflow

## Step 1
Upload the Excel file containing DSA questions.

## Step 2
Select a topic/subtopic for revision.

## Step 3
The system retrieves:

- Related questions
- Notes
- Contextual resources
- Explanations

## Step 4
AI generates structured revision summaries.

## Step 5
Ask follow-up questions using the chatbot.

---

# 🎯 Motivation Behind the Project

During DSA preparation, solving questions once is not enough.

The real challenge is:

- remembering patterns,
- retaining intuition,
- revising efficiently,
- and reconnecting related concepts.

This project was built to make DSA revision:

- smarter,
- faster,
- structured,
- and AI-assisted.

---

# 🚀 Future Improvements

Planned features for upcoming versions:

- 📊 Revision analytics dashboard
- 🔁 Spaced repetition tracking
- 🧠 Better retrieval evaluation metrics
- 🎯 Personalized explanation styles
- 🗂️ Multi-file support
- 📈 Multi-LLM comparison system
- 🎤 Voice-based revision assistant
- 📝 Auto-generated flashcards & quizzes
- ☁️ Cloud deployment

---

# 🤝 Contributions

Contributions, suggestions, and feedback are welcome.

Feel free to:

- Open issues
- Suggest improvements
- Fork the project
- Submit pull requests

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed by: **Shreyansh Dangi**

GitHub: https://github.com/ShreyanshDangi

---

# ⭐ Support

If you found this project useful, consider giving the repository a star ⭐

It helps the project reach more developers and learners.

