import streamlit as st
import pandas as pd
import pdfplumber
import json
import re
import time
import uuid
import os
from datetime import datetime, timedelta
from typing import TypedDict

# LangChain / OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ==============================
# Setup LLM + DB
# ==============================
# ==============================
# OpenAI key via secrets or env
# ==============================
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY is missing. Add it in `.streamlit/secrets.toml` (local) or in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Use this client when creating the vector store
try:
    feedback_db = Chroma(
        embedding_function=embeddings,
        client=chroma_client,             # pass the custom client
        persist_directory="./feedback_db" # optional
    )
except Exception as e:
    st.warning(f"Chroma init failed: {e}")
    feedback_db = None


task_storage = {}

# ==============================
# PDF reading
# ==============================
def read_pdf_content(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


# ==============================
# Extraction pipeline (same as Flask)
# ==============================
class BillState(TypedDict):
    task_id: str
    content: str
    rag_examples: str
    formatted_prompt: str
    llm_response: str
    parsed_response: str
    json_valid: bool
    cleanup_response: str
    final_data: dict
    timings: dict

def extract_bill_data(content: str):
    task_id = str(uuid.uuid4())
    prompt = f"""
Extract bill data and return ONLY valid JSON:
{{
 "bill_date": "YYYY-MM-DD",
 "biller_name": "pharmacy name",
 "buyer_name": "buyer name",
 "bill_number": "bill number",
 "items": [
   {{
     "item_name": "item name",
     "count": 1,
     "per_unit_cost": 0.0,
     "total_price": 0.0,
     "tax_rate": 0.0
   }}
 ]
}}
Rules:
- If count missing, use 1
- If per_unit_cost missing, use total_price
- If tax_rate < 1, convert to %
Bill text:
{content}
"""
    response = llm.invoke(prompt)
    cleaned = re.sub(r"```json|```", "", response.content).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        data = {"error": "Invalid JSON from LLM", "raw": cleaned}

    # ✅ Always attach task_id
    data["task_id"] = task_id

    # ✅ Store in session_state
    if "tasks" not in st.session_state:
        st.session_state.tasks = {}
    st.session_state.tasks[task_id] = {
        "filename": "uploaded.pdf",  # placeholder
        "raw_json": data,
        "editable_json": data,
        "content": content,
        "timestamp": datetime.now()
    }

    return data



def submit_feedback(task_id, feedback_type, reason=""):
    if "tasks" not in st.session_state or task_id not in st.session_state.tasks:
        return {"error": "Task expired"}

    task_data = st.session_state.tasks[task_id]

    if feedback_db:
        try:
            doc = Document(
                page_content=task_data["content"][:500],
                metadata={
                    "task_id": task_id,
                    "feedback": feedback_type,
                    "reason": reason,
                    "timestamp": task_data["timestamp"].isoformat(),
                },
            )
            feedback_db.add_documents([doc])
        except Exception as e:
            return {"error": str(e)}

    del st.session_state.tasks[task_id]
    return {"message": "Feedback stored"}

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Bill Extractor", page_icon="📄", layout="wide")
st.title("📄 Bill Extractor — Table View + Feedback")

if "tasks" not in st.session_state:
    st.session_state.tasks = {}

pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if pdfs and st.button("Extract Bills"):
    st.session_state.tasks = {}
    for pdf in pdfs[:3]:
        with st.spinner(f"Processing {pdf.name}..."):
            content = read_pdf_content(pdf)
            data = extract_bill_data(content)
            st.session_state.tasks[data["task_id"]] = {
                "filename": pdf.name,
                "raw_json": data,
                "editable_json": data,
            }

if not st.session_state.get("tasks"):
    # Show welcome card
    with st.container():
        st.markdown(
            """
            <div style="
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
                background-color: white; 
                text-align: center;">
                <h3 style="color:#444;">👋 Welcome to Bill Extractor</h3>
                <p style="color:#666; font-size:16px;">
                    Please upload & extract a bill to get started.<br>
                    After that, you can review, edit, and send feedback here.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    for task_id, task in st.session_state.tasks.items():
        st.subheader(f"📄 {task['filename']} (Task ID: {task_id})")

        df = pd.DataFrame(task["editable_json"].items(), columns=["Field", "Value"])
        df["Value"] = df["Value"].astype(str)
        st.data_editor(df, use_container_width=True, disabled=["Field"])

        # ✅ Put Post button + checkbox side by side
        col1, col2 = st.columns([1, 3])
        with col1:
            st.button("Post", key=f"post_{task_id}")
        with col2:
            st.checkbox("Confirm", key=f"confirm_{task_id}")


if st.session_state.tasks:
    choices = {f"{v['filename']} ({k})": k for k, v in st.session_state.tasks.items()}
    choice = st.selectbox("Select file", list(choices.keys()))
    task_id = choices[choice]
    feedback_type = st.radio("Feedback", ["positive", "negative"], horizontal=True)
    reason = st.text_input("Reason (optional)")

    if st.button("Submit feedback"):
        result = submit_feedback(task_id, feedback_type, reason)
        st.json(result)
        if "message" in result:
            st.session_state.tasks.pop(task_id, None)
