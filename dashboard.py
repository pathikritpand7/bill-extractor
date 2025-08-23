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
from langchain_community.vectorstores import FAISS
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

# Example: create empty feedback DB
feedback_db = None

if "feedback_docs" not in st.session_state:
    st.session_state.feedback_docs = []

def get_feedback_db():
    global feedback_db
    if feedback_db is None:
        # Use stored documents to create FAISS index
        docs = st.session_state.feedback_docs
        if docs:
            feedback_db = FAISS.from_documents(docs, embeddings)
        else:
            
            dummy_doc = Document(page_content="init", metadata={}, id="dummy_0")
            feedback_db = FAISS.from_documents([dummy_doc], embeddings)
            feedback_db.docstore.delete(["dummy_0"])
    return feedback_db


# Load existing FAISS index if present
if os.path.exists("./feedback_db"):
    feedback_db = FAISS.load_local("./feedback_db", embeddings, allow_dangerous_deserialization=True)
else:
    feedback_db = get_feedback_db()

# Optional: immediately save the index folder if you want
feedback_db.save_local("./feedback_db")


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
    global feedback_db
    if "tasks" not in st.session_state or task_id not in st.session_state.tasks:
        return {"error": "Task expired"}

    task_data = st.session_state.tasks[task_id]

    # Use 'content' safely
    content_text = task_data.get("content", "")
    if not content_text:
        return {"error": "'content' missing in task_data"}

    if feedback_db:
        try:
            doc = Document(
                page_content=content_text[:500],
                metadata={
                    "task_id": task_id,
                    "feedback": feedback_type,
                    "reason": reason,
                    "timestamp": task_data["timestamp"].isoformat(),
                },
            )
            st.session_state.feedback_docs.append(doc)
            feedback_db = get_feedback_db()
            feedback_db.add_documents([doc])
            feedback_db.save_local("./feedback_db")
        except Exception as e:
            return {"error": str(e)}

    del st.session_state.tasks[task_id]
    return {"message": "Feedback stored"}



# ==============================
# Streamlit UI
# ==============================
import streamlit as st
from datetime import datetime
from typing import TypedDict
import uuid
import os
import json

# -----------------------------
# Import your backend functions
# -----------------------------
# from your_backend_module import read_pdf_content, extract_bill_data, submit_feedback, safe_float
# For this snippet, assume they are already defined in the same script

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Bill Extractor", page_icon="📄", layout="wide")
st.title("📄 Bill Extractor")

# -----------------------------
# Session State Initialization
# -----------------------------
if "last_jsons" not in st.session_state:
    st.session_state.last_jsons = []

if "task_ids" not in st.session_state:
    st.session_state.task_ids = []

if "task_id" not in st.session_state:
    st.session_state.task_id = None

if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {}

if "post_flag" not in st.session_state:
    st.session_state.post_flag = {}

# -----------------------------
# Upload PDFs
# -----------------------------
st.header("Upload PDF(s)")
pdfs = st.file_uploader("Choose up to 5 PDFs", type=["pdf"], accept_multiple_files=True)

if pdfs and st.button("Extract bills"):
    st.session_state.last_jsons = []
    st.session_state.task_ids = []
    with st.spinner("Extracting bills..."):
        for pdf in pdfs[:5]:
            try:
                pdf_text = read_pdf_content(pdf)
                data = extract_bill_data(pdf_text)
                st.session_state.last_jsons.append(data)
                st.session_state.task_ids.append(data.get("task_id"))
                st.success(f"✅ Extracted successfully: {pdf.name}")
                if data.get("task_id"):
                    st.session_state.task_id = data["task_id"]
            except Exception as e:
                st.error(f"❌ Extraction failed for {pdf.name}: {str(e)}")

# -----------------------------
# Initialize edit/post states
# -----------------------------
for idx in range(len(st.session_state.last_jsons)):
    if idx not in st.session_state.edit_mode:
        st.session_state.edit_mode[idx] = False
    if idx not in st.session_state.post_flag:
        st.session_state.post_flag[idx] = False

# -----------------------------
# Bill Viewer & Editor
# -----------------------------
st.header("📑 Bill Viewer & Editor")

for idx, bill in enumerate(st.session_state.last_jsons):
    with st.container():
        st.subheader(f"Bill #{idx+1} - {bill.get('bill_number', 'N/A')}")

        # Edit mode toggle
        if st.session_state.edit_mode[idx]:
            st.info("Editing Mode Enabled ✏️")
        else:
            st.caption("Read-only mode 🔒")

        # Invoice details
        st.markdown("Invoice Details")
        col1, col2 = st.columns(2)
        with col1:
            bill_date = st.text_input(
                "Bill Date", bill.get("bill_date", ""),
                disabled=not st.session_state.edit_mode[idx], key=f"date_{idx}"
            )
            bill_number = st.text_input(
                "Bill Number", bill.get("bill_number", ""),
                disabled=not st.session_state.edit_mode[idx], key=f"num_{idx}"
            )
            biller_name = st.text_input(
                "Biller Name", bill.get("biller_name", ""),
                disabled=not st.session_state.edit_mode[idx], key=f"biller_{idx}"
            )
        with col2:
            buyer_name = st.text_input(
                "Buyer Name", bill.get("buyer_name", ""),
                disabled=not st.session_state.edit_mode[idx], key=f"buyer_{idx}"
            )
            cumulative_tax = st.text_input(
                "Cumulative Tax", str(bill.get("cumulative_tax", 0.0)),
                disabled=not st.session_state.edit_mode[idx], key=f"tax_{idx}"
            )
            cumulative_total = st.text_input(
                "Cumulative Total", str(bill.get("cumulative_total", 0.0)),
                disabled=not st.session_state.edit_mode[idx], key=f"total_{idx}"
            )

        # Items table
        st.markdown("Bill Items")
        headers = ["Item Name", "Qty", "Per Unit Cost", "Tax Rate", "Total Price", "Total Tax", "Total Price With Tax"]
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        for col, h in zip(cols, headers):
            col.markdown(f"**{h}**")

        for j, item in enumerate(bill.get("items", [])):
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1, 1, 1, 1, 1, 1])
            with c1:
                item_name = st.text_input("Item Name", item.get("item_name", ""), disabled=not st.session_state.edit_mode[idx], key=f"name_{idx}_{j}", label_visibility="collapsed")
            with c2:
                count = st.text_input("Qty", str(item.get("count", 1)), disabled=not st.session_state.edit_mode[idx], key=f"count_{idx}_{j}", label_visibility="collapsed")
            with c3:
                per_unit_cost = st.text_input("Per Unit Cost", str(item.get("per_unit_cost", 0.0)), disabled=not st.session_state.edit_mode[idx], key=f"price_{idx}_{j}", label_visibility="collapsed")
            with c4:
                tax_rate = st.text_input("Tax Rate", str(item.get("tax_rate", 0.0)), disabled=not st.session_state.edit_mode[idx], key=f"taxrate_{idx}_{j}", label_visibility="collapsed")
            with c5:
                total_price = st.text_input("Total Price", str(item.get("total_price", 0.0)), disabled=not st.session_state.edit_mode[idx], key=f"tprice_{idx}_{j}", label_visibility="collapsed")
            with c6:
                total_tax = st.text_input("Total Tax", str(item.get("total_tax", 0.0)), disabled=not st.session_state.edit_mode[idx], key=f"ttax_{idx}_{j}", label_visibility="collapsed")
            with c7:
                total_price_with_tax = st.text_input("Total Price With Tax", str(item.get("total_price_with_tax", 0.0)), disabled=not st.session_state.edit_mode[idx], key=f"tpwt_{idx}_{j}", label_visibility="collapsed")

        # Action buttons
        # Action buttons
        colA, colB, colC = st.columns([1, 5, 1])  # More space in middle to push Save right
        with colA:
            st.session_state.post_flag[idx] = st.checkbox(
                "Select for Posting", value=st.session_state.post_flag[idx], key=f"post_{idx}"
            )
        with colB:
            if not st.session_state.edit_mode[idx] and st.button("✏️ Edit", key=f"edit_{idx}"):
                st.session_state.edit_mode[idx] = True
                st.success(f"Editing enabled for Bill {bill.get('bill_number', '')}")
        with colC:
            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
            if st.session_state.edit_mode[idx] and st.button("💾 Save", key=f"save_{idx}"):
                st.session_state.edit_mode[idx] = False
                st.success(f"Bill {bill.get('bill_number', '')} updated!")
            st.markdown("</div>", unsafe_allow_html=True)



    st.divider()

# -----------------------------
# Post Selected Bills
# -----------------------------
if st.button("🚀 Post Selected Bills"):
    selected_bills = [st.session_state.last_jsons[i] for i, v in st.session_state.post_flag.items() if v]
    if selected_bills:
        st.success(f"Posted {len(selected_bills)} bill(s) successfully!")
        st.json(selected_bills)
    else:
        st.warning("No bills selected for posting!")

# -----------------------------
# Feedback Section
# -----------------------------
st.header("Send Feedback")
disabled = st.session_state.task_id is None
if disabled:
    st.info("Upload a PDF first (task_id required). Tasks expire after ~5 minutes.")

col1, col2 = st.columns([1, 2])
with col1:
    feedback_type = st.radio("Feedback", ["positive", "negative"], horizontal=True, disabled=disabled)
with col2:
    reason = st.text_input("Reason (optional)", disabled=disabled)

if st.button("Submit Feedback", disabled=disabled):
    payload = {
        "task_id": st.session_state.task_id,
        "feedback": feedback_type,
        "reason": reason or ""
    }
    resp_json = submit_feedback(payload["task_id"], payload["feedback"], payload["reason"])
    st.subheader("Feedback Result")
    st.json(resp_json)
    if "error" not in resp_json:
        st.success("✅ Feedback sent. Task closed.")
        st.session_state.task_id = None
    else:
        st.error(f"❌ {resp_json['error']}")

