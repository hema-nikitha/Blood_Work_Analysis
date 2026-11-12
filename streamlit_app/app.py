# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Blood Work Analysis", page_icon="🩸", layout="wide")
st.title("🩸 Blood Work Analysis")
st.caption("Extracts blood test values, flags them HIGH/LOW/NORMAL, and generates a summary and diet plan.")


@st.cache_resource
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Add it to your .env file.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemma-4-31b-it", google_api_key=api_key)


def extract_values(llm, bloodwork: str) -> str:
    prompt = f"""
You are a medical data extraction assistant.

Extract all test values and classify each as HIGH, LOW,
or NORMAL based on the reference ranges provided.

Format:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{bloodwork}
"""
    return llm.invoke(prompt).text


def build_diet_plan(llm, extracted: str) -> str:
    prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.
Based on the blood work analysis below, write:
1. A short health summary in 3 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only 2 sections: 1. Foods to avoid 2. Foods to eat more often
Don't include any other sections in the diet plan

Blood work analysis:
{extracted}
"""
    return llm.invoke(prompt).text


# ---------- Input ----------
uploaded = st.file_uploader("Upload a blood report (.txt)", type=["txt"])

if uploaded is not None:
    bloodwork = uploaded.read().decode("utf-8")
else:
    default_path = "bloodwork.txt"
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            bloodwork = f.read()
        st.info("Using bloodwork.txt from the app folder. Upload a file to use a different report.")
    else:
        bloodwork = None
        st.info("Upload a blood report to begin.")

with st.expander("Show report text"):
    if bloodwork:
        st.text(bloodwork)

# ---------- Analysis ----------
if bloodwork and st.button("Analyze", type="primary"):
    llm = get_llm()

    with st.spinner("Stage 1: extracting and classifying values..."):
        extracted = extract_values(llm, bloodwork)
    st.subheader("Stage 1: Extracted Values")
    st.code(extracted, language="text")

    with st.spinner("Stage 2: writing summary and diet plan..."):
        plan = build_diet_plan(llm, extracted)
    st.subheader("Stage 2: Health Summary and Diet Plan")
    st.markdown(plan)

    st.caption("This is an AI-generated summary, not medical advice. Please consult a doctor.")