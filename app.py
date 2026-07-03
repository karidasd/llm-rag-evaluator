import streamlit as st
import os
import shutil
import tempfile
import pandas as pd
import plotly.express as px

from core.rag_pipeline import RAGPipelineManager
from core.evaluator import RagasEvaluator
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAGEval v2.0", page_icon="🦑", layout="wide")

st.title("🦑 RAGEval v2.0: AI Data Pipeline Evaluator")
st.markdown("Evaluate LLMs via Groq using **Ragas** metrics. Now with **PDF Support**, **ChromaDB**, and **Web Search Fallback**!")

# Sidebar
with st.sidebar:
    st.header("Settings ⚙️")
    groq_key = st.text_input("Groq API Key (Optional if in .env)", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    st.subheader("Model Comparison (LLMs)")
    model_a = st.selectbox("Model A", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=0)
    model_b = st.selectbox("Model B", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=2)
    
    st.subheader("Advanced Features")
    enable_web_fallback = st.toggle("🌐 Enable Web Search Fallback", value=True, help="If the models don't know the answer, they will search DuckDuckGo.")
    
    st.write("---")
    if st.button("🗑️ Clear Vector Database (ChromaDB)"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.success("Database cleared! Please upload data again on your next evaluation.")

st.write("### 1. Data Context (RAG)")
input_method = st.radio("Choose Input Method:", ["Upload PDF Document", "Paste Text"])

uploaded_pdf = None
sample_text = ""

if input_method == "Upload PDF Document":
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
else:
    sample_text = st.text_area(
        "Enter the text to be used as your knowledge base:",
        height=150,
        value="Artificial Intelligence (AI) is evolving rapidly. Llama 3 was created by Meta and GPT-4 by OpenAI."
    )

st.write("### 2. Query")
question = st.text_input("Ask a question:", value="What is Artificial Intelligence?")

if st.button("🚀 Start Evaluation", type="primary"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Groq API Key is missing! Please add it in the sidebar.")
        st.stop()
        
    if input_method == "Upload PDF Document" and not uploaded_pdf:
        st.warning("Please upload a PDF first or use the Text mode.")
        st.stop()
        
    with st.spinner("Processing data, generating embeddings, and building Vector DB (ChromaDB)..."):
        try:
            rag_a = RAGPipelineManager(model_name=model_a)
            rag_b = RAGPipelineManager(model_name=model_b)
            
            db_populated = os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0
            
            # If the database is empty or the user provides fresh text, index it
            if not db_populated:
                if input_method == "Upload PDF Document":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_pdf.getvalue())
                        tmp_path = tmp.name
                    rag_a.index_pdf(tmp_path)
                    os.unlink(tmp_path)
                else:
                    texts = [t.strip() for t in sample_text.split(".") if t.strip()]
                    rag_a.index_text(texts)
            
            evaluator = RagasEvaluator()
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")
            st.stop()

    st.success("✅ Data loaded into ChromaDB! Running LLM models...")
    
    col1, col2 = st.columns(2)
    
    # Model A Run
    with col1:
        st.subheader(f"Model A: {model_a}")
        with st.spinner("Generating answer..."):
            res_a = rag_a.run_query(question, use_web_fallback=enable_web_fallback)
        st.info(f"**Answer:**\n\n{res_a['answer']}")
        with st.spinner("Running Ragas Evaluation..."):
            eval_a = evaluator.evaluate(question, res_a["answer"], res_a["contexts"])
        st.json(eval_a)
            
    # Model B Run
    with col2:
        st.subheader(f"Model B: {model_b}")
        with st.spinner("Generating answer..."):
            res_b = rag_b.run_query(question, use_web_fallback=enable_web_fallback)
        st.info(f"**Answer:**\n\n{res_b['answer']}")
        with st.spinner("Running Ragas Evaluation..."):
            eval_b = evaluator.evaluate(question, res_b["answer"], res_b["contexts"])
        st.json(eval_b)
        
    # Plotting comparison
    st.write("---")
    st.write("### 📊 Evaluation Comparison (Ragas Scores)")
    
    df = pd.DataFrame({
        "Model": [model_a, model_b],
        "Faithfulness": [eval_a.get("faithfulness", 0), eval_b.get("faithfulness", 0)],
        "Answer Relevancy": [eval_a.get("answer_relevancy", 0), eval_b.get("answer_relevancy", 0)]
    })
    
    fig = px.bar(
        df, 
        x="Model", 
        y=["Faithfulness", "Answer Relevancy"], 
        barmode="group",
        title="Ragas Scores by Model",
        range_y=[0, 1],
        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export Report
    st.write("### 📥 Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Ragas Report (CSV)",
        data=csv,
        file_name='rageval_report.csv',
        mime='text/csv',
    )
