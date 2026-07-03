import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px

from core.rag_pipeline import RAGPipelineManager
from core.evaluator import RagasEvaluator

# Load environment variables (.env file)
load_dotenv()

st.set_page_config(page_title="RAGEval", page_icon="🦑", layout="wide")

st.title("🦑 RAGEval: AI Data Pipeline Evaluator")
st.markdown("Evaluate different LLMs (via Groq) on your own data using **Ragas** metrics (Faithfulness, Answer Relevancy). Everything runs **Free** using local embeddings (HuggingFace) and Groq LPUs!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings ⚙️")
    groq_key = st.text_input("Groq API Key (Optional if in .env)", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    st.subheader("Model Comparison (LLMs)")
    model_a = st.selectbox("Model A", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=0)
    model_b = st.selectbox("Model B", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=2)

st.write("### 1. Data Context (RAG)")
sample_text = st.text_area(
    "Enter the text to be used as your knowledge base:",
    height=200,
    value="Artificial Intelligence (AI) is evolving rapidly. In 2024, Large Language Models like GPT-4 and Llama 3 dominated the market. Llama 3 was created by Meta and is open-source, while GPT-4 belongs to OpenAI. Local embeddings help with data privacy because they run directly on the user's computer and do not require any API."
)

st.write("### 2. Query")
question = st.text_input("Ask a question about the text:", value="Which company created Llama 3 and what are the advantages of local embeddings?")

if st.button("🚀 Start Evaluation", type="primary"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Groq API Key is missing! Please provide it in the sidebar or .env file.")
        st.stop()
        
    with st.spinner("Processing data, generating embeddings, and building Vector DB..."):
        try:
            # Initialize Pipelines
            rag_a = RAGPipelineManager(model_name=model_a)
            rag_b = RAGPipelineManager(model_name=model_b)
            
            # Index data (Very simple text chunking by sentence for the demo)
            texts = [t.strip() for t in sample_text.split(".") if t.strip()]
            rag_a.index_documents(texts)
            rag_b.index_documents(texts)
            
            evaluator = RagasEvaluator()
        except Exception as e:
            st.error(f"Initialization Error: {str(e)}")
            st.stop()

    st.success("✅ Data loaded! Running LLM models...")
    
    col1, col2 = st.columns(2)
    
    # Model A Run
    with col1:
        st.subheader(f"Model A: {model_a}")
        with st.spinner("Generating answer..."):
            res_a = rag_a.run_query(question)
        st.info(f"**Answer:**\n\n{res_a['answer']}")
        with st.spinner("Running Ragas Evaluation..."):
            eval_a = evaluator.evaluate(question, res_a["answer"], res_a["contexts"])
        st.json(eval_a)
            
    # Model B Run
    with col2:
        st.subheader(f"Model B: {model_b}")
        with st.spinner("Generating answer..."):
            res_b = rag_b.run_query(question)
        st.info(f"**Answer:**\n\n{res_b['answer']}")
        with st.spinner("Running Ragas Evaluation..."):
            eval_b = evaluator.evaluate(question, res_b["answer"], res_b["contexts"])
        st.json(eval_b)
        
    # Plotting comparison
    st.write("---")
    st.write("### 📊 Evaluation Comparison (Ragas Scores)")
    st.caption("Score 1.0 is perfect. **Faithfulness**: How much the answer is grounded in the given context (absence of hallucinations). **Answer Relevancy**: How well the answer addresses the question.")
    
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
