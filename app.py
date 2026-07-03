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
st.markdown("Αξιολόγησε διαφορετικά LLM Models (μέσω Groq) στα δικά σου δεδομένα με βάση τις μετρικές **Ragas** (Faithfulness, Answer Relevancy). Όλα τρέχουν **Δωρεάν** χρησιμοποιώντας local embeddings (HuggingFace) και Groq LPUs!")

# Sidebar for settings
with st.sidebar:
    st.header("Ρυθμίσεις ⚙️")
    groq_key = st.text_input("Groq API Key (Προαιρετικό αν υπάρχει στο .env)", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        
    st.subheader("Σύγκριση Μοντέλων (LLM)")
    model_a = st.selectbox("Μοντέλο A", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=0)
    model_b = st.selectbox("Μοντέλο B", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], index=2)

st.write("### 1. Προσθήκη Δεδομένων (Context)")
sample_text = st.text_area(
    "Εισάγετε το κείμενο που θα χρησιμοποιηθεί ως βάση δεδομένων (RAG Data):",
    height=200,
    value="Η Τεχνητή Νοημοσύνη (AI) εξελίσσεται ταχύτατα. Το 2024, τα Large Language Models όπως το GPT-4 και το Llama 3 κυριάρχησαν στην αγορά. Το Llama 3 δημιουργήθηκε από την Meta και είναι open-source, ενώ το GPT-4 ανήκει στην OpenAI. Τα τοπικά embeddings βοηθούν στην ιδιωτικότητα των δεδομένων διότι τρέχουν απευθείας στον υπολογιστή του χρήστη και δεν χρειάζονται κανένα API."
)

st.write("### 2. Ερώτηση")
question = st.text_input("Ρώτησε κάτι σχετικά με το κείμενο:", value="Ποια εταιρεία έφτιαξε το Llama 3 και τι πλεονεκτήματα έχουν τα τοπικά embeddings;")

if st.button("🚀 Έναρξη Αξιολόγησης", type="primary"):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Το Groq API Key λείπει! Βάλε το στο sidebar ή στο .env αρχείο.")
        st.stop()
        
    with st.spinner("Επεξεργασία δεδομένων, δημιουργία embeddings και Vector Database..."):
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
            st.error(f"Σφάλμα κατά την προετοιμασία: {str(e)}")
            st.stop()

    st.success("✅ Δεδομένα φορτώθηκαν! Τρέχουμε τα LLM μοντέλα...")
    
    col1, col2 = st.columns(2)
    
    # Model A Run
    with col1:
        st.subheader(f"Μοντέλο A: {model_a}")
        with st.spinner("Παραγωγή απάντησης..."):
            res_a = rag_a.run_query(question)
        st.info(f"**Απάντηση:**\n\n{res_a['answer']}")
        with st.spinner("Αξιολόγηση Ragas..."):
            eval_a = evaluator.evaluate(question, res_a["answer"], res_a["contexts"])
        st.json(eval_a)
            
    # Model B Run
    with col2:
        st.subheader(f"Μοντέλο B: {model_b}")
        with st.spinner("Παραγωγή απάντησης..."):
            res_b = rag_b.run_query(question)
        st.info(f"**Απάντηση:**\n\n{res_b['answer']}")
        with st.spinner("Αξιολόγηση Ragas..."):
            eval_b = evaluator.evaluate(question, res_b["answer"], res_b["contexts"])
        st.json(eval_b)
        
    # Plotting comparison
    st.write("---")
    st.write("### 📊 Σύγκριση Αποδοτικότητας (Ragas Scores)")
    st.caption("Το 1.0 είναι το άριστα. **Faithfulness**: Κατά πόσο η απάντηση στηρίζεται αποκλειστικά στα δεδομένα (απουσία παραισθήσεων). **Answer Relevancy**: Κατά πόσο η απάντηση σχετίζεται όντως με την ερώτηση.")
    
    df = pd.DataFrame({
        "Μοντέλο": [model_a, model_b],
        "Faithfulness": [eval_a.get("faithfulness", 0), eval_b.get("faithfulness", 0)],
        "Answer Relevancy": [eval_a.get("answer_relevancy", 0), eval_b.get("answer_relevancy", 0)]
    })
    
    fig = px.bar(
        df, 
        x="Μοντέλο", 
        y=["Faithfulness", "Answer Relevancy"], 
        barmode="group",
        title="Βαθμολογίες Ragas ανά Μοντέλο",
        range_y=[0, 1],
        color_discrete_sequence=["#1f77b4", "#ff7f0e"]
    )
    st.plotly_chart(fig, use_container_width=True)
