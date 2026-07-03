# 🦑 RAGEval: AI Data Pipeline Evaluator

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Haystack](https://img.shields.io/badge/Haystack-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

Το **RAGEval** είναι ένα open-source εργαλείο (DevTool) γραμμένο σε Python, σχεδιασμένο για Data Scientists και Developers. Σας επιτρέπει να αξιολογείτε και να συγκρίνετε διαφορετικά Large Language Models (LLMs) και στρατηγικές RAG (Retrieval-Augmented Generation) πάνω στα δικά σας δεδομένα.

## ✨ Χαρακτηριστικά
- 🚀 **Αστραπιαία Inference:** Χρήση του [Groq](https://groq.com/) API για κλήση μοντέλων όπως το Llama 3 και το Mixtral σε κλάσματα δευτερολέπτου.
- 🔒 **Privacy-First (Τοπικά Embeddings):** Χρησιμοποιεί τοπικά μοντέλα από το HuggingFace (`all-MiniLM-L6-v2`) για τα embeddings, διασφαλίζοντας ότι τα δεδομένα σας (έγγραφα) παραμένουν στον υπολογιστή σας χωρίς περιττά κόστη API.
- 📐 **Αξιολόγηση με Ragas:** Ενσωματωμένο το [Ragas Framework](https://docs.ragas.io/) για να μετράει το `Faithfulness` (απουσία παραισθήσεων) και το `Answer Relevancy` (κατά πόσο απάντησε σωστά στην ερώτηση).
- 🧩 **Modular Αρχιτεκτονική:** Χτισμένο πάνω στο [Haystack 2.x](https://haystack.deepset.ai/) για εύκολη προσθήκη διαφορετικών Vector Databases στο μέλλον.
- 📊 **Beautiful Web UI:** Γραφικό περιβάλλον με [Streamlit](https://streamlit.io/) και διαδραστικά γραφήματα Plotly.

## 🛠️ Τεχνολογίες (Tech Stack)
*   **Orchestration:** Haystack AI
*   **LLM Provider:** Groq
*   **Evaluation:** Ragas, LangChain
*   **UI / Visualization:** Streamlit, Plotly, Pandas
*   **Embeddings:** Sentence-Transformers (HuggingFace)

## 🚀 Εγκατάσταση και Χρήση

### 1. Κλωνοποίηση του Repository
```bash
git clone https://github.com/your-username/rageval.git
cd rageval
```

### 2. Εγκατάσταση Εξαρτήσεων (Dependencies)
Προτείνεται η χρήση ενός εικονικού περιβάλλοντος (virtual environment):
```bash
python -m venv venv
# Για Windows:
venv\Scripts\activate
# Για Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Ρύθμιση API Keys
Για να λειτουργήσει το εργαλείο δωρεάν με τα κορυφαία μοντέλα, θα χρειαστείτε ένα κλειδί από το Groq.
1. Πάρτε ένα δωρεάν API key από το [console.groq.com](https://console.groq.com/keys)
2. Αντιγράψτε το αρχείο `.env.example` και μετονομάστε το σε `.env`
3. Τοποθετήστε το κλειδί σας μέσα στο `.env`:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxx
```
*(Εναλλακτικά, μπορείτε να βάλετε το κλειδί απευθείας μέσα από το UI της εφαρμογής).*

### 4. Εκτέλεση Εφαρμογής
```bash
streamlit run app.py
```
Η εφαρμογή θα ανοίξει αυτόματα στον browser σας στη διεύθυνση `http://localhost:8501`.

## 🤝 Συνεισφορά (Contributing)
Είστε ευπρόσδεκτοι να ανοίξετε Issues ή Pull Requests για βελτιώσεις, προσθήκη νέων metrics (π.χ. Context Precision) ή νέων Vector Databases (π.χ. ChromaDB, Qdrant).

## 📄 Άδεια (License)
Διανέμεται υπό την άδεια MIT.
