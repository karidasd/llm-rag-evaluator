import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

class RagasEvaluator:
    def __init__(self, groq_api_key: str = None):
        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API Key is missing for Evaluator!")
            
        # Use Groq as the Judge LLM for Ragas via Langchain OpenAI integration
        # We use llama3-8b for the judge to keep it free and fast
        self.judge_llm = ChatOpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model="llama3-8b-8192"
        )
        
        # Use local embeddings for Ragas Answer Relevancy
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
    def evaluate(self, question: str, answer: str, contexts: list[str]):
        """Evaluates a single Q&A generation."""
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        dataset = Dataset.from_dict(data)
        
        # Pass llm and embeddings explicitly to avoid OpenAI defaults
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.judge_llm,
            embeddings=self.embeddings
        )
        return dict(result)
