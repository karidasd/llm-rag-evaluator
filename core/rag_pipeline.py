import os
from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

class RAGPipelineManager:
    def __init__(self, model_name: str = "llama3-8b-8192", groq_api_key: str = None):
        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API Key is missing!")
            
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.document_store = InMemoryDocumentStore()
        
        # Initialize Embedders (Local)
        # Using a fast, local embedding model
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
        self.text_embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
        
        # Initialize Retriever
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        
        # Initialize Prompt Builder
        template = """
        Answer the question based ONLY on the following context documents.
        If the answer is not contained in the documents, just say "I don't know".
        
        Context Documents:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}
        
        Question: {{question}}
        Answer:
        """
        self.prompt_builder = PromptBuilder(template=template)
        
        # Initialize Generator (Groq via OpenAI compatible API format)
        self.generator = OpenAIGenerator(
            api_key=Secret.from_token(self.groq_api_key),
            api_base_url="https://api.groq.com/openai/v1",
            model=self.model_name
        )
        
    def index_documents(self, texts: list[str]):
        """Creates documents, generates embeddings, and saves them to vector store."""
        docs = [Document(content=text) for text in texts]
        self.doc_embedder.warm_up()
        docs_with_embeddings = self.doc_embedder.run(docs)["documents"]
        self.document_store.write_documents(docs_with_embeddings)
        
    def run_query(self, question: str):
        """Runs the RAG pipeline."""
        pipeline = Pipeline()
        pipeline.add_component("text_embedder", self.text_embedder)
        pipeline.add_component("retriever", self.retriever)
        pipeline.add_component("prompt_builder", self.prompt_builder)
        pipeline.add_component("llm", self.generator)
        
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")
        
        self.text_embedder.warm_up()
        results = pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        
        answer = results["llm"]["replies"][0]
        retrieved_docs = results["retriever"]["documents"] # we need context for Ragas
        
        return {
            "answer": answer,
            "contexts": [doc.content for doc in retrieved_docs]
        }
