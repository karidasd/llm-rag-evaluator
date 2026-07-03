import os
import shutil
from pathlib import Path
from haystack import Document, Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.utils import Secret
from duckduckgo_search import DDGS

class RAGPipelineManager:
    def __init__(self, model_name: str = "llama3-8b-8192", groq_api_key: str = None):
        if not groq_api_key:
            groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Groq API Key is missing!")
            
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.db_path = "./chroma_db"
            
        # Initialize Persistent Vector Store
        self.document_store = ChromaDocumentStore(persist_path=self.db_path)
        
        # Initialize Embedders (Local)
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
        self.text_embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
        
        # Initialize Retriever
        self.retriever = ChromaEmbeddingRetriever(document_store=self.document_store)
        
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
        
        # Initialize Generator
        self.generator = OpenAIGenerator(
            api_key=Secret.from_token(self.groq_api_key),
            api_base_url="https://api.groq.com/openai/v1",
            model=self.model_name
        )
        
    def index_text(self, texts: list[str]):
        """Indexes raw text into ChromaDB."""
        docs = [Document(content=text) for text in texts]
        self._embed_and_write(docs)
        
    def index_pdf(self, pdf_path: str):
        """Extracts text from PDF, chunks it, and indexes it into ChromaDB."""
        converter = PyPDFToDocument()
        results = converter.run(sources=[pdf_path])
        docs = results["documents"]
        
        splitter = DocumentSplitter(split_by="word", split_length=300, split_overlap=30)
        chunked_docs = splitter.run(documents=docs)["documents"]
        
        self._embed_and_write(chunked_docs)
        
    def _embed_and_write(self, docs):
        self.doc_embedder.warm_up()
        docs_with_embeddings = self.doc_embedder.run(documents=docs)["documents"]
        self.document_store.write_documents(docs_with_embeddings)
        
    def run_query(self, question: str, use_web_fallback: bool = True):
        """Runs the RAG pipeline with optional Web Search Fallback."""
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
        retrieved_docs = results["retriever"]["documents"]
        
        # Web Search Fallback logic
        if use_web_fallback and "i don't know" in answer.lower():
            try:
                ddgs = DDGS()
                search_results = list(ddgs.text(question, max_results=3))
                web_contexts = [r['body'] for r in search_results]
                
                # Ask LLM again with Web context
                fallback_template = """
                Answer the question based on the following Web Search results:
                {% for ctx in web_contexts %}
                    - {{ ctx }}
                {% endfor %}
                
                Question: {{question}}
                Answer:
                """
                fallback_prompt = PromptBuilder(template=fallback_template)
                
                fallback_pipeline = Pipeline()
                fallback_pipeline.add_component("prompt_builder", fallback_prompt)
                fallback_pipeline.add_component("llm", self.generator)
                fallback_pipeline.connect("prompt_builder", "llm")
                
                fb_res = fallback_pipeline.run({
                    "prompt_builder": {"question": question, "web_contexts": web_contexts}
                })
                answer = "(🌐 Web Search Fallback) " + fb_res["llm"]["replies"][0]
                
                # Overwrite retrieved_docs with web context for Ragas Evaluation
                retrieved_docs = [Document(content=c) for c in web_contexts]
                
            except Exception as e:
                answer = f"I don't know. (Web search failed: {str(e)})"
        
        return {
            "answer": answer,
            "contexts": [doc.content for doc in retrieved_docs]
        }
