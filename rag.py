# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import logging
import os
import shutil

set_debug(True)
set_verbose(True)

ollama_endpoint = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, is_local: bool = True, openai_api_key: str = "", llm_model: str = "", embedding_model: str = ""):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model_name = llm_model.split("-")[0]
        # self.model = ChatOllama(model=llm_model) if is_local else ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        # self.embeddings = OllamaEmbeddings(model=embedding_model) if is_local else OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")



        self.chunk_size = 1024 if is_local else 3072
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100)# if is_local else None

        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant developed to answer questions only based on the uploaded document. If the question is not related to the document, tell the user that the question is out of scope.
            Context:
            {context}
            
            Question:
            {question}
            
            You only give precise and detailed answer to the question, no yapping. And when giving answer, please start with "Answer:".
            """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """

        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        

        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            collection_name=f"pdf_collection_{self.model_name}",
            documents=chunks,
            embedding=self.embeddings,
            persist_directory = f"chroma_db_{self.model_name}",
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("Please upload a document before asking a question.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        # return chain.invoke(formatted_input)
        return chain.stream(formatted_input)


    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
