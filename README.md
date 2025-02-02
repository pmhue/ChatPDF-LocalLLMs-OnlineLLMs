# Local ChatPDF with DeepSeek R1 / OpenAI API

**ChatPDF** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

This project is inspired by [chatpdf-rag-deepseek-r1](https://github.com/paquino11/chatpdf-rag-deepseek-r1).

## Features

- **PDF Upload**: Upload one or multiple PDF documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

## User Interface

<img src="image/ui.png" alt="User Interface" width="1560"/>
---

## Installation

Follow the steps below to set up and run the application:

### 1. Clone the Repository

```bash
git clone https://github.com/DucTriCE/ChatPDF-Deepseek-OpenAI.git
cd ChatPDF-Deepseek-OpenAI
```

### 2. Install Docker

Make sure Docker and Docker Compose is installed on your system. You can follow the instructions [here](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04) (For Ubuntu 20.04, you can find different tutorial based on your system).

### 3. Build and Run the Docker Container

Build the Docker image and run the container using Docker Compose:

```bash
docker-compose up --build
```
If already built, just run the application using the following command:

```bash
docker-compose up
```

## Usage

### 1. Start the Application

The application will be available at `http://localhost:8501`.

### 2. Choosing Models
You can choose between two models: `deepseek-r1:latest` and `gpt-4o-mini`. If you select the OpenAI model (`gpt-4o-mini`), you will need to specify your OpenAI API key in the provided input field.

### 3. Upload Documents

- Navigate to the **Upload a Document** section in the web interface.
- Upload one or multiple PDF files to process their content.
- Each file will be ingested automatically and confirmation messages will show processing time.

### 4. Ask Questions

- Type your question in the chat input box and press Enter.
- Adjust retrieval settings (`k` and `similarity threshold`) in the **Settings** section for better responses.


## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── rag.py                  # Core RAG logic for PDF ingestion and question-answering
├── requirements.txt        # List of required Python dependencies
├── chroma_db_*/            # Local persistent vector store (auto-generated)
├── Dockerfile              # Dockerfile for building the image
├── docker-compose.yml      # Docker Compose configuration
└── README.md               # Project documentation
```


## Configuration

You can modify the following parameters in `rag.py` to suit your needs:

**Models**:
   - Default Deepseek LLM: `deepseek-r1:latest` (7B parameters)
   - Default Embedding: `mxbai-embed-large` (1024 dimensions)
   - Default OpenAI LLM: `gpt-4o-mini` (8B parameters)
   - Default OpenAI Embedding: `text-embedding-3-large` (3072 dimensions)
   - Any Ollama-compatible model can be used by updating the `llm_model` parameter
---

## Requirements

- **Docker**: For containerization and running the application.
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **PyPDF**: For PDF document processing.
- **ChromaDB**: Vector store for document embeddings.


## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Ollama](https://ollama.ai/)

