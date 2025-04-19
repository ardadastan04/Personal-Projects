# PDF Question & Answer System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content. The system uses semantic search and Google's Gemini AI to provide accurate answers based on the document content.

## Features

- PDF document upload and processing
- Semantic search for relevant content
- Natural language question answering
- Chat-like interface for interaction
- Context-aware responses

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `keys.py` file with your Google API key:
```python
API_KEY = "your-api-key-here"
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload a PDF document using the file uploader
2. Wait for the document to be processed
3. Ask questions about the document content
4. View the AI-generated answers
5. Clear chat or upload a new document as needed

## Technologies Used

- Streamlit for the web interface
- PyMuPDF for PDF processing
- Sentence Transformers for text embeddings
- FAISS for semantic search
- Google's Gemini AI for question answering

## License

MIT License 