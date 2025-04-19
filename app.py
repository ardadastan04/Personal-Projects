# Import libraries
from annoy import AnnoyIndex
import langchain
import fitz  # pymupdf is imported as fitz
from sentence_transformers import SentenceTransformer
import streamlit as st
from google import genai
from keys import API_KEY
import tempfile

# Initialize session state to store document processing results
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_pdf(uploaded_file):
    # Create a temporary file to store the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    # Process PDF
    pdf_document = fitz.open(pdf_path)
    text_content = ""
    for page in pdf_document:
        text_content += page.get_text()
    
    # Split text into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(text_content)
    
    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    
    # Create Annoy index
    dimension = embeddings.shape[1]
    index = AnnoyIndex(dimension, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)  # 10 trees
    
    return text_chunks, index, model

def process_question(question, index, model, text_chunks):
    # Get relevant context
    question_embedding = model.encode([question])[0]
    indices = index.get_nns_by_vector(question_embedding, 5)
    relevant_chunks = [text_chunks[i] for i in indices]
    
    # Combine relevant chunks into context
    context = "\n\n".join(relevant_chunks)
    
    # Create prompt
    prompt = f"""Based on the following text, answer the question. If the answer cannot be found in the text, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        # Initialize Gemini
        client = genai.Client(api_key=API_KEY)
        
        # Generate answer
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error in generating answer: {str(e)}"

def main():
    st.title("PDF Question & Answer System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file and not st.session_state.processed:
        with st.spinner('Processing PDF...'):
            st.session_state.text_chunks, st.session_state.index, st.session_state.model = process_pdf(uploaded_file)
            st.session_state.processed = True
            st.success('PDF processed successfully!')
    
    # Chat interface
    if st.session_state.processed:
        # Display chat history
        for q, a in st.session_state.chat_history:
            with st.container():
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
        
        # Question input
        question = st.text_input("Ask a question about your document:", key="question_input")
        
        if st.button("Ask"):
            if question:
                with st.spinner('Generating answer...'):
                    answer = process_question(
                        question,
                        st.session_state.index,
                        st.session_state.model,
                        st.session_state.text_chunks
                    )
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    # Clear input
                    st.rerun()
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # New document button
        if st.button("Process New Document"):
            st.session_state.processed = False
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
