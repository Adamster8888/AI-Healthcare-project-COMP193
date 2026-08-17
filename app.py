import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="Clinical Text Summarizer", layout="wide")

st.title("📄 Clinical Text Summarizer")
st.markdown("Summarize clinical documents using AI-powered analysis")

@st.cache_resource
def create_summarizer():
    """Load and cache the summarization model"""
    device = 0 if torch.cuda.is_available() else -1
    model_name = "hossboll/clinical-t5"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=14096)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device,
    )

def summarize_text(summarizer, text):
    """Summarize the input text"""
    if len(text.split()) < 30:
        return "The file does not contain enough text to summarize."
    
    result = summarizer(
        text,
        max_length=2048,
        max_new_tokens=2048,
        min_length=250,
        do_sample=False,
    )
    
    return result[0]["summary_text"]

def read_pdf(file_path):
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    input_type = st.radio("Choose input type:", ["Text Input", "File Upload"])

# Main content
if input_type == "Text Input":
    st.subheader("Enter Text to Summarize")
    text_input = st.text_area("Paste your text here:", height=300, placeholder="Enter clinical text...")
    
    if st.button("Summarize Text", type="primary"):
        if text_input.strip():
            with st.spinner("Loading model and generating summary..."):
                summarizer = create_summarizer()
                summary = summarize_text(summarizer, text_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Text")
                st.text_area("", value=text_input, height=300, disabled=True)
            
            with col2:
                st.subheader("Summary")
                st.text_area("", value=summary, height=300, disabled=True)
        else:
            st.warning("Please enter some text to summarize.")

else:  # File Upload
    st.subheader("Upload a File to Summarize")
    uploaded_file = st.file_uploader(
        "Choose a text or PDF file",
        type=["txt", "pdf"],
        help="Supported formats: .txt, .pdf"
    )
    
    if uploaded_file is not None:
        if st.button("Summarize File", type="primary"):
            with st.spinner("Loading model and processing file..."):
                # Read file based on type
                if uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                else:  # PDF
                    from io import BytesIO
                    pdf_bytes = BytesIO(uploaded_file.read())
                    reader = PdfReader(pdf_bytes)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                summarizer = create_summarizer()
                summary = summarize_text(summarizer, text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Text")
                st.text_area("", value=text[:2000] + "..." if len(text) > 2000 else text, 
                            height=300, disabled=True)
            
            with col2:
                st.subheader("Summary")
                st.text_area("", value=summary, height=300, disabled=True)
            
            # Option to download summary
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"summary_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )

st.markdown("---")
st.markdown("**Note:** The first run may take a few moments to load the model.")
