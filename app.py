import streamlit as st
from io import BytesIO

from main import (
    create_patient_summary,
    create_summarizer as load_summarizer,
    read_pdf,
)

st.set_page_config(page_title="Clinical Text Summarizer", layout="wide")

st.title("📄 Clinical Text Summarizer")
st.markdown("Summarize clinical documents using AI-powered analysis")

@st.cache_resource
def create_summarizer():
    """Load and cache the same FLAN-T5 summarizer used by main.py."""
    return load_summarizer()


def read_uploaded_file(uploaded_file):
    """Read an uploaded TXT or PDF using the same PDF extraction logic."""
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8")

    return read_pdf(BytesIO(uploaded_file.getvalue()))


def generate_summary(text):
    """Run the exact section-based summary workflow from main.py."""
    summarizer = create_summarizer()
    return create_patient_summary(summarizer, text)

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
                summary = generate_summary(text_input)
            
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
                text = read_uploaded_file(uploaded_file)
                summary = generate_summary(text)
            
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
st.markdown("The first run may take a few moments to load the model.")
