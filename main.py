from pathlib import Path
from pypdf import PdfReader
import torch
from transformers import pipeline


def create_summarizer():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="summarization",
        model="sshleifer/distilbart-cnn-12-6",
        framework="pt",
        device=device,
    )


def summarize_text(summarizer, text):
    if len(text.split()) < 30:
        return "The file does not contain enough text to summarize."

    result = summarizer(
        text,
        max_length=400,
        min_length=250,
        do_sample=False,
    )

    return result[0]["summary_text"]


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"

    if not input_dir.exists():
        print("Input directory not found.")
        return

    text_files = sorted(input_dir.glob("*.txt"))
    pdf_files = sorted(input_dir.glob("*.pdf"))

    input_files = text_files + pdf_files

    if not text_files:
        print("No text files found in the input directory.")
        return

    print("Loading summarization model...")
    summarizer = create_summarizer()
    print("Model loaded successfully.\n")

    for file_path in input_files:
        print(f"--- {file_path.name} ---")

    if file_path.suffix == ".txt":
        with file_path.open("r", encoding="utf-8") as handle:
            text = handle.read()

    elif file_path.suffix == ".pdf":
        text = read_pdf(file_path)

    summary = summarize_text(summarizer, text)

    print("Original text:")
    print(text)

    print("\nGenerated summary:")
    print(summary)


if __name__ == "__main__":
    main()