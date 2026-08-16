from pathlib import Path
from pypdf import PdfReader
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def create_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    model_name = "hossboll/clinical-t5"


    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=14096) #THIS FIXES THE MAX TOKEN ISSUE
    #but if it is set too high, your computer will run out of memory.

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device,
    )


def summarize_text(summarizer, text):
    if len(text.split()) < 30:
        return "The file does not contain enough text to summarize."

    result = summarizer(
        text,
        max_length=2048,
        max_new_tokens=2048, #max length and max new tokens must be the same
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

    if not input_files:
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