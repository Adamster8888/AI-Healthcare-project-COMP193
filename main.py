from pathlib import Path

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
        max_length=100,
        min_length=25,
        do_sample=False,
    )

    return result[0]["summary_text"]


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"

    if not input_dir.exists():
        print("Input directory not found.")
        return

    text_files = sorted(input_dir.glob("*.txt"))

    if not text_files:
        print("No text files found in the input directory.")
        return

    print("Loading summarization model...")
    summarizer = create_summarizer()
    print("Model loaded successfully.\n")

    for file_path in text_files:
        print(f"--- {file_path.name} ---")

        with file_path.open("r", encoding="utf-8") as handle:
            text = handle.read()

        summary = summarize_text(summarizer, text)

        print("Original text:")
        print(text)

        print("\nGenerated summary:")
        print(summary)
        print()


if __name__ == "__main__":
    main()