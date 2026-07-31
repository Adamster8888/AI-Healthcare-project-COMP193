from pathlib import Path
import torch


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    text_files = sorted(input_dir.glob("*.txt"))

    if not input_dir.exists():
        print("Input directory not found.")
        return

    if not text_files:
        print("No text files found in the input directory.")
        return

    for file_path in text_files:
        print(f"--- {file_path.name} ---")
        with file_path.open("r", encoding="utf-8") as handle:
            print(handle.read())


if __name__ == "__main__":
    main()
