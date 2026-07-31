from pathlib import Path


def main():
    input_dir = Path(__file__).resolve().parent
    text_files = sorted(input_dir.glob("*.txt"))

    if not text_files:
        print("No text files found in the input directory.")
        return

    for file_path in text_files:
        print(f"--- {file_path.name} ---")
        with file_path.open("r", encoding="utf-8") as handle:
            print(handle.read())


if __name__ == "__main__":
    main()
