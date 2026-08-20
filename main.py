from pathlib import Path
import re

import torch
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# =========================================================
# SETTINGS
# =========================================================

# FLAN-T5 is the pretrained instruction-following model used for the parts
# of the patient record that benefit from AI-generated summarization.
MODEL_NAME = "google/flan-t5-base"

# These are the section headings the program expects to find in the
# structured patient-record files. They are used to separate one large
# record into smaller categories before processing.
SECTION_NAMES = [
    "ALLERGIES",
    "MEDICATIONS",
    "CONDITIONS",
    "CARE PLANS",
    "REPORTS",
    "OBSERVATIONS",
    "PROCEDURES",
    "IMMUNIZATIONS",
    "ENCOUNTERS",
    "IMAGING STUDIES",
]


# =========================================================
# LOAD AI MODEL
# =========================================================

# Loads the tokenizer and pretrained FLAN-T5 model, then wraps them in a
# Transformers pipeline. CUDA is used when available; otherwise the model
# runs on the CPU through PyTorch.
def create_summarizer():
    device = 0 if torch.cuda.is_available() else -1

    print("Using GPU." if device == 0 else "Using CPU.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device,
    )


# =========================================================
# READ PDF
# =========================================================

# Opens a PDF and extracts selectable text from every page. The extracted
# page text is combined into one string so it can be processed like a .txt file.
def read_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)
        else:
            print(
                f"Warning: Could not extract text "
                f"from page {page_number}."
            )

    return "\n".join(pages)


# =========================================================
# BASIC CLEANING
# =========================================================

# Removes extra spaces, tabs, and leading/trailing whitespace from one line.
def clean_line(line):
    return " ".join(line.strip().split())


# Detects divider lines made only from characters such as ----- or =====
# so they can be ignored while parsing the record.
def is_separator(line):
    line = line.strip()

    if not line:
        return False

    return set(line) <= {"-", "=", "_"}


# =========================================================
# SPLIT RECORD INTO SECTIONS
# =========================================================

# Reads the record line by line and sorts its contents into sections such as
# medications, conditions, reports, observations, procedures, and encounters.
def split_into_sections(text):
    sections = {
        "PATIENT INFORMATION": []
    }

    current_section = "PATIENT INFORMATION"

    for line in text.splitlines():

        cleaned = line.strip()

        if is_separator(cleaned):
            continue

        possible_heading = cleaned.upper().rstrip(":")

        if possible_heading in SECTION_NAMES:

            current_section = possible_heading

            if current_section not in sections:
                sections[current_section] = []

        else:
            sections[current_section].append(line)

    final_sections = {}

    for name, lines in sections.items():

        section_text = "\n".join(lines).strip()

        if section_text:
            final_sections[name] = section_text

    return final_sections


# =========================================================
# PATIENT INFORMATION
# =========================================================

# Searches the patient-information section for a labeled field such as Age,
# Race, Gender, or Birth Date and returns the value after the colon.
def get_field(text, field_name):
    pattern = rf"^{re.escape(field_name)}:\s*(.+)$"

    for line in text.splitlines():

        match = re.match(
            pattern,
            line.strip(),
            re.IGNORECASE
        )

        if match:
            return clean_line(match.group(1))

    return None


# Converts the demographic portion of the record into a short readable
# background sentence while preserving the demographic values from the file.
def summarize_patient_information(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    # First line in these files is usually the patient name.
    name = lines[0] if lines else None

    race = get_field(text, "Race")
    ethnicity = get_field(text, "Ethnicity")
    gender = get_field(text, "Gender")
    age = get_field(text, "Age")
    birth_date = get_field(text, "Birth Date")

    gender_word = {
        "M": "male",
        "F": "female",
    }.get(gender, gender)

    parts = []

    if name:
        parts.append(name)

    description = []

    if age:
        description.append(f"{age}-year-old")

    if gender_word:
        description.append(gender_word)

    if race:
        description.append(race)

    if ethnicity:
        description.append(ethnicity)

    if description:
        parts.append(", ".join(description))

    if birth_date:
        parts.append(f"born {birth_date}")

    if not parts:
        return "Patient demographic information unavailable."

    return ". ".join(parts) + "."


# =========================================================
# ALLERGIES
# =========================================================

# Cleans the allergy section and preserves it directly instead of asking
# the AI to rewrite information that is already short and important.
def summarize_allergies(text):
    cleaned = " ".join(
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    )

    if not cleaned:
        return "No allergy information documented."

    if "no known allergies" in cleaned.lower():
        return "No known allergies."

    return cleaned


# =========================================================
# MEDICATIONS
# =========================================================

# Removes the date, CURRENT/STOPPED status, and diagnosis description from
# a medication entry so only the medication name and dose remain.
def extract_medication_name(line):
    # Remove date/status information from beginning.
    line = re.sub(
        r"^\d{4}-\d{2}-\d{2}\s*\[[^\]]+\]\s*:\s*",
        "",
        line
    )

    # Remove diagnosis explanation after "for".
    line = re.split(
        r"\s+for\s+",
        line,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    return clean_line(line)


# Separates current medications from stopped medications, removes duplicates,
# and converts the medication list into a short readable summary.
def summarize_medications(text):
    current = []
    stopped = []

    for line in text.splitlines():

        cleaned = clean_line(line)

        if not cleaned:
            continue

        medication = extract_medication_name(cleaned)

        if "[CURRENT]" in cleaned.upper():

            if medication not in current:
                current.append(medication)

        elif "[STOPPED]" in cleaned.upper():

            if medication not in stopped:
                stopped.append(medication)

    sentences = []

    if current:
        sentences.append(
            "Current medications include "
            + ", ".join(current[:5])
            + "."
        )
    else:
        sentences.append(
            "No current medications are documented."
        )

    if stopped:
        sentences.append(
            "Previously documented medications include "
            + ", ".join(stopped[:5])
            + "."
        )

    return " ".join(sentences)


# =========================================================
# CONDITIONS
# =========================================================

# Removes the date range from a condition entry so the program can work with
# the condition name itself instead of the entire raw line.
def extract_condition_name(line):
    if ":" not in line:
        return None

    condition = line.split(":", 1)[1].strip()

    condition = re.sub(
        r"\s+\d{4}-\d{2}-\d{2}.*$",
        "",
        condition
    )

    return clean_line(condition)


# Collects unique documented conditions and counts repeated medication-review
# entries so the final output is shorter than the original condition history.
def summarize_conditions(text):
    conditions = []

    medication_reviews = 0

    for line in text.splitlines():

        cleaned = clean_line(line)

        if not cleaned:
            continue

        condition = extract_condition_name(cleaned)

        if not condition:
            continue

        if "medication review due" in condition.lower():
            medication_reviews += 1
            continue

        if condition not in conditions:
            conditions.append(condition)

    sentences = []

    if conditions:
        sentences.append(
            "Documented conditions include "
            + ", ".join(conditions[:6])
            + "."
        )

    if medication_reviews > 0:
        sentences.append(
            "The record also contains multiple medication review entries."
        )

    if not sentences:
        return "No significant conditions were identified."

    return " ".join(sentences)


# =========================================================
# REPORTS / LABS
# =========================================================

# Searches a group of lab-report lines for the first line containing a
# requested term such as Hemoglobin, Glucose, Platelets, or Creatinine.
def find_value(lines, search_term):
    for line in lines:

        if search_term.lower() in line.lower():
            return clean_line(line)

    return None


# Performs a small cleanup on a lab-value line before it is shown in the summary.
def get_value_only(line):
    if not line:
        return None

    # Remove leading dash
    line = line.strip().lstrip("-").strip()

    return line


# Finds the newest dated laboratory report and selects several useful values
# from that report instead of printing the entire laboratory history.
def summarize_reports(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return "No laboratory reports documented."

    # First dated line is the newest report in these records.
    date = None
    report_name = None
    start_index = None

    for index, line in enumerate(lines):

        match = re.match(
            r"(\d{4}-\d{2}-\d{2})\s*:\s*(.+)",
            line
        )

        if match:
            date = match.group(1)
            report_name = match.group(2)
            start_index = index
            break

    if start_index is None:
        return "Laboratory information is available in the record."

    report_lines = []

    for line in lines[start_index + 1:]:

        if re.match(
            r"\d{4}-\d{2}-\d{2}\s*:",
            line
        ):
            break

        report_lines.append(line)

    useful_terms = [
        "Leukocytes",
        "Hemoglobin",
        "Hematocrit",
        "Platelets",
        "Glucose",
        "Creatinine",
    ]

    values = []

    for term in useful_terms:

        value = find_value(
            report_lines,
            term
        )

        if value:

            value = get_value_only(value)

            if value not in values:
                values.append(value)

    sentence = ""

    if date and report_name:
        sentence = (
            f"The most recent report was a "
            f"{report_name} dated {date}."
        )

    if values:
        sentence += (
            " Key documented values include "
            + "; ".join(values[:5])
            + "."
        )

    return sentence.strip()


# =========================================================
# OBSERVATIONS / VITALS
# =========================================================

# Finds the first occurrence of a requested observation. Because the source
# records are ordered newest-to-oldest, this represents the latest value.
def find_latest_line(lines, phrase):
    for line in lines:

        if phrase.lower() in line.lower():
            return clean_line(line)

    return None


# Removes the observation label from a line so only the measurement/value
# remains for the final readable vital-sign sentence.
def value_after_phrase(line, phrase):
    if not line:
        return None

    lower = line.lower()
    location = lower.find(
        phrase.lower()
    )

    if location == -1:
        return line

    value = line[
        location + len(phrase):
    ].strip()

    return value


# Pulls the most recent vital signs and measurements, including blood pressure,
# heart rate, respiratory rate, BMI, weight, height, and pain score.
def summarize_observations(text):
    lines = text.splitlines()

    respiration = find_latest_line(
        lines,
        "Respiratory rate"
    )

    heart_rate = find_latest_line(
        lines,
        "Heart rate"
    )

    systolic = find_latest_line(
        lines,
        "Systolic Blood Pressure"
    )

    diastolic = find_latest_line(
        lines,
        "Diastolic Blood Pressure"
    )

    bmi = find_latest_line(
        lines,
        "Body mass index (BMI) [Ratio]"
    )

    weight = find_latest_line(
        lines,
        "Body Weight"
    )

    height = find_latest_line(
        lines,
        "Body Height"
    )

    pain = find_latest_line(
        lines,
        "Pain severity"
    )

    values = []

    if systolic and diastolic:

        sys_value = value_after_phrase(
            systolic,
            "Systolic Blood Pressure"
        )

        dia_value = value_after_phrase(
            diastolic,
            "Diastolic Blood Pressure"
        )

        values.append(
            f"blood pressure {sys_value} / {dia_value}"
        )

    if heart_rate:

        values.append(
            "heart rate "
            + value_after_phrase(
                heart_rate,
                "Heart rate"
            )
        )

    if respiration:

        values.append(
            "respiratory rate "
            + value_after_phrase(
                respiration,
                "Respiratory rate"
            )
        )

    if bmi:

        values.append(
            "BMI "
            + value_after_phrase(
                bmi,
                "Body mass index (BMI) [Ratio]"
            )
        )

    if weight:

        values.append(
            "weight "
            + value_after_phrase(
                weight,
                "Body Weight"
            )
        )

    if height:

        values.append(
            "height "
            + value_after_phrase(
                height,
                "Body Height"
            )
        )

    if pain:

        # Get just the reported number if possible.
        match = re.search(
            r"Reported\s+([\d.]+)",
            pain,
            re.IGNORECASE
        )

        if match:
            values.append(
                f"pain score {match.group(1)}/10"
            )

    if not values:
        return "No recent vital signs were identified."

    return (
        "Most recent documented observations include "
        + ", ".join(values)
        + "."
    )


# =========================================================
# PROCEDURES
# =========================================================

# Keeps a small set of the most recent procedures and formats them into one
# concise sentence instead of listing the entire procedure history.
def summarize_procedures(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return "No procedures documented."

    procedure_names = []

    for line in lines[:6]:

        if ":" in line:
            procedure = line.split(
                ":",
                1
            )[1].strip()
        else:
            procedure = line

        if procedure not in procedure_names:
            procedure_names.append(procedure)

    return (
        "Recent procedures include "
        + "; ".join(procedure_names[:4])
        + "."
    )


# =========================================================
# IMMUNIZATIONS
# =========================================================

# Extracts unique recent vaccine names from the immunization history and
# combines them into a shorter readable sentence.
def summarize_immunizations(text):
    vaccines = []

    for line in text.splitlines():

        cleaned = clean_line(line)

        if not cleaned:
            continue

        if ":" in cleaned:
            vaccine = cleaned.split(
                ":",
                1
            )[1].strip()
        else:
            vaccine = cleaned

        if vaccine not in vaccines:
            vaccines.append(vaccine)

    if not vaccines:
        return "No immunizations documented."

    return (
        "Recent documented immunizations include "
        + ", ".join(vaccines[:5])
        + "."
    )


# =========================================================
# ENCOUNTERS
# =========================================================

# Examines recent encounter lines, identifies the latest date, and extracts
# recent encounter reasons when they are present in the source record.
def summarize_encounters(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return "No encounters documented."

    dates = []

    reasons = []

    for line in lines[:5]:

        date_match = re.match(
            r"(\d{4}-\d{2}-\d{2})",
            line
        )

        if date_match:
            dates.append(
                date_match.group(1)
            )

        encounter_match = re.search(
            r"Encounter for (.+)$",
            line,
            re.IGNORECASE
        )

        if encounter_match:

            reason = clean_line(
                encounter_match.group(1)
            )

            if reason not in reasons:
                reasons.append(reason)

    result = ""

    if dates:
        result += (
            f"The most recent documented encounter "
            f"occurred on {dates[0]}."
        )

    if reasons:
        result += (
            " Recent encounter reasons include "
            + ", ".join(reasons[:3])
            + "."
        )
    else:
        result += (
            " The record shows regular outpatient "
            "follow-up encounters."
        )

    return result.strip()


# =========================================================
# OPTIONAL AI SUMMARY FOR NARRATIVE SECTIONS
# =========================================================

# Sends narrative text to FLAN-T5 with instructions to summarize only the
# information present in the record and avoid inventing additional facts.
def summarize_with_ai(
    summarizer,
    section_name,
    text
):
    prompt = f"""
Summarize the following section from a patient medical record.

Section: {section_name}

Use only facts present in the record.
Do not invent information.
Do not mention research studies or clinical trials.
Write no more than two short sentences.

Record:
{text}

Summary:
"""

    result = summarizer(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
    )

    return result[0]["generated_text"].strip()


# =========================================================
# CARE PLANS
# =========================================================

# Care plans contain more narrative language than the structured sections,
# so this function routes them through the AI summarization function.
def summarize_care_plans(
    summarizer,
    text
):
    return summarize_with_ai(
        summarizer,
        "CARE PLANS",
        text
    )


# =========================================================
# IMAGING
# =========================================================

# Preserves a small number of recent imaging-study entries in a concise format.
def summarize_imaging(text):
    lines = [
        clean_line(line)
        for line in text.splitlines()
        if clean_line(line)
    ]

    if not lines:
        return "No imaging studies documented."

    return (
        "Documented imaging includes "
        + "; ".join(lines[:3])
        + "."
    )


# =========================================================
# CREATE FINAL NARRATIVE SUMMARY
# =========================================================

# Coordinates the entire summarization process. It splits the record into
# sections, sends each section to the appropriate processing function, and
# combines the results into the final patient-summary format.
def create_patient_summary(
    summarizer,
    text
):
    sections = split_into_sections(
        text
    )

    print("\nDetected sections:")

    for name in sections:
        print(f" - {name}")

    print()

    output = []

    if "PATIENT INFORMATION" in sections:

        output.append(
            "BACKGROUND:\n"
            + summarize_patient_information(
                sections[
                    "PATIENT INFORMATION"
                ]
            )
        )

    if "ALLERGIES" in sections:

        output.append(
            "ALLERGIES:\n"
            + summarize_allergies(
                sections["ALLERGIES"]
            )
        )

    if "MEDICATIONS" in sections:

        output.append(
            "MEDICATIONS:\n"
            + summarize_medications(
                sections["MEDICATIONS"]
            )
        )

    if "CONDITIONS" in sections:

        output.append(
            "CONDITIONS:\n"
            + summarize_conditions(
                sections["CONDITIONS"]
            )
        )

    if "REPORTS" in sections:

        output.append(
            "RECENT LABS:\n"
            + summarize_reports(
                sections["REPORTS"]
            )
        )

    if "OBSERVATIONS" in sections:

        output.append(
            "RECENT VITALS:\n"
            + summarize_observations(
                sections["OBSERVATIONS"]
            )
        )

    if "CARE PLANS" in sections:

        print(
            "Summarizing care plan with AI..."
        )

        output.append(
            "CARE PLAN:\n"
            + summarize_care_plans(
                summarizer,
                sections["CARE PLANS"]
            )
        )

    if "PROCEDURES" in sections:

        output.append(
            "RECENT PROCEDURES:\n"
            + summarize_procedures(
                sections["PROCEDURES"]
            )
        )

    if "IMMUNIZATIONS" in sections:

        output.append(
            "IMMUNIZATIONS:\n"
            + summarize_immunizations(
                sections["IMMUNIZATIONS"]
            )
        )

    if "ENCOUNTERS" in sections:

        output.append(
            "RECENT ENCOUNTERS:\n"
            + summarize_encounters(
                sections["ENCOUNTERS"]
            )
        )

    if "IMAGING STUDIES" in sections:

        output.append(
            "IMAGING:\n"
            + summarize_imaging(
                sections[
                    "IMAGING STUDIES"
                ]
            )
        )

    return "\n\n".join(output)


# =========================================================
# MAIN
# =========================================================

# Main program entry point. It locates the input folder, finds supported TXT
# and PDF files, loads the AI model once, processes each file, and prints
# the generated patient summary to the terminal.
def main():
    base_dir = Path(
        __file__
    ).resolve().parent

    input_dir = (
        base_dir / "input"
    )

    if not input_dir.exists():

        print(
            "Input directory not found."
        )

        return

    text_files = sorted(
        input_dir.glob("*.txt")
    )

    pdf_files = sorted(
        input_dir.glob("*.pdf")
    )

    input_files = (
        text_files
        + pdf_files
    )

    if not input_files:

        print(
            "No TXT or PDF files found "
            "in the input directory."
        )

        return

    print(
        "Loading AI model..."
    )

    summarizer = create_summarizer()

    print(
        "Model loaded successfully.\n"
    )

    for file_path in input_files:

        print("=" * 70)

        print(
            f"Processing: "
            f"{file_path.name}"
        )

        print("=" * 70)

        if (
            file_path.suffix.lower()
            == ".txt"
        ):

            with file_path.open(
                "r",
                encoding="utf-8"
            ) as handle:

                text = handle.read()

        elif (
            file_path.suffix.lower()
            == ".pdf"
        ):

            text = read_pdf(
                file_path
            )

        else:

            continue

        if not text.strip():

            print(
                "No readable text found."
            )

            continue

        summary = create_patient_summary(
            summarizer,
            text
        )

        print("\n")
        print("=" * 70)

        print(
            "GENERATED PATIENT SUMMARY"
        )

        print("=" * 70)

        print(summary)

        print()


if __name__ == "__main__":
    main()