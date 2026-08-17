Test.txt:
Created a quick longer fake description that we can use to feed the AI

fake_patient_medical_record.pdf
This is a quick fake pdf to test the ability for the code to read and summarize an inputed file other than a text file

main.py:
Slowly adding in the AI fundamentals I did install some transformers that should help run the AI more smoothly

app.py: identical to main.py as of 8/16. On execution, opens a webpage where you can paste text or upload files.

Update:
Now contains a streamlit app that is identical to main.py, except that it opens a web browser page where you can input files.


MUST READ:
to recreate my enviroment that has running code run this in terminal
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers sentencepiece
python -m pip install pypdf

python -m pip install streamlit
python -m streamlit run app.py

Notes:
If you cant see everything on one line do the following
- Mac (Option + z)
- Windows/Linux (Alt + Z)