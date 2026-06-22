import os
import json
import pdfplumber

from dotenv import load_dotenv
from docx import Document

from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


# ==========================================
# Load Environment Variables
# ==========================================

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in .env file")


# ==========================================
# Initialize Mistral
# ==========================================

llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=api_key,
    temperature=0
)


# ==========================================
# Extract Text From PDF
# ==========================================

def extract_pdf_text(pdf_path):

    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"PDF Error: {e}")

    return text


# ==========================================
# Extract Text From DOCX
# ==========================================

def extract_docx_text(docx_path):

    text = []

    try:
        doc = Document(docx_path)

        for para in doc.paragraphs:

            if para.text.strip():
                text.append(para.text)

    except Exception as e:
        print(f"DOCX Error: {e}")

    return "\n".join(text)


# ==========================================
# Universal File Reader
# ==========================================

def extract_text(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    else:
        raise ValueError(
            "Only PDF and DOCX files are supported"
        )


# ==========================================
# JSON Output Parser
# ==========================================

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="""
You are an expert ATS Resume Parser.

Extract all relevant information from the resume.

Return structured JSON with the following fields:

- name
- email
- phone
- location
- summary
- skills
- education
- experience
- projects
- certifications
- languages

{format_instructions}

Resume:

{resume}
""",
    input_variables=["resume"],
    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    }
)

chain = prompt | llm | parser


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    # Change filename here
    file_path = "documents/SHIBAM PAL RESUME.docx"

    print("\nReading Resume...\n")

    resume_text = extract_text(file_path)

    if not resume_text.strip():
        print("No text extracted from file.")
        exit()

    print("Text Extracted Successfully")
    print(f"Characters Extracted: {len(resume_text)}")

    print("\nParsing Resume Using Mistral...\n")

    try:

        result = chain.invoke(
            {
                "resume": resume_text
            }
        )

        print("\n===== PARSED RESUME =====\n")

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

        with open(
            "parsed_resume.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )

        print("\nSaved to parsed_resume.json")

    except Exception as e:

        print(f"\nError while parsing: {e}")