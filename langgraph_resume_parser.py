import os
import json
import pdfplumber

from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from docx import Document
from pydantic import BaseModel, Field, field_validator

from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, END


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")


# ==================================================
# LLM
# ==================================================

llm = ChatMistralAI(
    model="mistral-medium-latest",
    api_key=MISTRAL_API_KEY,
     temperature=0,
    timeout=120,   
    max_retries=3 
)


# ==================================================
# STRUCTURED OUTPUT SCHEMA 
# ==================================================

class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None
    score: Optional[str] = None


class Experience(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    duration: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)


class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None

    summary: Optional[str] = None

    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

    # handle LLM returning null instead of []
    @field_validator(
        "skills",
        "education",
        "experience",
        "projects",
        "certifications",
        "languages",
        mode="before",
    )
    @classmethod
    def fix_none_lists(cls, value):
        if value is None:
            return []
        return value


structured_llm = llm.with_structured_output(Resume)


# ==================================================
# PDF READER
# ==================================================

def extract_pdf_text(pdf_path: str) -> str:
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# ==================================================
# DOCX READER
# ==================================================

def extract_docx_text(docx_path: str) -> str:
    doc = Document(docx_path)

    return "\n".join(
        para.text for para in doc.paragraphs if para.text.strip()
    )


# ==================================================
# UNIVERSAL FILE READER
# ==================================================

def extract_text(file_path: str) -> str:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf_text(file_path)

    if ext == ".docx":
        return extract_docx_text(file_path)

    raise ValueError("Only PDF and DOCX files are supported.")


# ==================================================
# LANGGRAPH STATE
# ==================================================

class ResumeState(TypedDict):
    resume_text: str
    parsed_resume: dict


# ==================================================
# LANGGRAPH NODE
# ==================================================

def parse_resume(state: ResumeState):

    resume_text = state["resume_text"]

    result = structured_llm.invoke(
        f"""
You are an expert ATS Resume Parser.

Extract all information from the resume accurately.

Rules:
- Do NOT hallucinate
- Use empty values when missing
- Keep structure clean and consistent

Resume:
{resume_text}
"""
    )

    return {
        "parsed_resume": result.model_dump()
    }


# ==================================================
# BUILD LANGGRAPH
# ==================================================

builder = StateGraph(ResumeState)

builder.add_node("parse_resume", parse_resume)

builder.set_entry_point("parse_resume")

builder.add_edge("parse_resume", END)

graph = builder.compile()


# ==================================================
# MAIN
# ==================================================

def main():

    file_path = "documents/SHIBAM PAL RESUME.docx"

    print("\nReading Resume...\n")

    resume_text = extract_text(file_path)

    if not resume_text.strip():
        print("No text extracted from resume.")
        return

    print("Text Extracted Successfully")
    print(f"Characters Extracted: {len(resume_text)}")

    print("\nParsing Resume...\n")

    result = graph.invoke({
        "resume_text": resume_text
    })

    parsed_resume = result["parsed_resume"]

    print("\n===== PARSED RESUME =====\n")

    print(json.dumps(parsed_resume, indent=4, ensure_ascii=False))

    output_file = "parsed_resume.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_resume, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()