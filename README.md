AI/LLM Learning & Projects Portfolio

This repository contains my structured learning journey and hands-on implementations across Python, APIs, LLMs, and modern AI frameworks like LangChain and LangGraph.

Contents

1. 🐍 Python Revision (OOPs + Data Structures)
- Object-Oriented Programming (Classes, Objects, Inheritance, Polymorphism)
- Core Data Structures:
  - Lists
  - Tuples
  - Sets
  - Dictionaries
- Problem-solving using Python basics

2. APIs and JSON
- REST API fundamentals
- Working with `requests` library
- JSON parsing and serialization
- Handling API responses in Python

3. Embeddings & Semantic Search
- Text embeddings concepts
- Vector similarity (cosine similarity)
- Semantic search implementation
- Introduction to vector databases

4. Transformers
- Transformer architecture overview
- Attention mechanism
- Encoder-decoder models
- Foundation of modern LLMs

5. LLM Behavior Basics
- How LLMs generate text
- Tokenization basics
- Context windows
- Hallucination and limitations

6. Prompt Engineering
- Zero-shot and few-shot prompting
- Role-based prompting
- Structured outputs
- Prompt optimization techniques

7. LangChain Basics
- Chains (Sequential & Simple Chains)
- LCEL (LangChain Expression Language)
- Prompt Templates
- Chat Models & Chat Templates

Multi-Provider Setup
This project uses multiple LLM providers:
- OpenAI
- Mistral AI
- HuggingFace integrations

---

8. Resume Parser Project (LangChain + LangGraph)
Overview
An AI-powered resume parser that extracts structured information from resumes using LLMs and LangGraph orchestration.

Tech Stack
- LangChain
- LangGraph
- Mistral AI
- Pydantic (structured output)
- PDF & DOCX parsing libraries

Features
- Extracts:
  - Name, Email, Phone
  - Skills
  - Education
  - Experience
  - Projects
  - Certifications
  - Languages
- Converts unstructured resume → structured JSON
- Handles PDF and DOCX formats
- Robust LLM output validation

Workflow
1. Load resume (PDF/DOCX)
2. Extract raw text
3. Send to LLM via LangGraph node
4. Structured output parsing (Pydantic)
5. Save JSON output


