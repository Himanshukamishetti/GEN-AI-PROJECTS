# ✨ Generative AI Take-Home Projects

Two independent, runnable projects:

| Project | What it does | Main technologies |
|---|---|---|
| **01 · Chat with PDFs (RAG)** | Searches PDF content and answers questions with page/source references | Streamlit · LangChain · Gemini · ChromaDB |
| **02 · Structured Data Extraction** | Converts resume/invoice text or PDFs into validated JSON | FastAPI · Gemini structured output · Pydantic · pypdf |

> **Privacy note:** Keep API keys out of source control. Free-tier limits and data-use terms can vary. Do not upload confidential company documents without permission. Review Google's current [Gemini API pricing and terms](https://ai.google.dev/gemini-api/docs/pricing).

---

## 📁 Project layout

```text
GEN AI/
├── basic-rag/                 # Project 1: Streamlit PDF Q&A
│   ├── app.py
│   └── requirements.txt
├── data_extraction/           # Project 2: FastAPI text/PDF extraction
│   ├── app.py
│   └── requirements.txt
├── tests/                     # Offline validation tests
├── .env.example               # Template only; do not put a real key here
├── .gitignore
└── README.md
```

Each project has its **own virtual environment** and **own `venv` file**. You may use the same Gemini key in both `venv` files.

## ✅ Prerequisites

- Python **3.11** recommended.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) → **Get API key / API Keys**.
- Internet access for Gemini API requests and initial package installation.

Create a key in AI Studio, then keep it private. Never paste it into Python source code, chat, or GitHub.

## 🔐 Configure each project’s key

The example file is at the repository root. Make a separate `.env` inside each app folder. In VS Code, open a PowerShell terminal at the `GEN AI` root.

For **Project 1**:

```powershell
cd .\basic-rag
Copy-Item ..\.env.example .env
notepad .env
```

For **Project 2**, return to the root and repeat for its folder:

```powershell
cd ..
cd .\data_extraction
Copy-Item ..\.env.example .env
notepad .env
```

In each `.env`, set your real key:

```dotenv
GOOGLE_API_KEY=your_key_from_google_ai_studio
GEMINI_CHAT_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Save the file. The real key belongs in `.env`, **not** `.env.example`. If you have already created that project’s `.env`, edit it instead of overwriting it.

> If Gemini returns **503 UNAVAILABLE** due to high demand, stop the app and try a supported alternate model by changing `GEMINI_CHAT_MODEL` in that project's `.env` (for example, `gemini-3.1-flash-lite`, if available to your account). A **429** generally indicates a quota/rate limit; wait and retry or check AI Studio limits.

---

# 01 · 📄 Chat with PDFs — RAG

## What happens inside?

```text
PDF upload
   ↓
Extract text page by page (pypdf)
   ↓
Split text into overlapping chunks
   ↓
Create Gemini embeddings → store vectors in ChromaDB
   ↓
Retrieve the most relevant chunks for the question
   ↓
Gemini answers from those chunks and cites sources
```

ChromaDB is used as an **in-memory local vector store** in this version. You do not need to install or run a separate database server. The indexed collection disappears when the app/session is restarted, so upload the PDFs again after a restart.

### Run it — Windows PowerShell

Start at the `GEN AI` repository root. If the `.env` file is not created yet, follow the key setup above first.

```powershell
cd .\basic-rag
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually **http://localhost:8501**. Upload one or more PDFs and ask a question in the chat box.

If PowerShell blocks environment activation, run this once in that terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

If you already created `venv`, do not create it again; just activate it and install the requirements if needed.

### What to expect

- Upload up to **10 PDFs**, each up to **20 MB**.
- Text is split into chunks (900 characters, 150-character overlap) and embedded using Gemini.
- ChromaDB ranks chunks using cosine distance; the app filters weak matches before asking Gemini.
- Answers are instructed to use retrieved PDF text only and include source markers; the UI also lists filenames and page numbers.
- Empty/unreadable files, empty questions, weak retrieval, and API failures receive user-facing messages.

**Scanned/image-only PDFs:** This app extracts selectable text; it does not perform OCR. OCR the PDF first if no text is found.

### Stop the app

In the terminal running Streamlit, press **Ctrl + C**. To leave the virtual environment, run `deactivate`.

---

# 02 · 🧾 Structured Data Extraction — Text and PDF

## What happens inside?

```text
Text request or PDF upload
   ↓
For PDFs: extract selectable text with pypdf
   ↓
Gemini extracts only facts present in the source
   ↓
Pydantic validates the response schema
   ↓
API returns structured JSON
```

The API has two kinds of input:

| Endpoint | Input | Use it when |
|---|---|---|
| `POST /extract/resume` | JSON containing `text` | You already have resume text |
| `POST /extract/invoice` | JSON containing `text` | You already have invoice text |
| `POST /extract/resume-pdf` | PDF file upload | You have a resume/CV PDF |
| `POST /extract/invoice-pdf` | PDF file upload | You have an invoice PDF |

Missing scalar values are returned as `null`; missing lists are returned as `[]`. Dates and identifiers are kept in the format found in the source where possible. The model is instructed not to invent missing information.

### Run it — Windows PowerShell

Open a **second VS Code terminal** at the repository root. If `data_extraction/.env` is not created yet, follow the key setup above first.

```powershell
cd .\data_extraction
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/docs**. Keep the terminal running while testing.

### Try a resume PDF in Swagger

1. Expand **POST `/extract/resume-pdf`**.
2. Click **Try it out**.
3. Click **Choose File** beside the `file` field and select a text-based PDF.
4. Click **Execute**.
5. Scroll to **Responses**. The `200` response body is the extracted JSON.

For an invoice PDF, use **POST `/extract/invoice-pdf`**. For plain text, use `/extract/resume` or `/extract/invoice`; those endpoints expect JSON such as:

```json
{
  "text": "Asha Rao | asha@example.com\nData Engineer at Example Co, 2022-present\nSkills: Python, SQL"
}
```

The API accepts PDFs up to **10 MB** and **50 pages**. Extracted text is limited to **30,000 characters**. Password-protected and scanned/image-only PDFs are not supported by this parser; scanned files need OCR first.

### Stop the API

In the terminal running Uvicorn, press **Ctrl + C**. To leave the virtual environment, run `deactivate`.

---

## 🧪 Offline tests

The tests validate input handling and schema defaults. They **do not call Gemini** and do not require an API key. Activate the `data_extraction` virtual environment, install pytest, then run from the repository root:

```powershell
pip install pytest
cd ..
pytest -q
```

If the terminal is already at the repository root, skip `cd ..`.

## 🧰 Common fixes

| Symptom | What to check |
|---|---|
| `GOOGLE_API_KEY is missing` | Put `.env` next to the app's `app.py`; check the variable is spelled exactly `GOOGLE_API_KEY`. On Windows ensure it was not saved as `.env.txt`. |
| Yellow import squiggles in VS Code | Select that project's interpreter: `basic-rag\venv\Scripts\python.exe` or `data_extraction\venv\Scripts\python.exe`. |
| `503 UNAVAILABLE` | Provider/model is temporarily busy. Retry later or set another supported `GEMINI_CHAT_MODEL` in that app's `.env`. |
| `429` or quota message | Wait and retry; check model/project quota in AI Studio. |
| No text found in PDF | It may be scanned; OCR it first. |
| `Choose File` is missing in Swagger | Use `/extract/resume-pdf` or `/extract/invoice-pdf`, click **Try it out**, and ensure the updated `data_extraction/app.py` and requirements are installed. |
| PowerShell refuses venv activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |

