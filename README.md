# ⚖️ IPC Legal Assistant

An interactive, AI-assisted legal chatbot built for college coursework in **Cyber Laws and Information Crime (CLIC)**. This project enables students, researchers, and legal enthusiasts to search, navigate, and understand the **Indian Penal Code (IPC), 1860** using local dataset parsing and Google Gemini AI synthesis.

---

## 📌 Project Overview & Purpose

The **Indian Penal Code (IPC)** served as the foundational criminal code of India from 1860 until it was replaced by the **Bharatiya Nyaya Sanhita (BNS)** on **1 July 2024**. For offenses committed prior to 1 July 2024, the IPC remains the substantive law applied in pending investigations and legal proceedings.

This project fulfills the following objectives:
1. **Direct Statutory Retrieval**: Search through all 575 sections of the IPC using fast, keyword-weighted algorithms.
2. **AI-Powered Synthesis**: Send retrieved sections and queries to the **Google Gemini API** to generate plain-language explanations with accurate statutory references.
3. **No Hallucination Architecture**: If no relevant sections match the query, the chatbot explains that the dataset does not have sufficient information rather than fabricating non-existent provisions.
4. **Beginner-Friendly & Modular**: Uses standard Python and Streamlit without heavy machine learning frameworks or complicated setup.
5. **Offline Readiness**: Can run in offline retrieval mode immediately without requiring an API key.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[Streamlit Interface (app.py)]
    │
    ▼
[Keyword Search Engine (ipc_search.py)]
    │
    ├── Loads data/ipc.json (575 sections)
    ├── Extracts section numbers (e.g., "302", "420")
    ├── Tokenizes and filters stop-words
    └── Ranks sections by title, chapter, and text relevance
    │
    ├── (No sections found) ──> Informs user: "Not in dataset"
    │
    ▼
(Top 1–4 Matching Sections Retrieved)
    │
    ├── [GEMINI_API_KEY Configured?]
    │        ├── YES ──> Prompts Gemini with strict legal ground rules:
    │        │           • Only use retrieved text
    │        │           • Mention BNS 2024 transition
    │        │           • Output student-friendly synthesis
    │        │
    │        └── NO  ──> Displays retrieved statutory sections directly
    │                    with an informative setup guide
    ▼
[Streamlit Chat Response + Expandable Citations]
```

---

## 📁 Project Structure

```
ChatBot/
├── Data/
│   └── ipc.json           # Canonical IPC dataset (575 sections)
├── app.py                 # Streamlit web application & UI
├── ipc_search.py          # Dataset parser, tokenizer, and keyword search
├── requirements.txt       # Project Python dependencies
├── .env.example           # Template for environment configuration
├── .gitignore             # Ignores .env, cache files, and virtualenvs
└── README.md              # Project documentation and guide
```

---

## ⚙️ Prerequisites & Installation

### 1. Python Requirement
- **Python 3.10 to 3.13** installed on your system.
- Check your Python version:
  ```bash
  python --version
  ```

### 2. Install Dependencies
Open your terminal (PowerShell, Command Prompt, or Bash) in the project directory:

```bash
cd D:\College\MTech\CLIC\ChatBot
pip install -r requirements.txt
```

The key dependencies installed are:
* `streamlit`: Fast web UI framework for Python.
* `google-genai`: Official Google Gemini AI Python SDK.
* `python-dotenv`: Loads environment variables from `.env`.

---

## 🔑 Gemini API Key Configuration (Optional for initial setup)

The chatbot is built to run **immediately without an API key** in offline mode. To enable AI-generated summaries:

1. **Obtain a Free Key**:
   Visit [Google AI Studio](https://aistudio.google.com/) and click **"Get API key"**.

2. **Configure `.env`**:
   Copy `.env.example` to create a file named `.env`:
   ```bash
   cp .env.example .env
   ```
   *(Or create a new file named `.env` in the `ChatBot` folder).*

3. **Add your API Key**:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

4. `.env` is automatically ignored by `.gitignore` to prevent leaking API credentials.

---

## 🚀 How to Run the Application

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Streamlit will launch local web server and open the browser at:
```
http://localhost:8501
```

*(If port 8501 is occupied, Streamlit will automatically pick the next available port like 8502).*

---

## 🔍 How Search & AI Generation Work

### 1. Dataset Structure
Each record in `data/ipc.json` contains:
* `chapter`: Integer chapter number.
* `chapter_title`: Name of the chapter (e.g., *"OF OFFENCES AGAINST THE HUMAN BODY"*).
* `Section`: Section number or code (e.g., `302`, `29A`, `420`).
* `section_title`: Official title of the section (e.g., *"Punishment for murder"*).
* `section_desc`: Full statutory definition and penal provisions.

### 2. Search & Retrieval Pipeline (`ipc_search.py`)
1. **Section Extraction**: Regular expressions identify queries mentioning section numbers (e.g., `"Section 302"`, `"sec 420A"`, or `"378"`). If matched, that section receives a **+100 score priority**.
2. **Tokenization & Stop Words**: Removes conversational filler words (`what`, `is`, `punishment`, `for`, `tell`, `me`) while retaining legal nouns (`theft`, `murder`, `cheating`, `record`).
3. **Multi-Field Weighted Scoring**:
   * Exact query phrase in title: `+40`
   * Query phrase in section description: `+20`
   * Keyword match in section number: `+50`
   * Keyword match in section title: `+15`
   * Keyword match in chapter title: `+5`
   * Keyword match in description: `+2` per occurrence
4. **Relevance Threshold**: Sections scoring below the threshold (`12.0`) are excluded. If no sections meet this criteria, search returns empty.

### 3. AI Grounding & Prompt Strategy
When an API key is present:
* The retrieved sections are formatted into an isolated context block.
* Gemini is instructed to **act strictly as an academic legal assistant**.
* Gemini is bound to use **only the retrieved legal text** to avoid fabricating punishments or section numbers.
* The model is mandated to highlight the **historical transition** from the IPC (1860) to the **Bharatiya Nyaya Sanhita (BNS), 2023** effective 1 July 2024.

---

## 🏛️ Legal & Academic Disclaimers

> [!IMPORTANT]
> **BNS 2024 Transition:**  
> The Indian Penal Code (IPC), 1860 was replaced by the **Bharatiya Nyaya Sanhita (BNS), 2023** on **1 July 2024**.  
> * IPC applies to offences committed on or before **30 June 2024** (subject to transitional provisions).  
> * BNS applies to offences committed on or after **1 July 2024**.

> [!WARNING]
> **Academic Project Notice:**  
> This software is an educational prototype developed for the subject **Cyber Laws and Information Crime**. It does not provide legal advice and must not be used as a replacement for consultation with an Advocate or qualified legal professional.
