# ⚖️ IPC Legal Assistant

An interactive, AI-assisted legal chatbot built for college coursework in **Cyber Laws and Information Crime (CLIC)**. This project enables students, researchers, and legal enthusiasts to search, navigate, and understand the **Indian Penal Code (IPC), 1860** using local dataset parsing and a pluggable, multi-provider AI synthesis interface.

---

## 📌 Project Overview & Purpose

The **Indian Penal Code (IPC)** served as the foundational criminal code of India from 1860 until it was replaced by the **Bharatiya Nyaya Sanhita (BNS)** on **1 July 2024**. For offenses committed prior to 1 July 2024, the IPC remains the substantive law applied in pending investigations and legal proceedings.

This project fulfills the following objectives:
1. **Direct Statutory Retrieval**: Search through all 575 sections of the IPC using fast, keyword-weighted algorithms and exact section number string matching.
2. **Multi-Provider AI Synthesis**: A common abstraction layer (`ai_provider.py`) isolates AI integration, allowing effortless switching between **Google Gemini**, **Groq**, and **OpenRouter** simply by modifying environment variables.
3. **No Hallucination Architecture**: If no relevant sections match the query or a non-existent section is requested, the chatbot explains that the dataset does not have sufficient information rather than fabricating legal provisions.
4. **Beginner-Friendly & Modular**: Uses standard Python and Streamlit with zero complex machine learning pipelines.
5. **Offline Readiness & Graceful Fallback**: If an API key is missing or the external AI service encounters downtime/rate-limits, the chatbot automatically displays the exact statutory provisions and original legal text from the local dataset.

---

## 🏗️ System Architecture

```
User Query (e.g., "IPC 34" or "What is theft?")
    │
    ▼
[Streamlit Interface (app.py)]
    │
    ▼
[Keyword & Section Search Engine (ipc_search.py)]
    │
    ├── Exact Section Query (e.g. "IPC 34", "Section 34A")
    │        ├── Match Found ──> Returns strictly that section (no noisy digits)
    │        └── Not Found   ──> Informs user: "Section not found in dataset"
    │
    └── General Keyword Query (e.g. "What is theft?")
             └── Ranks by title phrase, section title, and statutory description
    │
    ▼
(Top Matching Sections Retrieved from data/ipc.json)
    │
    ▼
[Pluggable AI Provider Layer (ai_provider.py)]
    │
    ├── Checks .env Configuration:
    │        • AI_PROVIDER: gemini | groq | openrouter
    │        • AI_API_KEY
    │        • AI_MODEL (Optional)
    │
    ├── Provider Execution:
    │        ├── Google Gemini  ──> google-genai SDK (with 503 exponential backoff)
    │        ├── Groq           ──> groq SDK (high-speed inference)
    │        └── OpenRouter     ──> unified REST completions API
    │
    ├── [Success] ──> Synthesized legal explanation + expandable source cards
    └── [Failure / Key Missing] ──> Graceful Fallback:
                                    • States AI is unavailable
                                    • Displays exact Section number & title
                                    • Displays original legal text from JSON
```

---

## 📁 Project Structure

```
ChatBot/
├── Data/
│   └── ipc.json           # Canonical IPC dataset (575 sections, untouched)
├── app.py                 # Streamlit web application & UI orchestrator
├── ai_provider.py         # Multi-provider AI interface (Gemini, Groq, OpenRouter)
├── ipc_search.py          # Dataset parser, tokenizer, and section retrieval
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
Open your terminal in the project directory:

```bash
cd D:\College\MTech\CLIC\ChatBot
pip install -r requirements.txt
```

Installed packages:
* `streamlit`: Web user interface.
* `google-genai`: Official Google Gemini AI SDK.
* `groq`: Official Groq Cloud SDK.
* `requests`: HTTP client for OpenRouter.
* `python-dotenv`: Environment variable loader.

---

## 🤖 Configuring AI Providers

Switch providers at any time by updating your `.env` file. You do **not** need to touch `app.py` or any source code.

Copy the example template:
```bash
cp .env.example .env
```

### Option 1: Google Gemini (Default)
1. Get a free API key at [Google AI Studio](https://aistudio.google.com/).
2. Set in `.env`:
   ```env
   AI_PROVIDER=gemini
   AI_API_KEY=your_gemini_api_key_here
   AI_MODEL=gemini-2.5-flash
   ```
   *(Legacy key name `GEMINI_API_KEY` is also supported automatically).*

### Option 2: Groq (Ultra-Fast Inference)
1. Get a free API key at [Groq Console](https://console.groq.com/).
2. Set in `.env`:
   ```env
   AI_PROVIDER=groq
   AI_API_KEY=your_groq_api_key_here
   AI_MODEL=qwen/qwen3.8-27b
   ```
   *(Or use `AI_MODEL=openai/gpt-oss-120b`).*

### Option 3: OpenRouter (Unified API Access)
1. Get an API key at [OpenRouter](https://openrouter.ai/).
2. Set in `.env`:
   ```env
   AI_PROVIDER=openrouter
   AI_API_KEY=your_openrouter_api_key_here
   AI_MODEL=meta-llama/llama-3.3-70b-instruct:free
   ```

---

## 🚀 How to Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

If no API key is provided, the application automatically runs in **Offline Retrieval Mode**, allowing you to test dataset search and view original legal texts immediately.

---

## 🛡️ Error Handling & Graceful Fallbacks

The AI provider abstraction handles edge cases gracefully:
* **HTTP 503 (Overloaded):** Retries up to 3 times with exponential backoff (2s, 4s, 8s). If all retries fail, a user-friendly notice is shown.
* **HTTP 429 (Rate Limit):** Immediately caught and presented as a polite quota-exceeded message without locking the UI.
* **HTTP 401 (Invalid Key):** Notifies the user to check their `.env` credentials.
* **Timeouts / Disconnections:** Informs the user of connection difficulties.
* **Guaranteed Fallback:** In all error conditions, the matched statutory provisions from `data/ipc.json` are rendered with their exact Section numbers, titles, and statutory descriptions.

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
