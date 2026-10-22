"""
IPC Legal Assistant
-------------------
A web-based legal assistant chatbot for exploring and understanding the Indian Penal Code (IPC).
Developed as a college project for the subject Cyber Laws and Information Crime (CLIC).

Powered by Python, Streamlit, local keyword retrieval, and configurable AI providers (Gemini, Groq, OpenRouter).
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

# Import the local IPC search and dataset module
from ipc_search import load_ipc_dataset, search_ipc_sections, format_sections_for_prompt, detect_section_request

# Import the multi-provider AI interface
from ai_provider import get_ai_provider, BaseAIProvider, GeminiProvider

# Load environment variables from .env file if present
load_dotenv()


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPC Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, academic, legal interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .legal-card {
        background-color: #f8fafc;
        border-left: 4px solid #1e3a8a;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        font-size: 0.82rem;
        color: #64748b;
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 10px;
        margin-top: 15px;
        line-height: 1.4;
    }
    .bns-box {
        font-size: 0.85rem;
        background-color: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        color: #1e40af;
    }
    .setup-box {
        background-color: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 10px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dataset Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner="Loading IPC Dataset...")
def get_cached_ipc_dataset() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Loads and caches the IPC dataset to avoid reading disk on every rerun."""
    return load_ipc_dataset()


# ---------------------------------------------------------
# Backward Compatibility Helper
# ---------------------------------------------------------
def generate_gemini_response(
    api_key: str,
    user_query: str,
    retrieved_sections: List[Dict[str, Any]],
    sleep_fn: Any = time.sleep
) -> Tuple[Optional[str], Optional[str]]:
    """Backward-compatible helper that delegates to GeminiProvider."""
    p = GeminiProvider(api_key=api_key)
    return p.generate_response(user_query, retrieved_sections, sleep_fn=sleep_fn)


# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
def render_sidebar(dataset_count: int, provider: BaseAIProvider):
    with st.sidebar:
        st.markdown("## ⚖️ About Project")
        st.markdown(
            "**IPC Legal Assistant** is an educational chatbot developed for "
            "the academic course **Cyber Laws and Information Crime**."
        )
        st.divider()
        
        # Dataset Status Indicator
        st.markdown("### 📊 Dataset Status")
        if dataset_count > 0:
            st.success(f"✓ IPC Dataset: **{dataset_count} Sections** Loaded")
        else:
            st.error("✗ IPC Dataset: Not Loaded")
            
        # AI Provider Status Indicator
        st.markdown("### 🤖 AI Provider Configuration")
        if provider.is_configured():
            st.success(f"✓ **{provider.name}**: Active\n\nModel: `{provider.model}`")
        else:
            st.warning(f"⚠️ **{provider.name}**: Key Not Set\n\n*(Running in Offline Retrieval Mode)*")
            
        with st.expander("⚙️ Provider & API Setup Guide"):
            st.markdown(
                """
                Switch AI providers anytime in `.env` without changing code:
                
                **1. Google Gemini (Default)**
                ```bash
                AI_PROVIDER=gemini
                AI_API_KEY=your_gemini_key
                AI_MODEL=gemini-2.5-flash
                ```
                *Free key: [Google AI Studio](https://aistudio.google.com/)*

                **2. Groq (High Speed)**
                ```bash
                AI_PROVIDER=groq
                AI_API_KEY=your_groq_key
                AI_MODEL=qwen/qwen3.8-27b
                ```
                *Free key: [Groq Console](https://console.groq.com/)*

                **3. OpenRouter (Unified API)**
                ```bash
                AI_PROVIDER=openrouter
                AI_API_KEY=your_openrouter_key
                AI_MODEL=meta-llama/llama-3.3-70b-instruct:free
                ```
                *Key: [OpenRouter](https://openrouter.ai/)*
                
                *(You can still search and view raw IPC sections without an API key!)*
                """
            )
                
        st.divider()
        
        # BNS 2024 Legal Context
        st.markdown(
            """
            <div class="bns-box">
                <strong>🏛️ BNS 2024 Transition:</strong><br>
                The Indian Penal Code (1860) was repealed and replaced by the 
                <strong>Bharatiya Nyaya Sanhita (BNS), 2023</strong> effective 
                <strong>1 July 2024</strong>. Offences committed prior remain 
                prosecutable under the IPC.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Legal Disclaimer
        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>⚠️ Legal Disclaimer:</strong><br>
                This system is an academic demonstration developed for university study. 
                It does not constitute legal advice, representation, or authoritative statutory text. 
                Consult a qualified legal advocate for actual legal matters.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Load dataset
    dataset, load_error = get_cached_ipc_dataset()
    
    # Initialize the configured AI Provider (Gemini, Groq, OpenRouter)
    provider = get_ai_provider()
    api_configured = provider.is_configured()
    
    # Render Sidebar
    render_sidebar(len(dataset), provider)
    
    # Main Header
    st.markdown('<div class="main-header">⚖️ IPC Legal Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Search Indian Penal Code (IPC) provisions, understand penalties, and review legal definitions.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # If dataset failed to load, display clear error and halt
    if load_error or not dataset:
        st.error(
            f"❌ **Dataset Loading Failed:** {load_error or 'No records found.'}\n\n"
            "Please ensure `data/ipc.json` is located in the project directory."
        )
        return
        
    # Inform user if running in offline keyword mode
    if not api_configured:
        st.info(
            f"ℹ️ **Running in Keyword Search Mode:** No API key detected for {provider.name}. "
            "The assistant will retrieve matching IPC sections directly from the dataset. "
            "To enable AI explanations, configure `AI_API_KEY` in `.env`."
        )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display welcome banner if chat is empty
    if len(st.session_state.messages) == 0:
        with st.container():
            st.markdown("### 💬 Suggested Queries")
            st.markdown("You can ask about specific sections, offences, or cyber-related definitions:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("- **Section 302:** *What is the punishment for murder?*")
                st.markdown("- **Section 378 & 379:** *What constitutes theft and its punishment?*")
            with col2:
                st.markdown("- **Section 420:** *Explain cheating and dishonest delivery of property.*")
                st.markdown("- **Section 29A:** *How does IPC define an electronic record?*")
            st.divider()

    # Render previous messages from history
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        sections = message.get("sections", [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            # If there are attached sections, render them in an expander
            if sections:
                is_fallback = "temporarily unavailable" in content or "not configured" in content
                with st.expander(f"📚 View Retrieved IPC Sections ({len(sections)})", expanded=is_fallback):
                    for s in sections:
                        st.markdown(
                            f"**Section {s.get('Section')}: {s.get('section_title')}** "
                            f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*"
                        )
                        st.markdown(f"**Original Legal Text:**\n> {s.get('section_desc')}")
                        st.divider()

    # Chat Input Box
    user_query = st.chat_input("Ask a question about IPC sections, offences, or penalties...")
    
    if user_query:
        query_text = user_query.strip()
        if not query_text:
            return
            
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            st.markdown(query_text)
            
        # 2. Retrieve Relevant Sections from IPC dataset
        requested_sec = detect_section_request(query_text)
        matching_sections = search_ipc_sections(query_text, dataset, max_results=4)
        
        # 3. Generate Assistant Response
        with st.chat_message("assistant"):
            # Case A: No sections found in dataset
            if not matching_sections:
                if requested_sec:
                    no_results_text = (
                        f"Section **{requested_sec}** was not found in the Indian Penal Code (IPC) dataset.\n\n"
                        f"The IPC dataset does not contain Section {requested_sec}. "
                        "Please verify the section number or check whether this provision belongs to other legislation "
                        "(such as the Information Technology Act 2000 or the new Bharatiya Nyaya Sanhita 2023)."
                    )
                else:
                    no_results_text = (
                        "I could not find any relevant sections in the Indian Penal Code (IPC) dataset matching your query.\n\n"
                        "The available dataset does not contain sufficient information regarding this matter. "
                        "Please verify your keywords or section number and try again.\n\n"
                        "*Note: Special legislation like the Information Technology Act, 2000 or provisions enacted under "
                        "the new Bharatiya Nyaya Sanhita (BNS), 2023 may not be part of this historical IPC dataset.*"
                    )
                st.markdown(no_results_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": no_results_text,
                    "sections": []
                })
                
            # Case B: Sections found, AI Provider is configured
            elif api_configured:
                with st.spinner(f"Analyzing IPC sections with {provider.name}..."):
                    ai_response, api_error = provider.generate_response(
                        user_query=query_text,
                        retrieved_sections=matching_sections
                    )
                    
                if ai_response:
                    st.markdown(ai_response)
                    with st.expander(f"📚 View Retrieved IPC Sections ({len(matching_sections)})"):
                        for s in matching_sections:
                            st.markdown(
                                f"**Section {s.get('Section')}: {s.get('section_title')}** "
                                f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*"
                            )
                            st.markdown(f"**Original Legal Text:**\n> {s.get('section_desc')}")
                            st.divider()
                            
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "sections": matching_sections
                    })
                else:
                    # Fallback when AI provider is unavailable (e.g. 503, 429, or network issue)
                    notice_msg = (
                        f"⚠️ **AI-generated explanations are temporarily unavailable.** ({api_error})\n\n"
                        "Displaying the statutory provisions directly from the official IPC dataset below:"
                    )
                    st.warning(f"⚠️ **AI-generated explanations are temporarily unavailable.** ({api_error})")
                    st.markdown("### 📋 Statutory Provisions (IPC Dataset)")
                    for s in matching_sections:
                        st.markdown(
                            f"#### ⚖️ Section {s.get('Section')}: {s.get('section_title')}\n"
                            f"**Chapter {s.get('chapter')}:** {s.get('chapter_title')}\n\n"
                            f"**Original Legal Text:**\n"
                            f"> {s.get('section_desc')}\n"
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": notice_msg,
                        "sections": matching_sections
                    })
                    
            # Case C: Sections found, but AI Provider is not configured (Offline Retrieval Mode)
            else:
                summary_heading = (
                    f"⚠️ **AI-generated explanations are temporarily unavailable** ({provider.name} API key is not configured in `.env`).\n\n"
                    "Displaying official legal text directly from the IPC dataset (`data/ipc.json`):\n\n"
                )
                st.info(summary_heading)
                st.markdown("### 📋 Statutory Provisions (IPC Dataset)")
                for s in matching_sections:
                    st.markdown(
                        f"#### ⚖️ Section {s.get('Section')}: {s.get('section_title')}\n"
                        f"**Chapter {s.get('chapter')}:** {s.get('chapter_title')}\n\n"
                        f"**Original Legal Text:**\n"
                        f"> {s.get('section_desc')}\n"
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary_heading,
                    "sections": matching_sections
                })


if __name__ == "__main__":
    main()
