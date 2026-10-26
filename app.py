"""
IPC Legal Assistant
-------------------
A clean, professional legal assistant chatbot for exploring and understanding the Indian Penal Code (IPC).
Developed for the subject Cyber Laws and Information Crime (CLIC).

Supports real-time token streaming, multi-provider AI integration, and graceful dataset fallback.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple, Generator
import streamlit as st
from dotenv import load_dotenv

# Import dataset parser and search logic
from ipc_search import load_ipc_dataset, search_ipc_sections, detect_section_request

# Import multi-provider AI interface
from ai_provider import get_ai_provider, BaseAIProvider, GeminiProvider

# Load environment configuration
load_dotenv()


# ---------------------------------------------------------
# Page Configuration & Adaptive Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPC Legal Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None
    }
)

# Adaptive CSS compatible with both dark and light themes
st.markdown("""
<style>
    /* Hide Streamlit default top-right menu (Print, Record screen, etc.), toolbar, and footer */
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    footer {
        visibility: hidden !important;
        display: none !important;
    }

    /* Global container adjustments */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 820px;
    }
    
    /* Header typography */
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 1.0rem;
        opacity: 0.75;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }
    
    /* Suggested query label */
    .suggestion-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.65;
        margin-bottom: 0.5rem;
    }

    /* Subtle thinking indicator */
    .thinking-indicator {
        font-size: 0.88rem;
        font-style: italic;
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }

    /* Chat message spacing polish */
    [data-testid="stChatMessage"] {
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Dataset Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_ipc_dataset() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Loads and caches the IPC dataset."""
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
    """Legacy helper for backward compatibility; delegates to GeminiProvider."""
    p = GeminiProvider(api_key=api_key)
    return p.generate_response(user_query, retrieved_sections, sleep_fn=sleep_fn)


# ---------------------------------------------------------
# Sidebar Component (Minimal & Professional)
# ---------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚖️ IPC Legal Assistant")
        st.caption("Cyber Laws & Information Crime | College Project")
        st.markdown("---")
        
        st.markdown(
            "This application provides direct retrieval from the historical "
            "**Indian Penal Code, 1860** dataset coupled with real-time AI-assisted analysis."
        )
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "pending_prompt" in st.session_state:
                del st.session_state["pending_prompt"]
            st.rerun()


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Load dataset
    dataset, load_error = get_cached_ipc_dataset()
    
    # Initialize configured AI Provider (Groq, Gemini, OpenRouter)
    provider = get_ai_provider()
    
    # Render Sidebar
    render_sidebar()
    
    # Homepage Header
    st.markdown('<div class="app-title">⚖️ IPC Legal Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Your assistant for exploring the Indian Penal Code.</div>', unsafe_allow_html=True)
    
    # Dataset failure safety
    if load_error or not dataset:
        st.error("IPC dataset could not be loaded. Please ensure `data/ipc.json` is present.")
        return

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display suggested queries if conversation is empty
    if len(st.session_state.messages) == 0 and "pending_prompt" not in st.session_state:
        st.markdown('<div class="suggestion-label">Suggested Questions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚖️ What is Section 34?", use_container_width=True):
                st.session_state.pending_prompt = "What is Section 34?"
                st.rerun()
            if st.button("📖 What constitutes theft under IPC?", use_container_width=True):
                st.session_state.pending_prompt = "What constitutes theft under IPC?"
                st.rerun()
        with col2:
            if st.button("⚠️ Punishment for murder (Section 302)", use_container_width=True):
                st.session_state.pending_prompt = "What is the punishment for murder under Section 302?"
                st.rerun()
            if st.button("💻 Electronic record definition (Section 29A)", use_container_width=True):
                st.session_state.pending_prompt = "How is an electronic record defined under Section 29A?"
                st.rerun()
        st.markdown("<div style='margin-bottom: 1.0rem;'></div>", unsafe_allow_html=True)

    # Render previous messages from history in correct chronological order
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        sections = message.get("sections", [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            # Display relevant IPC section references below the answer
            if sections:
                with st.expander(f"📚 Relevant IPC Sections ({len(sections)})"):
                    for s in sections:
                        st.markdown(
                            f"**Section {s.get('Section')}: {s.get('section_title')}** "
                            f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*"
                        )
                        st.markdown(f"> {s.get('section_desc')}")
                        st.divider()

    # Capture user input (from chat input or from suggested question click)
    chat_input_val = st.chat_input("Ask about an IPC section, offence, or legal definition...")
    user_query = None

    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        user_query = st.session_state.pop("pending_prompt")
    elif chat_input_val:
        user_query = chat_input_val

    if user_query:
        query_text = user_query.strip()
        if not query_text:
            return
            
        # 1. Display User Message Immediately
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            st.markdown(query_text)
            
        # 2. Retrieve Relevant Sections from IPC dataset
        requested_sec = detect_section_request(query_text)
        matching_sections = search_ipc_sections(query_text, dataset, max_results=3)
        
        # 3. Generate Assistant Response (with Real-Time Streaming)
        with st.chat_message("assistant"):
            if provider.is_configured():
                # Placeholder for subtle typing / thinking indicator before the first token arrives
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown('<div class="thinking-indicator">Thinking...</div>', unsafe_allow_html=True)
                
                try:
                    # Obtain live stream generator from AI provider (synthesizes matching sections or general legal knowledge)
                    token_stream = provider.stream_response(
                        user_query=query_text,
                        retrieved_sections=matching_sections
                    )
                    
                    # Generator wrapper: clears thinking indicator on arrival of the first token
                    def progressive_stream():
                        cleared = False
                        for chunk in token_stream:
                            if not cleared:
                                thinking_placeholder.empty()
                                cleared = True
                            yield chunk
                        if not cleared:
                            thinking_placeholder.empty()
                            
                    # Streamlit's native streaming: displays tokens progressively with ChatGPT-like typing
                    full_response = st.write_stream(progressive_stream())
                    
                    # Display Relevant IPC Section References after the stream completes (if any were retrieved)
                    if matching_sections:
                        with st.expander(f"📚 Relevant IPC Sections ({len(matching_sections)})"):
                            for s in matching_sections:
                                st.markdown(
                                    f"**Section {s.get('Section')}: {s.get('section_title')}** "
                                    f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*"
                                )
                                st.markdown(f"> {s.get('section_desc')}")
                                st.divider()
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sections": matching_sections
                    })
                    
                except Exception as err:
                    # Clean up indicator if active
                    thinking_placeholder.empty()
                    err_str = str(err)
                    
                    if "429" in err_str:
                        error_notice = "AI service rate limit reached. Please wait a moment before sending another query."
                    else:
                        error_notice = "AI explanation is temporarily unavailable."
                    
                    # Graceful Fallback: if we have matching sections from local dataset, display them
                    if matching_sections:
                        fallback_intro = f"{error_notice} Displaying statutory provisions directly from the IPC dataset:"
                        st.info(fallback_intro)
                        for s in matching_sections:
                            st.markdown(
                                f"**Section {s.get('Section')}: {s.get('section_title')}** "
                                f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*\n\n"
                                f"> {s.get('section_desc')}"
                            )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": fallback_intro,
                            "sections": matching_sections
                        })
                    else:
                        st.warning(error_notice)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_notice,
                            "sections": []
                        })
                        
            # Offline Mode: AI Provider is not configured
            else:
                if matching_sections:
                    fallback_intro = "Displaying relevant statutory provisions directly from the IPC dataset:"
                    st.markdown(fallback_intro)
                    for s in matching_sections:
                        st.markdown(
                            f"**Section {s.get('Section')}: {s.get('section_title')}** "
                            f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*\n\n"
                            f"> {s.get('section_desc')}"
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_intro,
                        "sections": matching_sections
                    })
                else:
                    if requested_sec:
                        no_results_text = (
                            f"Section **{requested_sec}** was not found in the Indian Penal Code (IPC) dataset.\n\n"
                            "Please verify the section number or configure an AI API key in `.env` to enable intelligent "
                            "multi-statute search across BNS and IT Act provisions."
                        )
                    else:
                        no_results_text = (
                            "No matching statutory provisions were found in the local IPC dataset for this query.\n\n"
                            "Please configure an AI API key in `.env` (Groq, Gemini, or OpenRouter) to enable "
                            "intelligent responses for cyber laws, BNS updates, and legal concepts."
                        )
                    st.markdown(no_results_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": no_results_text,
                        "sections": []
                    })


if __name__ == "__main__":
    main()
