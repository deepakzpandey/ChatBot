"""
IPC Legal Assistant
-------------------
A web-based legal assistant chatbot for exploring and understanding the Indian Penal Code (IPC).
Developed as a college project for the subject Cyber Laws and Information Crime (CLIC).

Powered by Python, Streamlit, local keyword retrieval, and the Google Gemini API.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv

# Import the local IPC search and dataset module
from ipc_search import load_ipc_dataset, search_ipc_sections, format_sections_for_prompt

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
# Gemini API Helper
# ---------------------------------------------------------
def generate_gemini_response(
    api_key: str,
    user_query: str,
    retrieved_sections: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Sends the user query and retrieved IPC sections to Google Gemini to synthesize
    a legally accurate, concise, and beginner-friendly response.
    
    Returns:
        (response_text, error_message)
    """
    context_text = format_sections_for_prompt(retrieved_sections)
    
    system_instruction = (
        "You are an academic IPC (Indian Penal Code) Legal Assistant designed for college students "
        "studying Cyber Laws and Information Crime. Your duty is to explain legal provisions accurately, "
        "clearly, and objectively.\n\n"
        "STRICT LEGAL ACCURACY RULES:\n"
        "1. PRIMARY SOURCE: Rely strictly on the provided IPC sections below. Do NOT invent, assume, or fabricate "
        "any section numbers, offenses, or punishments.\n"
        "2. FACT VERIFICATION: Distinguish explicitly between what is directly codified in the retrieved text "
        "and any unverified general legal principles. If a specific detail is not stated in the text, explicitly state so.\n"
        "3. HISTORICAL CONTEXT & BNS TRANSITION: When answering, note that the Indian Penal Code (IPC), 1860 was replaced "
        "by the Bharatiya Nyaya Sanhita (BNS), 2023 on 1 July 2024. Clarify that IPC applies to offences committed on or before "
        "30 June 2024 (subject to transitional provisions), while BNS governs offences committed on or after 1 July 2024.\n"
        "4. CITATION: Explicitly cite the IPC Section number(s) and title(s) in your explanation.\n"
        "5. DISCLAIMER: Always maintain an educational tone and remind the user that this does not constitute formal legal counsel."
    )
    
    prompt = (
        f"{system_instruction}\n\n"
        f"=== RETRIEVED IPC SECTIONS FROM DATASET ===\n"
        f"{context_text}\n"
        f"===========================================\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"Please provide a structured, simple explanation:"
    )

    # 1. Try the modern google.genai SDK
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # We use gemini-3.8-flash as the fast, standard model
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text, None
        return None, "Empty response received from Gemini API."
    except Exception as err_genai:
        # 2. Fallback to google.generativeai if available
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=api_key)
            model = genai_legacy.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text, None
            return None, "Empty response from Gemini fallback model."
        except Exception as err_legacy:
            return None, f"Gemini API Error: {str(err_genai)}"


# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
def render_sidebar(dataset_count: int, api_key_present: bool):
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
            
        # Gemini API Status Indicator
        st.markdown("### 🤖 Gemini API Status")
        if api_key_present:
            st.success("✓ Gemini API: **Configured**")
        else:
            st.warning("⚠️ Gemini API: **Not Set** (Offline Retrieval Mode)")
            with st.expander("🔑 How to configure Gemini API Key"):
                st.markdown(
                    """
                    1. Get a free API key at **[Google AI Studio](https://aistudio.google.com/)**.
                    2. In the project folder, open the `.env` file.
                    3. Add your key:
                       ```bash
                       GEMINI_API_KEY=your_actual_key_here
                       ```
                    4. Save the file and reload this page.
                    *(You can still search and view raw IPC sections without a key!)*
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
    
    # Read API Key safely from environment or .env
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    api_key_present = bool(api_key)
    
    # Render Sidebar
    render_sidebar(len(dataset), api_key_present)
    
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
    if not api_key_present:
        st.info(
            "ℹ️ **Running in Keyword Search Mode:** No Gemini API key detected in `.env`. "
            "The assistant will retrieve matching IPC sections directly from the dataset. "
            "To enable AI explanations, add `GEMINI_API_KEY` to `.env`."
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
                with st.expander(f"📚 View Retrieved IPC Sections ({len(sections)})"):
                    for s in sections:
                        st.markdown(
                            f"**Section {s.get('Section')}: {s.get('section_title')}** "
                            f"*(Chapter {s.get('chapter')}: {s.get('chapter_title')})*"
                        )
                        st.markdown(f"> {s.get('section_desc')}")
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
        matching_sections = search_ipc_sections(query_text, dataset, max_results=4)
        
        # 3. Generate Assistant Response
        with st.chat_message("assistant"):
            # Case A: No sections found in dataset
            if not matching_sections:
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
                
            # Case B: Sections found, Gemini API Key is configured
            elif api_key_present:
                with st.spinner("Analyzing IPC sections with Gemini..."):
                    ai_response, api_error = generate_gemini_response(
                        api_key=api_key,
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
                            st.markdown(f"> {s.get('section_desc')}")
                            st.divider()
                            
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ai_response,
                        "sections": matching_sections
                    })
                else:
                    # Fallback if Gemini fails (e.g. network issue or invalid key)
                    fallback_text = (
                        f"⚠️ **Could not connect to Gemini API:** {api_error}\n\n"
                        "Displaying the directly retrieved IPC sections below:"
                    )
                    st.warning(fallback_text)
                    for s in matching_sections:
                        st.markdown(
                            f"### Section {s.get('Section')}: {s.get('section_title')}\n"
                            f"**Chapter {s.get('chapter')} ({s.get('chapter_title')})**\n\n"
                            f"{s.get('section_desc')}\n"
                        )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_text,
                        "sections": matching_sections
                    })
                    
            # Case C: Sections found, but Gemini API Key is missing (Offline Retrieval Mode)
            else:
                summary_heading = (
                    "### 📋 Retrieved Matching IPC Sections\n"
                    "*Gemini API key is not configured. To receive AI-generated summaries, "
                    "add `GEMINI_API_KEY` to `.env`. Below are the matched statutory provisions:*\n\n"
                )
                st.markdown(summary_heading)
                
                full_content_blocks = [summary_heading]
                for s in matching_sections:
                    sec_display = (
                        f"#### ⚖️ Section {s.get('Section')}: {s.get('section_title')}\n"
                        f"**Chapter {s.get('chapter')}:** {s.get('chapter_title')}\n\n"
                        f"> {s.get('section_desc')}\n\n"
                    )
                    st.markdown(sec_display)
                    full_content_blocks.append(sec_display)
                    
                full_message = "\n".join(full_content_blocks)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_message,
                    "sections": matching_sections
                })


if __name__ == "__main__":
    main()
