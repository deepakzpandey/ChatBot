"""
IPC Search Module
-----------------
Handles loading, parsing, and keyword-based retrieval for the Indian Penal Code (IPC) dataset.
Designed to be modular, robust, and beginner-friendly for educational purposes.
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional, Any


# Common English grammatical and conversational stop words to exclude from keyword scoring
STOP_WORDS = {
    "what", "is", "the", "a", "an", "for", "in", "of", "to", "and", "or",
    "tell", "me", "about", "which", "are", "by", "under", "code", "indian",
    "penal", "ipc", "law", "act", "can", "you", "i", "how", "does", "do",
    "explain", "give", "details", "on", "with", "from", "as", "at", "be",
    "this", "that", "there", "any", "all", "who", "whom", "whose", "why",
    "section", "sections", "sec"
}


def load_ipc_dataset(filepath: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Loads and parses the IPC JSON dataset from disk.
    
    Checks standard locations (data/ipc.json, Data/ipc.json) relative to this file
    or uses the provided filepath.
    
    Returns:
        (dataset_list, error_message): 
        dataset_list is non-empty if loading succeeds and error_message is None.
        If loading fails, dataset_list is empty and error_message contains details.
    """
    # Determine possible paths
    candidate_paths = []
    
    if filepath:
        candidate_paths.append(filepath)
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths.extend([
        os.path.join(base_dir, "data", "ipc.json"),
        os.path.join(base_dir, "Data", "ipc.json"),
        os.path.join(base_dir, "ipc.json"),
        os.path.join(os.getcwd(), "data", "ipc.json"),
        os.path.join(os.getcwd(), "Data", "ipc.json"),
    ])
    
    target_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            target_path = p
            break
            
    if not target_path:
        return [], f"Dataset file 'ipc.json' could not be found. Checked paths: {candidate_paths[:2]}"
        
    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            return [], f"Expected dataset to be a JSON array of objects, but got {type(data).__name__}."
            
        if len(data) == 0:
            return [], "Dataset is empty."
            
        # Clean and normalize records
        normalized_data = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
                
            # Safely extract and format fields regardless of int or str types
            raw_section = item.get("Section", "")
            section_str = str(raw_section).strip()
            
            raw_title = item.get("section_title", "")
            section_title = str(raw_title).strip()
            
            raw_desc = item.get("section_desc", "")
            section_desc = str(raw_desc).strip()
            
            raw_chap = item.get("chapter", "")
            chapter_str = str(raw_chap).strip()
            
            raw_chap_title = item.get("chapter_title", "")
            chapter_title = str(raw_chap_title).strip()
            
            # Clean up common encoding artifacts (e.g., replacement characters from quotes)
            section_desc = section_desc.replace("\ufffd", "'")
            section_title = section_title.replace("\ufffd", "'")
            
            normalized_data.append({
                "id": index,
                "Section": section_str,
                "section_title": section_title,
                "section_desc": section_desc,
                "chapter": chapter_str,
                "chapter_title": chapter_title,
            })
            
        return normalized_data, None

    except json.JSONDecodeError as err:
        return [], f"Malformed JSON in dataset ({target_path}): {str(err)}"
    except Exception as err:
        return [], f"Unexpected error reading dataset ({target_path}): {str(err)}"


def extract_section_numbers(query: str) -> List[str]:
    """
    Extracts potential IPC section numbers from a query string.
    Matches formats like 'Section 302', 'Sec 420A', 'sec. 376', or standalone '302'.
    """
    matches = re.findall(r"\b(?:section|sec\.?|u/s)?\s*([0-9]{1,3}[a-zA-Z]?)\b", query, flags=re.IGNORECASE)
    valid_sections = []
    for m in matches:
        m_clean = m.strip().upper()
        # Ensure it starts with 1-3 digits and optionally ends with a letter (e.g., 302, 29A, 120B)
        if re.match(r"^[0-9]{1,3}[A-Z]?$", m_clean):
            valid_sections.append(m_clean)
    return valid_sections


def tokenize(text: str) -> List[str]:
    """
    Breaks a text string into normalized lowercase alphanumeric tokens,
    filtering out short words and common stop words.
    """
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def search_ipc_sections(
    query: str,
    dataset: List[Dict[str, Any]],
    max_results: int = 4,
    min_score: float = 12.0
) -> List[Dict[str, Any]]:
    """
    Searches the IPC dataset using weighted keyword and section-number matching.
    
    Scoring weights:
    - Target section number match: +100
    - Exact query phrase in title: +40
    - Exact query phrase in description: +20
    - Token match in Section identifier: +50
    - Token match in section title: +15
    - Token match in chapter title: +5
    - Token match in section description: +2 (capped per term)
    
    Returns:
        List of matching section dictionaries sorted by relevance score descending.
        Returns an empty list if no record scores >= min_score.
    """
    if not query or not query.strip() or not dataset:
        return []
        
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    # 1. Check for explicit section references (e.g., "Section 302")
    target_sections = extract_section_numbers(query_lower)
    
    # 2. Tokenize query
    tokens = tokenize(query_lower)
    
    # If the user only typed stop words or special chars and no sections, return empty
    if not tokens and not target_sections:
        return []
        
    cleaned_query_phrase = " ".join(tokens)
    scored_results = []
    
    for item in dataset:
        score = 0.0
        sec = item["Section"].upper()
        title_lower = item["section_title"].lower()
        desc_lower = item["section_desc"].lower()
        chap_title_lower = item["chapter_title"].lower()
        
        # Priority 1: Direct Section Match
        if sec in target_sections:
            score += 100.0
            
        # Priority 2: Phrase matching
        if cleaned_query_phrase:
            if cleaned_query_phrase in title_lower:
                score += 40.0
            elif cleaned_query_phrase in desc_lower:
                score += 20.0
                
        # Priority 3: Token-level keyword matching
        for token in tokens:
            # Token matches the section number itself (e.g. user typed "302" or "420")
            if token == sec.lower():
                score += 50.0
                
            # Title match (high relevance)
            if re.search(r"\b" + re.escape(token) + r"\b", title_lower):
                score += 15.0
                
            # Chapter match (contextual relevance)
            if re.search(r"\b" + re.escape(token) + r"\b", chap_title_lower):
                score += 5.0
                
            # Description match (content relevance, capped per token)
            desc_matches = len(re.findall(r"\b" + re.escape(token) + r"\b", desc_lower))
            if desc_matches > 0:
                score += min(desc_matches * 2.0, 10.0)
                
        if score >= min_score:
            result_item = item.copy()
            result_item["relevance_score"] = round(score, 1)
            scored_results.append(result_item)
            
    # Sort descending by relevance score
    scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored_results[:max_results]


def format_sections_for_prompt(sections: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved section dictionaries into a clean text block
    to be included as context in the Gemini prompt.
    """
    if not sections:
        return "No relevant sections found."
        
    formatted_blocks = []
    for s in sections:
        block = (
            f"--- IPC SECTION {s.get('Section')} ---\n"
            f"Title: {s.get('section_title')}\n"
            f"Chapter {s.get('chapter')}: {s.get('chapter_title')}\n"
            f"Description: {s.get('section_desc')}\n"
        )
        formatted_blocks.append(block)
        
    return "\n".join(formatted_blocks)
