"""
IPC Search Module
-----------------
Handles loading, parsing, and keyword/section retrieval for the Indian Penal Code (IPC) dataset.
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
    "section", "sections", "sec", "provisions", "provision"
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
                
            # Safely extract and format fields strictly as strings regardless of JSON data type
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


def detect_section_request(query: str) -> Optional[str]:
    """
    Detects whether a user query specifically requests an IPC section number.
    Handles variations such as:
    - 'ipc 34', 'IPC 34', 'IPC section 34', 'ipc sec 34', 'ipc u/s 34'
    - 'section 34', 'sec 34', 'u/s 34'
    - 'Section 34A', 'ipc 34a', '120B'
    - 'what is section 34', 'explain 302', 'tell me about ipc 420'
    - Standalone numbers: '34', '302'
    
    Returns the normalized uppercase section string (e.g., '34', '34A', '120B', '302') or None.
    """
    if not query or not query.strip():
        return None
        
    q = query.strip()
    
    # 1. Matches with IPC prefixes: 'ipc 34', 'ipc section 34', 'ipc sec 34', 'ipc u/s 34'
    m = re.search(r"\b(?:ipc\s+section|ipc\s+sec\.?|ipc\s+u/s|ipc)\s*([0-9]{1,3}[a-zA-Z]{0,2})\b", q, re.IGNORECASE)
    if m:
        return m.group(1).upper()
        
    # 2. Matches with Section prefix: 'section 34', 'sec 34', 'sec. 34', 'u/s 34'
    m = re.search(r"\b(?:section|sec\.?|u/s)\s*([0-9]{1,3}[a-zA-Z]{0,2})\b", q, re.IGNORECASE)
    if m:
        return m.group(1).upper()
        
    # 3. Section codes containing letters (e.g. '120B', '34A', '376DA')
    m = re.search(r"\b([0-9]{1,3}[a-zA-Z]{1,2})\b", q)
    if m:
        return m.group(1).upper()
        
    # 4. Standalone section query or short question: '34', 'what is 34?', 'explain 302'
    m = re.search(r"^\s*(?:what\s+is\s+|explain\s+|tell\s+me\s+about\s+)?([0-9]{1,3})\s*\??\s*$", q, re.IGNORECASE)
    if m:
        return m.group(1).upper()
        
    return None


def extract_section_numbers(query: str) -> List[str]:
    """
    Extracts potential section numbers. Kept for backwards compatibility.
    Uses detect_section_request as primary extractor.
    """
    detected = detect_section_request(query)
    if detected:
        return [detected]
    return []


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
    Searches the IPC dataset.
    
    Requirements:
    1. Detect queries containing an IPC section number ('IPC 34', 'section 34', etc.).
    2. Match requested section numbers exactly as strings ('34', '34A', '120A', '120B').
    3. Display exact section first; do NOT include unrelated sections whose descriptions
       contain matching digits (e.g., '1934' or 'Act 34 of 1934') or unrelated keywords.
    4. For general queries ('What is theft under IPC?'), retain weighted keyword search
       and return only the most relevant results.
    
    Returns:
        List of matching section dictionaries.
    """
    if not query or not query.strip() or not dataset:
        return []
        
    query_clean = query.strip()
    
    # ------------------------------------------------------------------
    # Step 1: Detect explicit section number request (e.g. "IPC 34", "Section 34A")
    # ------------------------------------------------------------------
    requested_sec = detect_section_request(query_clean)
    
    if requested_sec:
        # Search dataset for an exact string match on Section
        exact_match = None
        for item in dataset:
            if str(item.get("Section", "")).strip().upper() == requested_sec:
                exact_match = item.copy()
                exact_match["relevance_score"] = 100.0
                break
                
        # Determine if the query contained additional substantive keywords
        tokens = tokenize(query_clean)
        # Exclude the section identifier and digits that are part of it
        remaining_tokens = [
            t for t in tokens 
            if t.upper() != requested_sec and not (t.isdigit() and t in requested_sec)
        ]
        
        # If the query was purely asking for this section (e.g., 'ipc 34', 'section 34', 'what is section 34')
        if not remaining_tokens:
            if exact_match:
                # Return strictly the requested section with no unrelated matches
                return [exact_match]
            else:
                # Requested section does not exist (e.g., 'Section 34A', 'Section 999')
                # Never return unrelated sections
                return []
        else:
            # Compound query (e.g., "Section 302 murder penalty")
            if not exact_match:
                # Section requested does not exist
                return []
                
            # Put exact match first
            results = [exact_match]
            
            # Search for related sections based on remaining keywords, without matching numbers
            cleaned_query_phrase = " ".join(remaining_tokens)
            scored_candidates = []
            
            for item in dataset:
                if str(item.get("Section", "")).strip().upper() == requested_sec:
                    continue  # Already included first
                    
                score = 0.0
                title_lower = item["section_title"].lower()
                desc_lower = item["section_desc"].lower()
                chap_lower = item["chapter_title"].lower()
                
                # Match phrase using word boundaries
                if cleaned_query_phrase and re.search(r"\b" + re.escape(cleaned_query_phrase) + r"\b", title_lower):
                    score += 30.0
                    
                for token in remaining_tokens:
                    # Ignore purely numeric tokens to avoid matching year/act numbers
                    if token.isdigit():
                        continue
                    if re.search(r"\b" + re.escape(token) + r"\b", title_lower):
                        score += 15.0
                    if re.search(r"\b" + re.escape(token) + r"\b", chap_lower):
                        score += 5.0
                    desc_matches = len(re.findall(r"\b" + re.escape(token) + r"\b", desc_lower))
                    if desc_matches > 0:
                        score += min(desc_matches * 2.0, 8.0)
                        
                if score >= min_score:
                    item_copy = item.copy()
                    item_copy["relevance_score"] = round(score, 1)
                    scored_candidates.append(item_copy)
                    
            scored_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
            results.extend(scored_candidates[:max_results - 1])
            return results

    # ------------------------------------------------------------------
    # Step 2: General keyword search (e.g. "What is theft under IPC?")
    # ------------------------------------------------------------------
    tokens = tokenize(query_clean)
    if not tokens:
        return []
        
    cleaned_query_phrase = " ".join(tokens)
    scored_results = []
    
    for item in dataset:
        score = 0.0
        sec = str(item.get("Section", "")).strip().upper()
        title_lower = item["section_title"].lower()
        desc_lower = item["section_desc"].lower()
        chap_title_lower = item["chapter_title"].lower()
        
        # Priority A: Exact title match (e.g. title is exactly 'Theft')
        if title_lower == cleaned_query_phrase:
            score += 60.0
        elif cleaned_query_phrase and re.search(r"\b" + re.escape(cleaned_query_phrase) + r"\b", title_lower):
            score += 35.0
            
        # Priority B: Token-level keyword matching with word boundaries
        for token in tokens:
            # Exact section number match if user typed a number as keyword
            if token.upper() == sec:
                score += 50.0
                
            # Title word match
            if re.search(r"\b" + re.escape(token) + r"\b", title_lower):
                score += 20.0
                
            # Chapter word match
            if re.search(r"\b" + re.escape(token) + r"\b", chap_title_lower):
                score += 5.0
                
            # Description word match using word boundaries
            desc_matches = len(re.findall(r"\b" + re.escape(token) + r"\b", desc_lower))
            if desc_matches > 0:
                score += min(desc_matches * 2.0, 10.0)
                
        if score >= min_score:
            result_item = item.copy()
            result_item["relevance_score"] = round(score, 1)
            scored_results.append(result_item)
            
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
