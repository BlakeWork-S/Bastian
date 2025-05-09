import streamlit as st
import google.generativeai as genai
import os
import openai # Added
import anthropic # Added
import csv
import time
import pandas as pd
import io # For CSV parsing from string/bytes
import json # For saving/loading configurations
from datetime import datetime # For naming config files

# --- 1. CONFIGURATION & API KEY ---
# For local development, create a .streamlit/secrets.toml file with:
# GEMINI_API_KEY = "YOUR_API_KEY"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")) # Added
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY")) # Added

# --- DEFAULT VALUES (Bastian can override in UI / load from config) ---
DEFAULT_MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Updated default to suggested experimental model
DEFAULT_TOPICS_KEYWORDS_CSV_STRING = """topic_input,primary_keyword,secondary_keywords
how to hire baristas,hire baristas,"barista hiring process, qualities of a good barista, interview questions for baristas"
how to hire line cooks,hire line cooks,"line cook job duties, kitchen staffing, culinary team building"
""" # Example

DEFAULT_APPROVED_INTERNAL_LINKS = """https://workstream.us/product/automated-hiring: Workstream's Automated Hiring Platform
https://workstream.us/product/text-to-apply: Text-to-Apply Feature by Workstream
"""
DEFAULT_APPROVED_EXTERNAL_LINKS = """https://www.bls.gov/ooh/: Bureau of Labor Statistics Occupational Outlook Handbook
https://www.shrm.org/resourcesandtools/tools-and-samples/hr-qa: SHRM HR Q&A
"""
DEFAULT_BRAND_GUIDELINES = "Workstream Brand Voice: Professional yet friendly, practical, authoritative."
DEFAULT_SEO_SUMMARY = "SEO Best Practices: Natural keyword integration, user intent focus, clear structure."

# --- PROMPT TEMPLATES DEFAULTS (Same as previous "Advanced Control" version) ---
PROMPT_TEMPLATES_DEFAULTS = {
    "page_title": """
Task: Generate a Page Title for an article on Workstream's website.
Topic (from CSV/Input): [TOPIC_INPUT]
Primary Keyword (from CSV/Input): [PRIMARY_KEYWORD]
Contextual Information Provided Separately: Workstream Brand Guidelines, SEO Best Practices Summary
Constraints: Max 60 chars. Compelling, SEO-friendly. Include [PRIMARY_KEYWORD].
Output Instructions: Output ONLY the Page Title. No extra text/quotes.
    """,
    "meta_description": """
Task: Generate a Meta Description for an article on Workstream's website.
Topic (from CSV/Input): [TOPIC_INPUT]
Primary Keyword (from CSV/Input): [PRIMARY_KEYWORD]
Secondary Keywords (from CSV/Input, comma-separated): [SECONDARY_KEYWORDS_LIST]
Contextual Information Provided Separately: Workstream Brand Guidelines, SEO Best Practices Summary
Constraints: 140-160 chars. Engaging. Include [PRIMARY_KEYWORD] & ideally a [SECONDARY_KEYWORDS_LIST] keyword. CTA.
Output Instructions: Output ONLY the Meta Description. No extra text/quotes.
    """,
    "h1_tag": """
Task: Generate an H1 Tag for an article on Workstream's website.
Topic (from CSV/Input): [TOPIC_INPUT]
Primary Keyword (from CSV/Input): [PRIMARY_KEYWORD]
Contextual Information Provided Separately: Workstream Brand Guidelines, SEO Best Practices Summary
Constraints: Max 100 chars (aim 60-70). Clear, user-focused, reflect [PRIMARY_KEYWORD]/[TOPIC_INPUT].
Output Instructions: Output ONLY the H1 Tag. No extra text/quotes.
    """,
    "subtitle": """
Task: Generate a Subtitle (lead paragraph/dek) for an article on Workstream's website.
Topic (from CSV/Input): [TOPIC_INPUT]
Contextual Information Provided Separately: Workstream Brand Guidelines
Constraints: Max 200 chars. Value-proposition focused for [TOPIC_INPUT].
Output Instructions: CRITICAL: Wrap entire output in a SINGLE <p class="lead"> tag. Example: <p class="lead">Subtitle here.</p> NO MARKDOWN FENCES (```html). No other text/labels.
    """,
    "alt_text": """
Task: Generate Alt Text for ONE representative image for an article on Workstream's website.
Article Topic (from CSV/Input): [TOPIC_INPUT]
Primary Keyword (from CSV/Input): [PRIMARY_KEYWORD]
Contextual Information Provided Separately: Workstream Brand Guidelines
Context for Image: Imagine a single, clear, relevant image for an article about [TOPIC_INPUT].
Constraints: 10-15 words ideally (max 125 chars). Descriptive. Include [PRIMARY_KEYWORD] if natural. NO "Image of...".
Output Instructions: Output ONLY the Alt Text. No extra text/quotes/labels.
    """,
    "main_text_html": """
Task: Generate the Main Body Text (HTML format) for an SEO-optimized article for Workstream's website.
Topic (from CSV/Input): [TOPIC_INPUT]
Primary Keyword (from CSV/Input): [PRIMARY_KEYWORD]
Secondary Keywords (from CSV/Input, comma-separated list): [SECONDARY_KEYWORDS_LIST]
Contextual Information Provided Separately: Workstream Brand Guidelines, SEO Best Practices Summary, Approved Internal URLs List, Approved External URLs List
Link Inclusion Targets (numbers): Internal Links: [TARGET_NUMBER_INTERNAL_LINKS], External Links: [TARGET_NUMBER_EXTERNAL_LINKS]
---
**Instructions & Constraints (Strictly Follow):**
1.  Content Focus: Comprehensive, high-quality article on [TOPIC_INPUT] for Workstream's audience.
2.  Word Count: Target 700-1100 words.
3.  Tone & Style: Adhere to Workstream Brand Guidelines.
4.  Keyword Integration: Naturally integrate [PRIMARY_KEYWORD] and [SECONDARY_KEYWORDS_LIST]. AVOID STUFFING.
5.  HTML Structure: Use ONLY `<p>`, `<h2>`, `<h3>`, `<ul>`, `<li>`, `<strong>`, `<em>`, `<a href="URL">Anchor</a>`. Structure: Intro, 2-4 H2 sections (with H3s/lists), Conclusion.
6.  Internal Links: Include [TARGET_NUMBER_INTERNAL_LINKS]. CRITICAL: ONLY use URLs from "Approved Internal URLs List". Anchor text natural, descriptive (3-7 words), relevant.
7.  External Links: Aim to include [TARGET_NUMBER_EXTERNAL_LINKS] external links if the target is greater than 0.
    CRITICAL: Prioritize using URLs from the "Approved External URLs List" provided.
    If highly relevant, authoritative, non-competitive external links are needed beyond this list to meet the target or significantly enhance the article's value (e.g., for recent statistics, official non-commercial resources), you MAY source them.
    ALL external links, whether from the approved list or newly sourced, MUST add significant, direct value to the reader. Avoid generic links. Ensure anchor text is descriptive and natural. Absolutely NO links to competitor websites.
8.  Quality & Accuracy: Factually accurate, current, helpful. No fabricated info. Original.
Output Instructions: Output ONLY raw HTML for article body. NO `<html>`, `<head>`, `<body>` tags. NO MARKDOWN FENCES (```html). No other text/labels/preambles.
    """
}

# --- CSV Column Headers ---
CSV_COLUMN_HEADERS = [
    "topic_input", "primary_keyword", "secondary_keywords",
    "page_title", "meta_description", "h1_tag", "subtitle", "alt_text", "main_text_html",
    "found_internal_links_in_html", "found_external_links_in_html"
]

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    defaults = {
        'model_name': DEFAULT_MODEL_NAME,
        'topics_df': pd.DataFrame(columns=['topic_input', 'primary_keyword', 'secondary_keywords']), 
        'uploaded_topics_filename': None,
        'approved_internal_links': DEFAULT_APPROVED_INTERNAL_LINKS,
        'approved_external_links': DEFAULT_APPROVED_EXTERNAL_LINKS,
        'brand_guidelines': DEFAULT_BRAND_GUIDELINES,
        'seo_summary': DEFAULT_SEO_SUMMARY,
        'target_internal_links': 2,
        'target_external_links': 1,
        'llm_temperature': 0.6,
        'editable_prompts': PROMPT_TEMPLATES_DEFAULTS.copy(),
        'config_loaded_successfully': False,
        'active_config_name': "Defaults"
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    if st.session_state.topics_df.empty:
        try:
            default_df = pd.read_csv(io.StringIO(DEFAULT_TOPICS_KEYWORDS_CSV_STRING))
            st.session_state.topics_df = default_df
        except pd.errors.EmptyDataError: 
             st.session_state.topics_df = pd.DataFrame(columns=['topic_input', 'primary_keyword', 'secondary_keywords'])

init_session_state()


# --- CONFIGURATION SAVE/LOAD FUNCTIONS ---
def get_current_config_dict():
    config = {}
    for key in ['model_name', 'approved_internal_links', 'approved_external_links',
                'brand_guidelines', 'seo_summary', 'target_internal_links',
                'target_external_links', 'llm_temperature', 'editable_prompts']:
        config[key] = st.session_state[key]
    config['topics_df_as_list'] = st.session_state.topics_df.to_dict(orient='records')
    return config

def load_config_from_dict(config_dict):
    for key, value in config_dict.items():
        if key == 'topics_df_as_list': 
            st.session_state.topics_df = pd.DataFrame(value)
        elif key in st.session_state:
            st.session_state[key] = value
    st.session_state.config_loaded_successfully = True

# --- GEMINI API INTERACTION FUNCTION ---
def generate_content(prompt_text, model_name, temperature, retries=3, delay_seconds=5):
    st.write(f"Attempting to generate content with model: {model_name}, Temperature: {temperature}")

    if "gemini" in model_name:
        if not GEMINI_API_KEY: 
            st.error("GEMINI_API_KEY not configured.")
            st.write("Error: GEMINI_API_KEY not configured.")
            return "ERROR: API Key missing."
        try:
            st.write(f"Configuring Gemini with API key: {'*' * (len(GEMINI_API_KEY) - 4) + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'Not Set'}")
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(temperature=temperature)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            for attempt in range(retries):
                try:
                    st.write(f"Gemini API call attempt {attempt + 1}")
                    response = model.generate_content(prompt_text, generation_config=generation_config, safety_settings=safety_settings)
                    if response.candidates and response.candidates[0].content.parts:
                        generated_text = response.text.strip()
                        st.write(f"Gemini response successful: {generated_text[:100]}...")
                        return generated_text
                    else:
                        fb = response.prompt_feedback
                        reason = fb.block_reason if fb else "Unknown"
                        st.warning(f"Gemini response empty/blocked. Reason: {reason}")
                        st.write(f"Gemini response empty/blocked. Reason: {reason}, Attempt: {attempt+1}")
                        if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                        else: return f"ERROR: Blocked - {reason}"
                except Exception as e:
                    st.error(f"Gemini API Error (Attempt {attempt+1}): {e}")
                    st.write(f"Gemini API Error (Attempt {attempt+1}): {e}")
                    if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                    else: return f"ERROR: API call failed - {e}"
            st.write("Gemini max retries reached.")
            return "ERROR: Max retries."
        except Exception as e:
            st.error(f"Gemini Config/Setup Error: {e}")
            st.write(f"Gemini Config/Setup Error: {e}")
            return f"ERROR: Setup - {e}"

    elif "gpt" in model_name:
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not configured.")
            st.write("Error: OPENAI_API_KEY not configured.")
            return "ERROR: API Key missing."
        try:
            st.write(f"Configuring OpenAI with API key: {'*' * (len(OPENAI_API_KEY) - 4) + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'Not Set'}")
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            for attempt in range(retries):
                try:
                    st.write(f"OpenAI API call attempt {attempt + 1}")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=temperature
                    )
                    if response.choices and response.choices[0].message.content:
                        generated_text = response.choices[0].message.content.strip()
                        st.write(f"OpenAI response successful: {generated_text[:100]}...")
                        return generated_text
                    else:
                        st.warning("OpenAI response empty.")
                        st.write(f"OpenAI response empty. Attempt: {attempt+1}")
                        if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                        else: return "ERROR: OpenAI response empty after retries."
                except Exception as e:
                    st.error(f"OpenAI API Error (Attempt {attempt+1}): {e}")
                    st.write(f"OpenAI API Error (Attempt {attempt+1}): {e}")
                    if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                    else: return f"ERROR: API call failed - {e}"
            st.write("OpenAI max retries reached.")
            return "ERROR: Max retries."
        except Exception as e:
            st.error(f"OpenAI Config/Setup Error: {e}")
            st.write(f"OpenAI Config/Setup Error: {e}")
            return f"ERROR: Setup - {e}"

    elif "claude" in model_name:
        if not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY not configured.")
            st.write("Error: ANTHROPIC_API_KEY not configured.")
            return "ERROR: API Key missing."
        try:
            st.write(f"Configuring Anthropic with API key: {'*' * (len(ANTHROPIC_API_KEY) - 4) + ANTHROPIC_API_KEY[-4:] if ANTHROPIC_API_KEY else 'Not Set'}")
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            for attempt in range(retries):
                try:
                    st.write(f"Anthropic API call attempt {attempt + 1}")
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=2000,
                        temperature=temperature,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt_text
                            }
                        ]
                    )
                    if response.content and isinstance(response.content, list) and response.content[0].text:
                        generated_text = response.content[0].text.strip()
                        st.write(f"Anthropic response successful: {generated_text[:100]}...")
                        return generated_text
                    else:
                        st.warning("Anthropic response empty or not in expected format.")
                        st.write(f"Anthropic response empty. Attempt: {attempt+1}, Response: {response}")
                        if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                        else: return "ERROR: Anthropic response empty/invalid after retries."
                except Exception as e:
                    st.error(f"Anthropic API Error (Attempt {attempt+1}): {e}")
                    st.write(f"Anthropic API Error (Attempt {attempt+1}): {e}")
                    if attempt < retries - 1: time.sleep(delay_seconds * (attempt + 1))
                    else: return f"ERROR: API call failed - {e}"
            st.write("Anthropic max retries reached.")
            return "ERROR: Max retries."
        except Exception as e:
            st.error(f"Anthropic Config/Setup Error: {e}")
            st.write(f"Anthropic Config/Setup Error: {e}")
            return f"ERROR: Setup - {e}"
    else:
        st.error(f"Unsupported model provider for model: {model_name}")
        st.write(f"Error: Unsupported model provider for model: {model_name}")
        return "ERROR: Unsupported model provider."

# --- UI LAYOUT ---
st.set_page_config(layout="wide", page_title="AI SEO Content Engine - Full Control")
st.title("🤖 AI-Powered SEO Content Generation Engine (Full Control Panel)")
st.markdown(f"**Empowering Bastian to craft perfect AI content!** (Current Config: *{st.session_state.active_config_name}*)")

if not GEMINI_API_KEY:
    st.error("CRITICAL: Gemini API Key missing. App functionality disabled. Please set it in `.streamlit/secrets.toml` (for local) or as an Environment Variable on your deployment platform (e.g., Vercel).")
    st.stop()

# --- DEFINE TABS ---
tab_main_app, tab_instructions = st.tabs(["⚙️ Main Application", "📖 Instructions & Help"])

# --- INSTRUCTIONS TAB CONTENT ---
with tab_instructions:
    st.header("📖 How to Use the AI SEO Content Engine")
    st.markdown("""
    Welcome, Bastian! This tool is designed to give you full control over generating SEO-optimized content using AI.
    Here's a breakdown of how to use it effectively:
    """)

    st.subheader("1. Initial Setup (First Time & Key Management)")
    st.markdown("""
    - **API Keys:** This application requires API Keys for the selected LLM provider.
        - **Gemini:** `GEMINI_API_KEY`
        - **OpenAI:** `OPENAI_API_KEY`
        - **Anthropic:** `ANTHROPIC_API_KEY`
    - **Local Use:** Ensure you have a `.streamlit/secrets.toml` file in the same directory as `app.py` with your keys, e.g.:
      ```toml
      GEMINI_API_KEY = "YOUR_GEMINI_KEY"
      OPENAI_API_KEY = "YOUR_OPENAI_KEY"
      ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_KEY"
      ```
    - **Deployed (e.g., Vercel):** The API keys must be set as environment variables (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) on the hosting platform.
    - **If an API key for the selected model provider is missing, the app won't be able to generate content with that provider.**
    """)

    st.subheader("2. Configuration Management (Sidebar 🛠️)")
    st.markdown("""
    - **Save Configuration:**
        - After you've set up your prompts, link lists, topics, and other settings perfectly for a specific task (e.g., "/hire/" cluster), give it a name in "Configuration Name (for saving)" and click "💾 Save Current Configuration."
        - This will trigger a **download of a JSON file** to your computer. Keep this file safe!
        - The "Current Config" name at the top of the page will update.
    - **Load Configuration:**
        - To restore a previously saved setup, use the "📂 Load Configuration from JSON File" uploader to select your saved JSON file.
        - The app will populate all fields with the loaded settings. You might see the page refresh.
    - **Why this is important:** This allows you to have different "setups" for different content types without re-entering everything.
    """)

    st.subheader("3. Global Settings (Sidebar 🛠️)")
    st.markdown("""
    - **Select Model:** Choose the AI model from the dropdown. This list includes models from Gemini, OpenAI, and Anthropic.
    - **LLM Temperature:** Controls AI creativity.
        - `0.0 - 0.3`: More factual, predictable, less creative. Good for constrained tasks.
        - `0.4 - 0.7`: Balanced (default is `0.6`).
        - `0.8 - 1.0`: More creative, diverse, but higher risk of unexpected or off-topic output.
    """)

    st.subheader("4. Inputs & Contextual Data (Main Area - Left Column 📝)")
    st.markdown("""
    - **Topics & Keywords:**
        - **Upload CSV:** The best way for multiple topics. Create a CSV file with columns: `topic_input`, `primary_keyword`, `secondary_keywords`. (Secondary keywords should be a comma-separated list in their cell).
        - **Data Editor:** After uploading or if using defaults, you can directly edit the topics, primary keywords, and secondary keywords in the table. You can also add or delete rows.
    - **Approved Link Lists:**
        - **CRITICAL for link quality!** Provide lists of URLs the AI is *allowed* to use.
        - Format: `https://full.url/path: Brief description of the page` (one entry per line). The AI uses the description to understand context.
    - **Core Contextual Data:**
        - **Brand Guidelines:** Tell the AI about Workstream's voice, tone, audience.
        - **SEO Best Practices:** Remind the AI of key SEO principles.
    - **Link Generation Targets:** Specify how many internal/external links the `main_text_html` prompt should aim for.
    """)

    st.subheader("5. Prompt Engineering Zone (Main Area - Right Column 🔧)")
    st.markdown("""
    - This is where you **directly instruct the AI.** Each content piece (Title, Meta, Main Text, etc.) has its own prompt.
    - **Click on an expander** (e.g., "Edit Prompt for: `main_text_html`") to view and edit the prompt.
    - **Placeholders:** Prompts use `[PLACEHOLDERS]` (e.g., `[TOPIC_INPUT]`, `[PRIMARY_KEYWORD]`, `[APPROVED_INTERNAL_LINKS_TEXT]`). The system automatically fills these with the relevant data from your inputs before sending to the AI. **Do not remove or change the square brackets of these placeholders.**
    - **Editing Prompts:**
        - Be specific and clear in your instructions.
        - If the AI makes mistakes (e.g., markdown fences ` ```html ` around HTML output), modify the prompt to explicitly forbid it (e.g., "Output ONLY raw HTML. DO NOT use markdown fences.").
        - Iterate! Small changes can have big impacts.
    - **Your edits to prompts are part of the "Current Configuration"** and will be saved if you use "Save Current Configuration."
    """)

    st.subheader("6. Execute Generation (Bottom Buttons 🚀)")
    st.markdown("""
    - **✨ Generate Content for ALL Topics ✨:** Processes all topics currently loaded in the "Topics & Keywords" editor/CSV.
    - **🧪 Test with FIRST Topic Only:** Ideal for quickly testing prompt changes. Processes only the first topic in the list.
    - **Progress:** A progress bar and status messages will appear.
    """)

    st.subheader("7. Review Output & Download 📊")
    st.markdown("""
    - Generated content appears in a table. Review it carefully.
        - **Check Links:** Are `found_internal_links_in_html` and `found_external_links_in_html` showing the correct URLs from your approved lists? Manually verify the actual links in the `main_text_html`.
        - **Check Formatting:** Is the HTML clean? (e.g., no unwanted markdown).
        - **Check Quality:** Readability, tone, accuracy, SEO.
    - **📥 Download All Results as CSV:** Saves the generated content to a CSV file for HubSpot import or further review.
    """)

    st.subheader("General Workflow for Refinement:")
    st.markdown("""
    1. **Load a base configuration** or start with defaults.
    2. **Prepare your Topics & Keywords** (upload CSV or edit in table).
    3. **Verify Link Lists & Contextual Data.**
    4. **Tweak a specific prompt** in the "Prompt Engineering Zone."
    5. Use **"🧪 Test with FIRST Topic Only"** to see the impact of your prompt change.
    6. **Review the single output.** If good, proceed. If not, go back to step 4.
    7. Once happy with individual prompt elements, run **"✨ Generate Content for ALL Topics ✨"** for a larger batch.
    8. **Thoroughly review the full output CSV.**
    9. **Save your improved configuration** often!
    """)
    st.markdown("---")
    st.success("Happy Content Generating!")


# --- MAIN APPLICATION TAB CONTENT ---
with tab_main_app:
    # --- Sidebar for Configuration Management & Global Settings (Copied from previous version) ---
    with st.sidebar:
        st.header("🛠️ Engine Configuration")
        st.caption("Manage global settings and save/load your work.")

        st.subheader("Save/Load Configuration")
        config_name = st.text_input("Configuration Name (for saving):", value=f"config_{st.session_state.active_config_name.replace('.json','').split('_')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}", key="config_save_name")
        
        if st.button("💾 Save Current Configuration", key="save_config_button"):
            config_to_save = get_current_config_dict()
            try:
                json_str = json.dumps(config_to_save, indent=2)
                st.download_button(
                    label="📥 Download Config JSON",
                    data=json_str,
                    file_name=f"{config_name}.json",
                    mime="application/json",
                    key="download_config_json_button" # Unique key
                )
                st.success(f"Configuration ready for download as '{config_name}.json'.")
                st.session_state.active_config_name = config_name # Update active config name
            except Exception as e:
                st.error(f"Error preparing config for download: {e}")

        uploaded_config_file = st.file_uploader("📂 Load Configuration from JSON File", type=["json"], key="config_uploader")
        
        # Initialize flag in session state if it doesn't exist
        if 'config_just_processed' not in st.session_state:
            st.session_state.config_just_processed = False

        if uploaded_config_file is not None and not st.session_state.config_just_processed:
            try:
                config_data = json.load(uploaded_config_file)
                load_config_from_dict(config_data) # This function now updates session_state
                st.session_state.active_config_name = uploaded_config_file.name
                st.success(f"Configuration '{uploaded_config_file.name}' loaded! UI elements reflecting new settings.")
                st.session_state.config_just_processed = True # Set flag to True after processing
                time.sleep(0.1) # Small delay might sometimes help with state updates before rerun
                st.rerun()
            except Exception as e:
                st.error(f"Error loading configuration file: {e}")
                st.session_state.config_just_processed = False # Reset flag on error to allow re-attempt
        elif uploaded_config_file is None and st.session_state.config_just_processed:
            # Reset the flag if no file is currently uploaded, allowing next upload to be processed
            st.session_state.config_just_processed = False
        
        st.markdown("---")
        # Directly update session state from widget
        st.session_state.model_name = st.selectbox(
            "Select Model:", # Changed label
            ["gemini-2.5-pro-exp-03-25", "gpt-4.1", "claude-3-7-sonnet-20250219"], # Updated model list
            index=["gemini-2.5-pro-exp-03-25", "gpt-4.1", "claude-3-7-sonnet-20250219"].index(st.session_state.model_name if st.session_state.model_name in ["gemini-2.5-pro-exp-03-25", "gpt-4.1", "claude-3-7-sonnet-20250219"] else "gemini-2.5-pro-exp-03-25"), # Updated index logic
            key="model_selector"
        )
        st.session_state.llm_temperature = st.slider(
            "LLM Temperature (Creativity):", 0.0, 1.0, st.session_state.llm_temperature, 0.05,
            help="Lower = more focused. Higher = more creative.", key="temp_slider"
        )
        st.markdown("---")
        st.info("Remember to save your configuration if you make significant changes!")


    # --- Main Area Layout: Two Columns for Inputs and Prompts (Copied from previous version) ---
    col_inputs_context, col_prompts_editor = st.columns([2, 3])

    with col_inputs_context:
        st.header("📝 Inputs & Contextual Data")
        st.caption("Define the topics, keywords, and guiding information for the AI.")

        st.subheader("🎯 Topics & Keywords")
        st.markdown("Upload a CSV file (`topic_input`, `primary_keyword`, `secondary_keywords`) or edit below.")
        
        uploaded_topics_csv = st.file_uploader("Upload Topics CSV", type="csv", key="topics_csv_uploader_main_tab") # Unique key
        if uploaded_topics_csv is not None:
            try:
                df = pd.read_csv(uploaded_topics_csv)
                for col in ['topic_input', 'primary_keyword', 'secondary_keywords']:
                    if col not in df.columns: df[col] = "" 
                st.session_state.topics_df = df[['topic_input', 'primary_keyword', 'secondary_keywords']]
                st.session_state.uploaded_topics_filename = uploaded_topics_csv.name
                st.success(f"Loaded '{uploaded_topics_csv.name}' successfully into editor below.")
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")
                
        st.caption(f"Editing data for: {st.session_state.uploaded_topics_filename or 'current session / defaults'}")
        if not st.session_state.topics_df.empty or st.session_state.uploaded_topics_filename: # Show editor if df has data or a file was just uploaded
            edited_df = st.data_editor(
                st.session_state.topics_df,
                num_rows="dynamic", 
                key="topics_data_editor_main_tab", # Unique key
                use_container_width=True,
                column_config={ # Optional: Make columns more user-friendly
                    "topic_input": st.column_config.TextColumn("Topic / Page Slug Target", required=True),
                    "primary_keyword": st.column_config.TextColumn("Primary Keyword"),
                    "secondary_keywords": st.column_config.TextColumn("Secondary Keywords (comma-sep)"),
                }
            )
            if not edited_df.equals(st.session_state.topics_df):
                 st.session_state.topics_df = edited_df
        else:
            st.info("No topics loaded. Upload a CSV above to populate the editor, or a default set may load.")


        st.subheader("🔗 Approved Link Lists")
        st.session_state.approved_internal_links = st.text_area("Workstream Internal URLs (URL: Description per line):", value=st.session_state.approved_internal_links, height=120, key="internal_links_text_area")
        st.session_state.approved_external_links = st.text_area("Authoritative External URLs (URL: Description per line):", value=st.session_state.approved_external_links, height=100, key="external_links_text_area")

        st.subheader("📜 Core Contextual Data")
        st.session_state.brand_guidelines = st.text_area("Workstream Brand Guidelines:", value=st.session_state.brand_guidelines, height=120, key="brand_guidelines_text_area")
        st.session_state.seo_summary = st.text_area("SEO Best Practices Summary:", value=st.session_state.seo_summary, height=120, key="seo_summary_text_area")

        st.subheader("⛓️ Link Generation Targets")
        st.session_state.target_internal_links = st.number_input("Target # Internal Links:", 0, 30, st.session_state.target_internal_links, key="target_int_links_input")
        st.session_state.target_external_links = st.number_input("Target # External Links:", 0, 15, st.session_state.target_external_links, key="target_ext_links_input")

    with col_prompts_editor:
        st.header("🔧 Prompt Engineering Zone")
        st.warning("These prompts are the AI's direct instructions. Edit carefully! Note `[PLACEHOLDERS]` which are filled by the system.")
        
        for prompt_key_iter, default_prompt_text_iter in PROMPT_TEMPLATES_DEFAULTS.items():
            current_prompt_val_iter = st.session_state.editable_prompts.get(prompt_key_iter, default_prompt_text_iter)
            with st.expander(f"Edit Prompt for: `{prompt_key_iter}`", expanded=(prompt_key_iter == "main_text_html")):
                edited_prompt_iter = st.text_area(
                    f"Instructions for `{prompt_key_iter}`:",
                    value=current_prompt_val_iter,
                    height=250,
                    key=f"prompt_editor_area_main_tab_{prompt_key_iter}" # Unique key
                )
                if edited_prompt_iter != st.session_state.editable_prompts.get(prompt_key_iter):
                    st.session_state.editable_prompts[prompt_key_iter] = edited_prompt_iter

    # --- GENERATION BUTTONS & OUTPUT AREA ---
    st.markdown("---")
    st.header("🚀 Execute Generation & Review Output")

    col_run_all, col_run_single = st.columns(2)
    with col_run_all:
        if st.button("✨ Generate Content for ALL Topics ✨", type="primary", use_container_width=True, help="Processes all topics from the editor/uploaded CSV.", key="gen_all_button"):
            st.session_state.run_mode = "all"
            st.session_state.trigger_generation = True 

    with col_run_single:
        if st.button("🧪 Test with FIRST Topic Only", use_container_width=True, help="Quickly test current settings using only the first topic in the list.", key="gen_first_button"):
            st.session_state.run_mode = "first_only"
            st.session_state.trigger_generation = True

    if 'trigger_generation' in st.session_state and st.session_state.trigger_generation:
        st.session_state.trigger_generation = False # Reset trigger
        
        topics_df_to_process = st.session_state.topics_df.copy() # Use a copy to avoid modifying session state during iteration
        if topics_df_to_process.empty:
            st.warning("No topics to process. Please upload a CSV or add topics in the editor.")
            st.stop()

        if st.session_state.run_mode == "first_only":
            topics_df_to_process = topics_df_to_process.head(1)
            if topics_df_to_process.empty:
                st.warning("No topics available to test with 'First Topic Only'. Add topics to the editor."); st.stop()
            st.info(f"🧪 Test Mode: Processing only the first topic: '{topics_df_to_process.iloc[0].get('topic_input', 'N/A')}'")
        
        st.info(f"🚀 Starting content generation for {len(topics_df_to_process)} topic(s) using {st.session_state.model_name}...")
        all_generated_data_list = []
        progress_bar = st.progress(0.0)
        status_text_area = st.empty()
        current_prompts = st.session_state.editable_prompts

        for i, topic_row in enumerate(topics_df_to_process.to_dict(orient='records')):
            current_progress = (i + 1) / len(topics_df_to_process)
            topic_input_val = topic_row.get('topic_input', 'N/A')
            primary_keyword_val = topic_row.get('primary_keyword', '')
            secondary_keywords_val = topic_row.get('secondary_keywords', '')
            status_text_area.info(f"🔄 Processing: **{topic_input_val}** ({i+1} of {len(topics_df_to_process)})...")
            output_row = {"topic_input": topic_input_val,"primary_keyword": primary_keyword_val,"secondary_keywords": secondary_keywords_val}
            for field_to_gen in ["page_title", "meta_description", "h1_tag", "subtitle", "alt_text", "main_text_html"]:
                prompt_template = current_prompts.get(field_to_gen)
                if not prompt_template:
                    st.warning(f"Prompt for '{field_to_gen}' missing for '{topic_input_val}'."); output_row[field_to_gen] = "ERROR: No Prompt"; continue
                status_text_area.info(f"🔄 Generating `{field_to_gen}` for: **{topic_input_val}**...")
                fmt_prompt = prompt_template
                fmt_prompt = fmt_prompt.replace("[TOPIC_INPUT]", topic_input_val)
                fmt_prompt = fmt_prompt.replace("[PRIMARY_KEYWORD]", primary_keyword_val)
                fmt_prompt = fmt_prompt.replace("[SECONDARY_KEYWORDS_LIST]", secondary_keywords_val)
                fmt_prompt = fmt_prompt.replace("[WORKSTREAM_BRAND_GUIDELINES]", st.session_state.brand_guidelines)
                fmt_prompt = fmt_prompt.replace("[SEO_BEST_PRACTICES_SUMMARY]", st.session_state.seo_summary)
                fmt_prompt = fmt_prompt.replace("[APPROVED_INTERNAL_LINKS_TEXT]", st.session_state.approved_internal_links)
                fmt_prompt = fmt_prompt.replace("[APPROVED_EXTERNAL_LINKS_TEXT]", st.session_state.approved_external_links)
                fmt_prompt = fmt_prompt.replace("[TARGET_NUMBER_INTERNAL_LINKS]", str(st.session_state.target_internal_links))
                fmt_prompt = fmt_prompt.replace("[TARGET_NUMBER_EXTERNAL_LINKS]", str(st.session_state.target_external_links))
                generated_val = generate_content(fmt_prompt, st.session_state.model_name, st.session_state.llm_temperature)
                output_row[field_to_gen] = generated_val
                if field_to_gen == "main_text_html" and isinstance(generated_val, str):
                    # Corrected link detection:
                    # Extract the full URL part (before the first colon) from the approved lists
                    approved_internal_urls_full = []
                    for line in st.session_state.approved_internal_links.splitlines():
                        if line.strip() and ":" in line:
                            approved_internal_urls_full.append(line.split(":", 1)[0].strip())
                    
                    approved_external_urls_full = []
                    for line in st.session_state.approved_external_links.splitlines():
                        if line.strip() and ":" in line:
                            approved_external_urls_full.append(line.split(":", 1)[0].strip())

                    output_row["found_internal_links_in_html"] = " | ".join(
                        [url for url in approved_internal_urls_full if url in generated_val]
                    )
                    output_row["found_external_links_in_html"] = " | ".join(
                        [url for url in approved_external_urls_full if url in generated_val]
                    )
                time.sleep(0.05) 
            all_generated_data_list.append(output_row)
            progress_bar.progress(current_progress)
        status_text_area.success(f"✅ Content generation complete for {len(topics_df_to_process)} topic(s)!")
        if all_generated_data_list:
            results_final_df = pd.DataFrame(all_generated_data_list, columns=CSV_COLUMN_HEADERS)
            st.subheader("📊 Generated Content Results")
            st.dataframe(results_final_df, height=600, use_container_width=True)
            @st.cache_data
            def convert_df_to_csv_bytes_final(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
            csv_final_bytes = convert_df_to_csv_bytes_final(results_final_df)
            st.download_button(label="📥 Download All Results as CSV",data=csv_final_bytes,file_name=f"ai_seo_content_{st.session_state.model_name.replace('/', '-')}_{time.strftime('%Y%m%d-%H%M%S')}.csv",mime="text/csv")
        else: st.warning("No data was generated. Check inputs and configurations.")


# --- Footer ---
st.markdown("---")
st.caption("AI SEO Content Engine for Workstream | Built for iterative refinement. Always review outputs carefully.") 
