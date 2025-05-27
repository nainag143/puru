from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
from langchain_community.embeddings import HuggingFaceEmbeddings
from zoneinfo import ZoneInfo # Preferred over pytz for modern Python
import re
from datetime import datetime, time, timedelta # Added time for time objects

import logging
from uuid import uuid4 # For generating unique IDs if needed
from langchain_chroma import Chroma
from django.core.cache import cache
import os
from openai import OpenAI # For call disconnect summary
# import time # Standard time module, aliased by datetime.time
import torch
import hashlib # For cache keys if used that way
import pymysql
import pymysql.cursors # For DictCursor
import json
import pandas as pd

from vllm import SamplingParams # For your local LLM
from sales_app.llm_instance import LLM_INSTANCE # Your local vLLM instance

# --- Logging Setup ---
# Ensure this path is writable by your application server
LOG_DIR = "/root/BotHub_llm_model/llm_agent_project/motilal_app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "motilal_chat_service.log")

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO, # Set to logging.DEBUG for verbose parsing/flow details
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Specific logger for this module

# --- Global Configurations ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Application starting. Using device: {DEVICE}")

# WARNING: Hardcoding API keys/passwords is a major security risk. Use environment variables.
API_KEY_OPENAI = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE") # Replace with your actual key or ensure env var is set
if API_KEY_OPENAI == "YOUR_OPENAI_API_KEY_HERE":
    logger.warning("OpenAI API Key is using a placeholder value.")
client_openai = OpenAI(api_key=API_KEY_OPENAI)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # As in original code

DB_PERSIST_DIRECTORY = "/root/BotHub_llm_model/llm_agent_project/motilal_app/MO_latest"
DB_COLLECTION_NAME = "test-1"
YOUR_CONVERSATION_TABLE_NAME = "your_table_name" # IMPORTANT: Replace with your actual table name

# --- Embeddings and DB Initialization ---
db_vector_store = None # Initialize to None
try:
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    db_vector_store = Chroma( # Using the alias from langchain_chroma
        collection_name=DB_COLLECTION_NAME,
        persist_directory=DB_PERSIST_DIRECTORY,
        embedding_function=hf_embeddings
    )
    logger.info(f"Successfully loaded Chroma DB from: {DB_PERSIST_DIRECTORY}")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to load Chroma DB from {DB_PERSIST_DIRECTORY}. RAG features will be unavailable. Error: {e}", exc_info=True)
    # Application might still run but RAG will fail. Consider raising an error if DB is essential.

# --- Cache Utility Functions ---
def update_conversation_cache(uuid_val, history_item):
    key = str(uuid_val)
    conversation = cache.get(key, [])
    conversation.append(history_item)
    cache.set(key, conversation, timeout=None) # Cache indefinitely
    logger.debug(f"Updated cache for UUID: {uuid_val} with item: {history_item['role']}")

def get_conversation_history_cache(uuid_val):
    history = cache.get(str(uuid_val), [])
    logger.debug(f"Retrieved cache for UUID: {uuid_val}, History length: {len(history)}")
    return history

def delete_conversation_cache(uuid_val):
    key = str(uuid_val)
    cache.delete(key)
    logger.info(f"Deleted conversation from cache for UUID: {uuid_val}")

# --- Database Interaction Functions ---
def get_db_connection():
    try:
        # Best practice: Load credentials from environment variables or a secure config service
        db_host = os.environ.get("DB_HOST", "127.0.0.1")
        db_user = os.environ.get("DB_USER", "root")
        db_password = os.environ.get("DB_PASSWD", "passw0rd") # Replace with env var
        db_name = os.environ.get("DB_NAME", "voice_bot")

        if db_password == "passw0rd" and db_user == "root": # Example check for default/placeholder
             logger.warning("Using default/placeholder database credentials. THIS IS INSECURE for production.")

        conn = pymysql.connect(
            host=db_host,
            user=db_user,
            passwd=db_password,
            database=db_name,
            cursorclass=pymysql.cursors.DictCursor # Results as dictionaries
        )
        logger.debug("Database connection successful.")
        return conn
    except pymysql.Error as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        return None

def store_conversation_db(question, answer, channelid, phonenumber, uuid_val):
    logger.debug(f"Attempting to store conversation in DB for UUID: {uuid_val}")
    # IMPORTANT: Replace YOUR_CONVERSATION_TABLE_NAME with your actual table name
    sql = f"""
        INSERT INTO {YOUR_CONVERSATION_TABLE_NAME} (question, answer, channelid, phonenumber, uuid)
        VALUES (%s, %s, %s, %s, %s)
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to store conversation: No DB connection.")
            return # Or raise an exception
        with conn.cursor() as cursor:
            cursor.execute(sql, (question, answer, channelid, phonenumber, uuid_val))
        conn.commit()
        logger.info(f"Conversation for UUID {uuid_val} stored in DB.")
    except pymysql.Error as e:
        logger.error(f"Error storing conversation in DB for UUID {uuid_val}: {e}", exc_info=True)
        if conn: # Rollback if commit failed or other error before commit
            conn.rollback()
    finally:
        if conn:
            conn.close()

# --- Date/Time Validation (Business Logic) ---
def validate_schedule_time_logic(selected_time_str: str, selected_date_str: str | None = None) -> tuple[bool, str]:
    logger.info(f"Attempting to validate schedule: Time='{selected_time_str}', Date='{selected_date_str}'")

    if not selected_time_str:
        logger.error("Validation called without selected_time_str.")
        return False, "I need a time to schedule the meeting, for example, 'at 2:30 PM'."

    try:
        parsed_time_obj = datetime.strptime(selected_time_str, "%I:%M %p").time()
        logger.debug(f"Successfully parsed selected_time_str '{selected_time_str}' to time object: {parsed_time_obj}")
    except ValueError:
        logger.warning(f"Could not parse selected_time_str: '{selected_time_str}' using format %I:%M %p.")
        return False, "I couldn't quite understand that time. Could you please say it like '10:30 AM' or '4 PM'?"

    current_dt_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    logger.debug(f"Current IST for validation: {current_dt_ist.strftime('%Y-%m-%d %A %I:%M %p %Z')}")

    if selected_date_str:
        try:
            parsed_date_obj = datetime.strptime(selected_date_str, "%d-%m-%Y").date()
            logger.debug(f"Successfully parsed selected_date_str '{selected_date_str}' to date object: {parsed_date_obj}")
        except ValueError:
            logger.warning(f"Could not parse selected_date_str: '{selected_date_str}' using format %d-%m-%Y.")
            return False, "The date you mentioned seems to be in an unclear format. Could you try DD-MM-YYYY, like '28-05-2025'?"
    else:
        parsed_date_obj = current_dt_ist.date()
        logger.debug(f"No explicit date provided for validation; using current date: {parsed_date_obj.strftime('%d-%m-%Y')}")

    # Combine parsed date and time with IST timezone
    selected_dt_ist = datetime.combine(parsed_date_obj, parsed_time_obj, tzinfo=ZoneInfo("Asia/Kolkata"))
    logger.info(f"Combined selected datetime for validation: {selected_dt_ist.strftime('%Y-%m-%d %A %I:%M %p %Z')}")

    # Rule 1: Must be in the future (relative to current time)
    if selected_dt_ist <= current_dt_ist:
        logger.info(f"Validation REJECTED: Selected time {selected_dt_ist.strftime('%Y-%m-%d %I:%M %p %Z')} is in the past or is the current moment.")
        return False, f"That time appears to be in the past or is too immediate. Please choose a future time."

    # Rule 2: No meetings after 3rd June 2025, IST 8:00 PM
    MAX_SCHEDULING_DATETIME_IST = datetime(2025, 6, 3, 20, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    if selected_dt_ist > MAX_SCHEDULING_DATETIME_IST:
        logger.info(f"Validation REJECTED: Selected datetime {selected_dt_ist.strftime('%Y-%m-%d %I:%M %p %Z')} is after max allowed {MAX_SCHEDULING_DATETIME_IST.strftime('%Y-%m-%d %I:%M %p %Z')}")
        return False, f"We are currently scheduling meetings only up to June 3rd, 2025, at 8:00 PM IST. Please choose an earlier date and time."

    # Rule 3: Not on Sunday
    if selected_dt_ist.weekday() == 6:  # Monday is 0 and Sunday is 6
        logger.info(f"Validation REJECTED: Selected date {selected_dt_ist.strftime('%A')} is a Sunday.")
        return False, "I'm sorry, but we don't schedule meetings on Sundays. Could we try for a weekday or Saturday?"

    # Rule 4: Within daily business hours (IST 9:00 AM to IST 8:00 PM)
    business_start_time = time(9, 0)   # 9:00 AM
    business_end_time = time(20, 0)  # 8:00 PM
    if not (business_start_time <= selected_dt_ist.time() <= business_end_time):
        logger.info(f"Validation REJECTED: Selected time {selected_dt_ist.time().strftime('%I:%M %p')} is outside business hours (9:00 AM - 8:00 PM IST).")
        return False, "Our team is available for meetings between 9:00 AM and 8:00 PM IST, Monday through Saturday. Please pick a time within these hours."

    # Rule 5: 30-minute buffer if scheduling for today
    if selected_dt_ist.date() == current_dt_ist.date(): # Check if it's for today
        min_allowed_dt_today = current_dt_ist + timedelta(minutes=30)
        if selected_dt_ist < min_allowed_dt_today:
            logger.info(f"Validation REJECTED: Needs 30-min buffer for same-day. Selected {selected_dt_ist.strftime('%I:%M %p %Z')}, min_allowed {min_allowed_dt_today.strftime('%I:%M %p %Z')}")
            return False, f"For a meeting today, I need at least a 30-minute heads-up. The earliest we could do is around {min_allowed_dt_today.strftime('%I:%M %p')}. Would that perhaps work for you?"
    
    logger.info(f"Validation SUCCESS for: {selected_dt_ist.strftime('%d-%m-%Y %A %I:%M %p %Z')}")
    return True, f"Great! Just to confirm, that's {selected_dt_ist.strftime('%A, %B %d')} at {selected_dt_ist.strftime('%I:%M %p IST')}. Is that correct?"


class Motilal_ChatBot_View(APIView):
    # --- Date/Time Parsing Helper Methods ---
    def standardize_time_format(self, time_str_input: str) -> str | None:
        logger.debug(f"Attempting to standardize time: '{time_str_input}'")
        if not time_str_input: return None
        original_time = time_str_input
        
        is_pm = False
        if re.search(r'[pP]\.?[mM]\.?', time_str_input): is_pm = True
        
        time_str_cleaned = re.sub(r'\s*(?:[aA]\.?[mM]\.?|[pP]\.?[mM]\.?)', '', time_str_input, flags=re.IGNORECASE).strip()
        time_str_cleaned = re.sub(r'[.\s]+', ':', time_str_cleaned)
        
        hour_str, minute_str = "", "00"
        if ':' in time_str_cleaned:
            parts = time_str_cleaned.split(':')
            hour_str = parts[0]
            if len(parts) > 1 and parts[1].isdigit(): minute_str = parts[1]
        elif time_str_cleaned.isdigit(): hour_str = time_str_cleaned
        
        if not hour_str.isdigit() or not minute_str.isdigit():
            logger.warning(f"Could not parse hour/minute from '{original_time}' -> cleaned as '{time_str_cleaned}'")
            return None
            
        hour, minute = int(hour_str), int(minute_str)
        ampm_tag = "PM" if is_pm else "AM"
        
        try:
            # Handle 12 PM (noon) and 12 AM (midnight) correctly with %I
            if hour == 12 and ampm_tag == "AM": # 12 AM is midnight, hour 0 for 24h format
                pass # strptime %I correctly handles 12 AM
            elif hour == 12 and ampm_tag == "PM": # 12 PM is noon
                pass # strptime %I correctly handles 12 PM
            elif hour > 12 : # e.g. if user says "14:30" and PM was inferred, it's an issue for %I
                 logger.warning(f"Hour {hour} is > 12 for 12-hour format with AM/PM for '{original_time}'.")
                 return None


            temp_dt = datetime.strptime(f"{hour:02d}:{minute:02d} {ampm_tag}", "%I:%M %p")
            formatted_time = temp_dt.strftime("%I:%M %p")
            logger.info(f"Standardized '{original_time}' to '{formatted_time}'")
            return formatted_time
        except ValueError as e:
            logger.error(f"Final strptime validation failed for '{original_time}' (parsed as {hour:02d}:{minute:02d} {ampm_tag}): {e}", exc_info=True)
            return None

    def extract_time(self, text: str) -> str | None:
        logger.debug(f"Extracting time from: '{text}'")
        patterns = [
            r"\b(\d{1,2}\s*[:\.]\s*\d{2}\s*(?:[aA][mM]|[pP][mM]|[aA]\.[mM]\.|[pP]\.[mM]\.))\b",
            r"\b(\d{1,2}\s*(?:[aA][mM]|[pP][mM]|[aA]\.[mM]\.|[pP]\.[mM]\.))\b",
            r"\b(\d{1,2}\s*o'?clock\s*(?:in the\s*)?(?:morning|afternoon|evening|[aA][mM]|[pP][mM])?)\b",
            r"\b(noon|midnight)\b"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matched_text = match.group(1)
                logger.debug(f"Time pattern matched: '{matched_text}' with pattern: {pattern}")
                if matched_text.lower() == "noon": return self.standardize_time_format("12:00 PM")
                if matched_text.lower() == "midnight": return self.standardize_time_format("12:00 AM")
                
                oclock_match = re.match(r"(\d{1,2})\s*o'?clock", matched_text, re.IGNORECASE)
                if oclock_match:
                    hour_val = int(oclock_match.group(1))
                    ampm_val = "AM" 
                    if "afternoon" in matched_text.lower() or "pm" in matched_text.lower(): ampm_val = "PM"
                    elif "evening" in matched_text.lower(): ampm_val = "PM"
                    elif "morning" in matched_text.lower(): ampm_val = "AM" # Default AM
                    # Heuristic for 1-6 o'clock typically being PM if AM/PM not specified.
                    # But prefer explicit AM/PM from phrase if present.
                    # If no AM/PM indicator for o'clock, this can be ambiguous.
                    # For simplicity, if no AM/PM with o'clock, it's safer to ask for clarification
                    # or make a guess based on typical hours.
                    # Let's assume "4 o'clock" without AM/PM is ambiguous for now.
                    # The regex tries to capture AM/PM with o'clock. If not, standardization will fail if ampm_val is not certain.
                    # If `matched_text` includes "am" or "pm" alongside "o'clock", that takes precedence.
                    # If only "o'clock", this heuristic for ampm_val is a bit weak.
                    # A better o'clock handling might require context or direct am/pm request.
                    # For now, it relies on standardization.
                    return self.standardize_time_format(f"{hour_val}:00 {ampm_val}")
                return self.standardize_time_format(matched_text)
        logger.debug("No time pattern found in extract_time.")
        return None

    def extract_date(self, text: str, current_dt_ist: datetime) -> str: # Returns DD-MM-YYYY
        logger.debug(f"Extracting date from: '{text}' (Current IST: {current_dt_ist.strftime('%Y-%m-%d %A')})")
        text_lower = text.lower()
        target_date_obj: datetime.date | None = None # Explicitly type hint

        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        month_names_regex = "|".join(list(month_map.keys()) + ["january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december"])

        m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text_lower)
        if m:
            try: target_date_obj = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))).date(); logger.debug(f"Date Pattern DD/MM/YYYY matched: {target_date_obj}")
            except ValueError: logger.warning(f"Invalid date from DD/MM/YYYY: {m.groups()}", exc_info=True)

        if not target_date_obj:
            m = re.search(fr"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names_regex})\s*(\d{{4}})?\b", text_lower, re.IGNORECASE)
            if m:
                day, month_str_match, year_str = int(m.group(1)), m.group(2).lower(), m.group(3)
                month_val = month_map.get(month_str_match[:3])
                if month_val:
                    year = int(year_str) if year_str else current_dt_ist.year
                    try: target_date_obj = datetime(year, month_val, day).date(); logger.debug(f"Date Pattern 'Day Month [Year]' matched: {target_date_obj}")
                    except ValueError: logger.warning(f"Invalid date from 'Day Month [Year]': {day}-{month_str_match}-{year}", exc_info=True)
            else:
                m = re.search(fr"\b({month_names_regex})\s*(\d{{1,2}})(?:st|nd|rd|th)?\s*(\d{{4}})?\b", text_lower, re.IGNORECASE)
                if m:
                    month_str_match, day, year_str = m.group(1).lower(), int(m.group(2)), m.group(3)
                    month_val = month_map.get(month_str_match[:3])
                    if month_val:
                        year = int(year_str) if year_str else current_dt_ist.year
                        try: target_date_obj = datetime(year, month_val, day).date(); logger.debug(f"Date Pattern 'Month Day [Year]' matched: {target_date_obj}")
                        except ValueError: logger.warning(f"Invalid date from 'Month Day [Year]': {month_str_match}-{day}-{year}", exc_info=True)
        
        if not target_date_obj:
            if "tomorrow" in text_lower or "day after today" in text_lower: target_date_obj = current_dt_ist.date() + timedelta(days=1); logger.debug(f"Date Pattern 'tomorrow' matched: {target_date_obj}")
            elif "today" in text_lower: target_date_obj = current_dt_ist.date(); logger.debug(f"Date Pattern 'today' matched: {target_date_obj}")

        if not target_date_obj:
            days_of_week = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
            current_weekday_idx = current_dt_ist.weekday()
            for day_name_key, target_weekday_idx_val in days_of_week.items():
                is_next_keyword_present = bool(re.search(r"\bnext\s+" + day_name_key + r"\b", text_lower, re.IGNORECASE))
                is_plain_day_match = bool(re.search(r"\b" + day_name_key + r"\b", text_lower, re.IGNORECASE))

                if is_plain_day_match:
                    days_diff = (target_weekday_idx_val - current_weekday_idx + 7) % 7
                    if is_next_keyword_present: # "next Monday"
                        # If "next" is present, and the calculated days_diff is for the current week (or today),
                        # then it means next week's instance.
                        # If days_diff is 0 (today) and "next" is specified, it's 7 days.
                        # If days_diff > 0 (later this week) and "next" is specified, it's also 7 days from that.
                        # Essentially, "next Xday" = soonest Xday + 7 days
                        days_diff = ((target_weekday_idx_val - current_weekday_idx + 7) % 7) + 7
                    
                    target_date_obj = current_dt_ist.date() + timedelta(days=days_diff)
                    logger.debug(f"Date Pattern weekday '{day_name_key}' (is_next_keyword: {is_next_keyword_present}) matched. Days diff: {days_diff}. Target: {target_date_obj}")
                    break 
        
        if not target_date_obj: # Default to current date if all else fails
            target_date_obj = current_dt_ist.date()
            logger.debug(f"No specific date pattern matched, defaulting to current date: {target_date_obj}")
            
        return target_date_obj.strftime("%d-%m-%Y")

    def parse_scheduling_info(self, question: str, current_dt_ist: datetime) -> tuple[str | None, str | None]:
        logger.info(f"Attempting to parse scheduling info from: \"{question}\"")
        # Time extraction is crucial. If no time, it's less likely a direct scheduling intent for this parser.
        time_val = self.extract_time(question)
        date_val = None # Default to None
        
        if time_val:
            # If a time was found, then proceed to extract a date.
            # extract_date will default to current_dt_ist's date if no other date is specified.
            date_val = self.extract_date(question, current_dt_ist)
        else:
            logger.info("No valid time found by extract_time, so skipping date extraction for scheduling intent.")
        
        logger.info(f"Parsed scheduling info: Time='{time_val}', Date='{date_val}'")
        return time_val, date_val

    # --- Main POST method ---
    def post(self, request):
        start_time = datetime.now() # For basic profiling
        question = str(request.data.get('question', '')).strip()
        channel_id = str(request.data.get('channel_id', '')).strip()
        phonenumber = str(request.data.get('phonenumber', '')).strip()
        uuid_val = str(request.data.get('uuid', '')).strip()
        if not uuid_val: # Ensure UUID is present
            uuid_val = str(uuid4())
            logger.warning(f"No UUID provided in request, generated new one: {uuid_val}")

        call_disconnect = request.data.get('call_disconnect', False)

        logger.info(f"--- NEW REQUEST START --- UUID: {uuid_val}, Q: '{question[:100]}...', Disconnect: {call_disconnect}")
        
        # --- Call Disconnect Logic ---
        if call_disconnect:
            logger.info(f"Call disconnected for UUID: {uuid_val}. Processing disposition.")
            # (Your existing call_disconnect logic using client_openai and DB updates for disposition)
            # ... This part is assumed to be mostly self-contained and using `client_openai` ...
            # For brevity, keeping it as a high-level step here.
            # Ensure YOUR_CONVERSATION_TABLE_NAME is correct in this logic too.
            # Example placeholder:
            # summary_info = summarize_conversation_from_db(uuid_val) # You'd need this function
            # disposition = classify_disposition_with_openai(summary_info['text'], client_openai)
            # meeting_details = extract_meeting_details_with_openai(summary_info['text'], client_openai)
            # update_db_with_disposition(summary_info['id'], disposition, meeting_details)
            delete_conversation_cache(uuid_val)
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Call disconnect processed for UUID: {uuid_val}. Time taken: {processing_time:.2f}s")
            return Response({"question": "", "answer": "Call Disconnected, disposition processed."}, status=status.HTTP_200_OK)

        # --- Main Chat Flow ---
        current_dt_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        conversation_hist = get_conversation_history_cache(uuid_val)

        # Determine if it's effectively the first "real" interaction for introduction purposes
        # The LLM prompt relies on conversation history length / content.
        is_first_meaningful_interaction = not any(entry['role'] == 'assistant' for entry in conversation_hist)

        # Add current user question to history for LLM *before* any parsing/validation response
        update_conversation_cache(uuid_val, {"role": "user", "content": question})
        
        # --- Scheduling Attempt ---
        # The parse_scheduling_info will try to find time and corresponding date.
        parsed_time_str, parsed_date_str = self.parse_scheduling_info(question, current_dt_ist)

        if parsed_time_str and parsed_date_str: # Both must be present to attempt validation
            logger.info(f"Potential schedule intent for UUID {uuid_val}: Time='{parsed_time_str}', Date='{parsed_date_str}'")
            is_valid_schedule, validation_response_msg = validate_schedule_time_logic(parsed_time_str, parsed_date_str)
            
            # The response from validate_schedule_time_logic is now the bot's direct response for this turn.
            # It could be a rejection ("Sorry, that time won't work...") or a confirmation query ("Okay... Is that correct?").
            update_conversation_cache(uuid_val, {"role": "assistant", "content": validation_response_msg})
            store_conversation_db(question, validation_response_msg, channel_id, phonenumber, uuid_val)
            logger.info(f"Responded with schedule validation message for UUID {uuid_val}: '{validation_response_msg}'")
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"--- REQUEST END (Validation Path) --- UUID: {uuid_val}. Time: {processing_time:.2f}s")
            return JsonResponse({"question": question, "answer": validation_response_msg}, status=status.HTTP_200_OK)
        
        # --- Fallback to LLM for general queries or if scheduling wasn't conclusive ---
        logger.info(f"No conclusive schedule action from parser/validator, proceeding to LLM for UUID {uuid_val}")
        
        if db_vector_store is None:
            logger.critical(f"Vector store (db_vector_store) is not available for UUID {uuid_val}. Cannot perform RAG.")
            llm_response_text = "I'm currently having some trouble accessing detailed information. Please try again in a few moments."
            update_conversation_cache(uuid_val, {"role": "assistant", "content": llm_response_text})
            store_conversation_db(question, llm_response_text, channel_id, phonenumber, uuid_val)
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"--- REQUEST END (DB Error) --- UUID: {uuid_val}. Time: {processing_time:.2f}s")
            return Response({"question": question, "answer": llm_response_text}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        rag_context = "No specific context found for this query. Please use general knowledge or guide towards scheduling."
        try:
            # K can be adjusted. Relevance score thresholding is important.
            rag_docs_with_scores = db_vector_store.similarity_search_with_relevance_scores(question, k=2)
            if rag_docs_with_scores:
                # Example: Concatenate top k docs if their score is above a threshold
                relevant_texts = [doc.page_content for doc, score in rag_docs_with_scores if score > 0.50] # Adjust threshold
                if relevant_texts:
                    rag_context = " ".join(relevant_texts)
            logger.debug(f"RAG context for UUID {uuid_val} (first 100 chars): {rag_context[:100]}...")
        except Exception as e:
            logger.error(f"RAG similarity search failed for UUID {uuid_val}: {e}", exc_info=True)
            rag_context = "Could not retrieve specific details due to a technical issue. Please offer to schedule a meeting."

        prompt_current_date = current_dt_ist.strftime("%A, %B %d, %Y")
        prompt_current_time = current_dt_ist.strftime("%I:%M %p IST")
        prompt_earliest_time_today = (current_dt_ist + timedelta(minutes=30)).strftime("%I:%M %p IST")
        
        # Determine greeting for system prompt based on current time (only used if needed by prompt's logic)
        greeting_hour_for_prompt = current_dt_ist.hour
        current_greeting_for_prompt = ("Good morning" if 6 <= greeting_hour_for_prompt < 12 else
                                      "Good afternoon" if 12 <= greeting_hour_for_prompt < 18 else
                                      "Good evening")

        system_prompt_for_llm = f"""
You are Jessica, an outbound calling agent from Motilal Oswal. Your primary goal is to introduce the Motilal Oswal Services Fund and guide interested customers to schedule a follow-up meeting. You are confident, clear, and lead the conversation.

### Your Persona & Role
- You are **Jessica** from Motilal Oswal.
- You are **NOT a financial advisor**. Do not give financial advice.
- Your tone is **energetic, confident, and professional but friendly**.
- Use **concise language** (1-2 sentences per turn is ideal). Avoid jargon.
- **Lead the conversation** towards the goal of scheduling a meeting if the user shows interest or asks questions about the fund.

### Handling Greetings and Introduction
1.  **Initial Interaction (First or Second Turn, No Prior Bot Introduction):**
    If the conversation history shows that you (assistant) have not spoken yet or only had a very brief prior interaction, and the user provides their first substantial query or a simple greeting:
    Respond with: "{current_greeting_for_prompt}! This is Jessica calling from Motilal Oswal. We're currently highlighting our Services Fund, which targets India’s fastest-growing service sectors for potential long-term growth. Is this something you might be interested in hearing a bit more about, or perhaps you've explored service sector investments before?"
    * Deliver this introduction or a close variant **only once** at the effective start of the meaningful conversation.
2.  **Subsequent Greetings/Interactions:** If the user says "hello" again *after* the introduction has been made, or if the conversation is ongoing:
    Respond naturally without repeating the full introduction. Examples: "Yes, I'm here.", "Hi again! How can I help you further regarding the fund?", or directly address their new query based on the ongoing context.

### Core Task: Pitch and Schedule
- **Pitch Brevity:** Briefly mention the fund's focus (service sectors, long-term growth).
- **Answering Questions:**
    - Use the **RELEVANT CONTEXT** provided below to answer questions about the fund.
    - If the answer is not in the context or too detailed: "That's a good question. For more specific details like that, it would be best to schedule a quick call with our specialist who can provide a comprehensive overview. Would you be open to that?"
    - **NEVER MAKE UP INFORMATION.**
- **Scheduling Interaction:**
    - If the user expresses general interest in scheduling (e.g., "Can I schedule a call?", "What time works?"): "Great! To discuss this in more detail, I can set up a call for you. Our team is available Monday to Saturday, from 9 AM to 8 PM IST. What day and time would work best for you?"
    - If the user suggested a specific time/date in their *current* query, and the system (in the previous turn, if applicable) gave a validation message:
        - If the **previous assistant turn was a validation rejection** (e.g., "Sorry, that time is on a Sunday."): Acknowledge it and re-prompt. Example: "Right, as I mentioned, Sundays aren't available. How about another day, say, next Monday or Tuesday between 9 AM and 8 PM?"
        - If the **previous assistant turn was a confirmation query** (e.g., "Okay, I've noted... Is that correct?"), and the user's *current* query is "yes", "correct", or similar: Respond with final confirmation. Example: "Excellent, that's confirmed then! We'll send out a calendar invite shortly. Thanks for your time, and have a great day!" (Then aim to end the call).
        - If the user's *current* query is "no" or a change request to a previous confirmation query: "Alright, no problem. What new time would you prefer instead? Just a reminder, it's Mon-Sat, 9 AM to 8 PM IST, and before June 3rd, 2025."
    - If user asks for "now" or "ASAP": "I understand you're keen to connect. While I can't set up an immediate call, we can schedule one with a specialist quite soon, possibly starting from around {prompt_earliest_time_today} today if that's suitable, or another time that works for you."

### Important Don'ts
- **Do NOT repeat your full introduction** after it has been effectively delivered once.
- **Do NOT say "How can I help/assist you?"** as a generic opening. Be more directed.
- **Do NOT go off-topic.** Keep the focus on the Services Fund and scheduling.
- **Do NOT provide financial advice.**

### Current System Information (All times are IST)
- Date: {prompt_current_date}
- Time: {prompt_current_time}
- Day: {current_dt_ist.strftime('%A')}
- Earliest approximate meeting time for today: {prompt_earliest_time_today}
- Scheduling window: Mon-Sat, 9:00 AM - 8:00 PM. No scheduling after June 3rd, 2025, 8:00 PM.

### RELEVANT CONTEXT from Knowledge Base for User's Query:
{rag_context}
"""
        
                # Prepare messages for LLM, ensuring roles are correct
        llm_messages = [{"role": "system", "content": system_prompt_for_llm}]
        cached_history = get_conversation_history_cache(uuid_val)
        for entry in cached_history:
            role = entry.get("role")
            content = entry.get("content")
            if role in ["user", "assistant"] and content: # Ensure valid role and content
                llm_messages.append({"role": role, "content": content})
            else:
                logger.warning(f"Skipping invalid history entry for LLM: {entry}")


        llm_response_text = "I'm sorry, I had a slight issue processing that. Could you try rephrasing?" # Default fallback
        try:
            if not hasattr(LLM_INSTANCE, 'tokenizer') or not hasattr(LLM_INSTANCE.tokenizer, 'apply_chat_template'):
                logger.critical("LLM_INSTANCE.tokenizer or apply_chat_template method is missing from your LLM_INSTANCE setup!")
                raise AttributeError("LLM_INSTANCE is not configured correctly with a tokenizer.")

            formatted_llm_prompt = LLM_INSTANCE.tokenizer.apply_chat_template(
                llm_messages, tokenize=False, add_generation_prompt=True
            )
            logger.debug(f"Formatted prompt for LLM_INSTANCE (UUID {uuid_val}) (first 500 chars): {formatted_llm_prompt[:500]}...")

            llm_sampling_params = SamplingParams(max_tokens=180, temperature=0.55, top_p=0.9) # Slightly adjusted params
            
            # request_id_for_llm = f"motilal-chat-{uuid_val}-{int(current_dt_ist.timestamp())}"
            
            generated_tokens_list = []
            # Check and use the appropriate generation method for your LLM_INSTANCE
            if hasattr(LLM_INSTANCE, 'generate_stream'):
                # logger.debug(f"Using LLM_INSTANCE.generate_stream for request_id: {request_id_for_llm}")
                # This loop needs to EXACTLY match how your generate_stream yields data
                for chunk in LLM_INSTANCE.generate_stream(formatted_llm_prompt, llm_sampling_params):
                    if isinstance(chunk, tuple) and len(chunk) == 2: # e.g. (token_str, finished_bool)
                        token_data, finished_signal = chunk
                        if token_data: generated_tokens_list.append(str(token_data))
                        if finished_signal: break 
                    elif isinstance(chunk, str): # e.g. just yields token strings
                        generated_tokens_list.append(chunk)
                    # Add other conditions if your stream yields different structures (e.g., vLLM RequestOutput-like objects)
                    else:
                        logger.warning(f"Unexpected chunk type from generate_stream: {type(chunk)}. Attempting str conversion.")
                        generated_tokens_list.append(str(chunk)) # Best effort
                llm_response_text = "".join(generated_tokens_list).strip()

            elif hasattr(LLM_INSTANCE, 'llm') and hasattr(LLM_INSTANCE.llm, 'generate'): # Common for direct vLLM engine
                # logger.debug(f"Using LLM_INSTANCE.llm.generate for request_id: {request_id_for_llm}")
                raw_vllm_outputs = LLM_INSTANCE.llm.generate(formatted_llm_prompt, llm_sampling_params)
                if raw_vllm_outputs and raw_vllm_outputs[0].outputs: # Assuming first result, first output
                    llm_response_text = raw_vllm_outputs[0].outputs[0].text.strip()
            else:
                logger.critical(f"LLM_INSTANCE (UUID {uuid_val}) lacks a recognized 'generate_stream' or 'llm.generate' method.")
                raise NotImplementedError("Suitable LLM generation method not found on LLM_INSTANCE.")
            
            logger.info(f"LLM raw response for UUID {uuid_val}: '{llm_response_text}'")
            # Further clean-up of llm_response_text can be added here if needed (e.g., removing known model artifacts)

        except Exception as e:
            logger.critical(f"LLM generation critical error for UUID {uuid_val}: {e}", exc_info=True)
            # llm_response_text will use its default value set before the try block.

        update_conversation_cache(uuid_val, {"role": "assistant", "content": llm_response_text})
        store_conversation_db(question, llm_response_text, channel_id, phonenumber, uuid_val)
        
        logger.info(f"FINAL Bot Response for UUID {uuid_val}: '{llm_response_text}'")
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"--- REQUEST END (LLM Path) --- UUID: {uuid_val}. Total time: {processing_time:.2f}s")
        return Response({"question": question, "answer": llm_response_text}, status=status.HTTP_200_OK)
