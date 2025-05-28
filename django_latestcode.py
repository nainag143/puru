from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from zoneinfo import ZoneInfo  # Replaces pytz
import re
from datetime import datetime, timedelta
import pytz

from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2Tokenizer
from django.core.cache import cache
import os
import pytz
from openai import OpenAI
import time
import torch
import datetime
import logging
from django.utils.decorators import method_decorator
# from django.views.decorators.cache import never_cache
from django.core.cache import cache
import hashlib
import pymysql
import json
import pandas as pd 
from datetime import datetime
import sqlite3
import argparse
import os
import sys
import time
# from db  import perform_search
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from vllm import SamplingParams
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Import the LLM class with streaming supportl
from vllm.utils import Counter
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from llm_agent_project.llm_instance import LLM_INSTANCE, TOKENIZER

os.makedirs("/root/BotHub_llm_model/llm_agent_project/motilal_app/logs", exist_ok=True)
logging.basicConfig(filename="/root/BotHub_llm_model/llm_agent_project/motilal_app/logs/motilal.log", 
                    level=logging.INFO,  # Set the minimum level of messages to log
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Ensure PyTorch is using GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}\n")
 
# File Paths
# text_file = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/output.txt"
# pdf_file = "/var/www/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal RD inputs 03 march 2025 v14.pdf"
# pdf_file = "/var/www/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund - Details_14_May_2025_V1.pdf"
# pdf_file = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund_and_Services_FAQ_2_doc_merged_2025_05_16_V2.pdf"
pdf_file = "/root/BotHub_llm_model/llm_agent_project/media/motilal_app/Motilal Oswal Services Fund_and_Services_FAQ_2_doc_merged_2025_05_16_V3.pdf"
 
# Set OpenAI API key
api_key = "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"
client=OpenAI(api_key=api_key)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
# Load Text Document
# with open(text_file, 'rb') as f:
#     content = f.read()
#     state_of_the_union = content.decode('utf-8', errors='ignore')
 
# Text Splitting
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# texts = text_splitter.create_documents([state_of_the_union])
 
# HuggingFace Embeddings
hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

 
# Load the existing collection
db = Chroma(collection_name="test-1", persist_directory="/root/BotHub_llm_model/llm_agent_project/motilal_app/MO_latest", embedding_function=hf)
        
# Load PDF and Create Embeddings
# loader = PyPDFLoader(pdf_file)
# docs = loader.load()
# embeddings = hf.embed_documents([doc.page_content for doc in docs])
 
# ChromaDB Setup
# client = chromadb.PersistentClient('vdb')
# collection = client.get_or_create_collection('test-2')
# if not collection.get()['ids']:
#     collection.add(
#         ids=[str(uuid4()) for _ in docs],
#         documents=[doc.page_content for doc in docs],
#         embeddings=embeddings
#     )
# db = Chroma(collection_name='test-2', persist_directory='vdb', embedding_function=hf)
 
# Tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
 
# def truncate_context(context, max_tokens=800):
#     tokens = tokenizer.encode(context)
#     return tokenizer.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else context
 
# Conversation History
conversation_history = {}
 
def update_conversation(uuid, history):
    key = str(uuid)
    conversation = cache.get(key, [])
    conversation.append(history)
    cache.set(key, conversation, timeout=None)

    # print(f"cache.get(str(uuid), []) : {cache.get(str(uuid), [])}")
 
def get_conversation_history(uuid):
    return cache.get(str(uuid), [])

def delete_conversation(uuid):
    key = str(uuid)
    logging.info(f"Deleted Conversation : {uuid}")
    cache.delete(key)
 
def remove_conversation_history(uuid):
    key = str(uuid)
    cache.delete(key)
 
 
def connection():
    conn = pymysql.connect(
        host = "127.0.0.1",
        user = "root",
        passwd = "passw0rd",
        database = "voice_bot"
    )    
    
    return conn 
 
def store_conversation(question, answer, channelid, phonenumber, uuid):
    print(f"Storing Conversation for {uuid}")
    try:
        conn = connection()
        
        cur = conn.cursor()
        question = question.replace("'" , " ")
        answer = answer.replace("'" , " ")
        query = f"""INSERT INTO tbl_botai (question, answer, channelid, phonenumber, uuid) VALUES('{question}' ,'{answer}' ,'{channelid}' ,'{phonenumber}' ,'{uuid}');"""
        
        cur.execute(query)
        conn.commit()
        print(f"Conversation Stored for {uuid}")
                
    except Exception as e:
        print(f"Error Database Connection Error for {uuid} : {e} ")
    
    finally:
        cur.close()
        conn.close()

def get_time_of_day():
    """Returns the appropriate greeting based on the current time"""
    current_hour = int(time.strftime("%H"))
    
    if 6 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"



# logging = logging.getlogging(__name__)

def validate_schedule_time(selected_time_str, selected_date_str=None):
    """
    Validates if the selected time is valid for scheduling a meeting
    
    Args:
        selected_time_str (str): Time string in format "HH:MM AM/PM"
        selected_date_str (str, optional): Date string in format "DD-MM-YYYY". Defaults to None (today).
    
    Returns:
        tuple: (is_valid, message) - Boolean indicating if time is valid and a message
    """
    print(f"\n===== VALIDATING SCHEDULE =====")
    print(f"VALIDATING: Time='{selected_time_str}', Date='{selected_date_str}'")
    
    try:
        # Parse user input like "12:10 PM"
        selected_time = datetime.strptime(selected_time_str, "%I:%M %p").time()
        print(f"PARSED TIME: {selected_time.strftime('%I:%M %p')}")
        
        # Current IST time
        current_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        print(f"CURRENT IST TIME: {current_time.strftime('%d-%m-%Y %I:%M %p')}")
        logging.info(f"Current IST time: {current_time.strftime('%d-%m-%Y %I:%M %p')}")
        
        # Use provided date if any, otherwise use current date
        if selected_date_str:
            try:
                # Try with format DD-MM-YYYY
                selected_date = datetime.strptime(selected_date_str, "%d-%m-%Y").date()
                print(f"PARSED DATE (DD-MM-YYYY): {selected_date.strftime('%d-%m-%Y')}")
            except ValueError:
                try:
                    # Try with format DD/MM/YYYY
                    selected_date = datetime.strptime(selected_date_str.replace('/', '-'), "%d-%m-%Y").date()
                    print(f"PARSED DATE (DD/MM/YYYY): {selected_date.strftime('%d-%m-%Y')}")
                except ValueError:
                    # Default to today if date parsing fails
                    logging.error(f"Could not parse date: {selected_date_str}, using today's date")
                    print(f"ERROR PARSING DATE '{selected_date_str}', DEFAULTING TO TODAY")
                    selected_date = current_time.date()
        else:
            # If no date specified, use today
            selected_date = current_time.date()
            print(f"NO DATE SPECIFIED, USING TODAY: {selected_date.strftime('%d-%m-%Y')}")
        
        # Combine date and time with timezone
        selected_datetime = datetime.combine(selected_date, selected_time).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        print(f"COMBINED DATE & TIME: {selected_datetime.strftime('%d-%m-%Y %I:%M %p')}")
        logging.info(f"Selected time: {selected_datetime.strftime('%d-%m-%Y %I:%M %p')}")
        
        # Check if the date is today and if the time has already passed
        if selected_date == current_time.date() and selected_datetime < current_time:
            print(f"REJECTION: Time has already passed today")
            logging.info("Meeting rejected: Time has already passed today")
            return False, f"That time has already passed today. Please choose a future time."

        # RULE 1: Check if the selected date is a Sunday
        day_of_week = selected_datetime.weekday()
        print(f"DAY OF WEEK: {day_of_week} (0=Monday, 6=Sunday)")
        
        if day_of_week == 6:  # 6 = Sunday
            print(f"REJECTION: Date falls on Sunday")
            logging.info("Meeting rejected: Date falls on Sunday")
            return False, "I can't schedule meetings on Sundays. Would Monday work for you instead?"

        # RULE 2: Check for 30-minute buffer (only if same day)
        if selected_date == current_time.date():
            min_allowed_time = current_time + timedelta(minutes=30)
            print(f"MINIMUM ALLOWED TIME (current + 30min): {min_allowed_time.strftime('%d-%m-%Y %I:%M %p')}")
            
            if selected_datetime < min_allowed_time:
                time_diff = (min_allowed_time - current_time).total_seconds() / 60
                print(f"REJECTION: Too soon. Only {time_diff:.0f} minutes from current time")
                logging.info(f"Meeting rejected: Too soon. Only {time_diff:.0f} minutes from current time")
                earliest_time = min_allowed_time.strftime("%I:%M %p")
                return False, f"Sorry, I need at least 30 minutes to prepare. The earliest I can schedule is {earliest_time} today. Would that work for you?"

        # RULE 3: Check business hours (9 AM to 8 PM)
        business_start = datetime.combine(selected_date, datetime.strptime("09:00 AM", "%I:%M %p").time())
        business_start = business_start.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        
        business_end = datetime.combine(selected_date, datetime.strptime("08:00 PM", "%I:%M %p").time())
        business_end = business_end.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
        
        print(f"BUSINESS HOURS: {business_start.strftime('%I:%M %p')} to {business_end.strftime('%I:%M %p')}")

        if selected_datetime < business_start or selected_datetime > business_end:
            print(f"REJECTION: Outside business hours")
            logging.info("Meeting rejected: Outside business hours")
            return False, "Our meeting hours are between 9:00 AM and 8:00 PM IST, Monday to Saturday. Please choose another time."

        # All checks passed - time is valid
        formatted_time = selected_datetime.strftime("%d-%m-%Y %I:%M %p")
        print(f"VALIDATION SUCCESS: Meeting scheduled for {formatted_time}")
        print("===== VALIDATION COMPLETE =====\n")
        logging.info(f"Meeting scheduled successfully for {formatted_time}")
        return True, f"Meeting confirmed for {formatted_time} IST."

    except Exception as e:
        print(f"VALIDATION ERROR: {str(e)}")
        logging.error(f"Time validation error: {str(e)}")
        return False, "I couldn't understand that time format. Please specify a time like '10:30 AM.'"


# # This API is used for testing 
# @csrf_exempt
# def motilal_bot(request):
#     if request.method == "GET":
            
#             # # DB Connection 
#             # conn = connection()
#             # cur = conn.cursor()

#             # cur.close()
#             # conn.close()

#             # Logic Goes Here 
#             text=request.data.get("text")
            
#         return JsonResponse({'message': 'Collection Bot API Working perfectly .'}, status=200)
    
    
#     return JsonResponse({'status': 'error', 'message': 'method not allowed'}, status=405)



# @method_decorator(never_cache, name='dispatch')



# India timezone
IST = pytz.timezone('Asia/Kolkata')

# Constants
MAX_MEETING_DATE_STR = "2025-06-03"
MAX_MEETING_DATE = datetime.strptime(MAX_MEETING_DATE_STR, "%Y-%m-%d").date()

def validate_and_format_meeting(user_date_str, user_time_str):
    """
    user_date_str: "YYYY-MM-DD"
    user_time_str: "HH:MM" or "HH:MM AM/PM"
    Returns (confirmed_date, confirmed_time_24h) or (None, None) if invalid
    """

    try:
        # Parse date
        meeting_date = datetime.strptime(user_date_str, "%Y-%m-%d").date()
        if meeting_date > MAX_MEETING_DATE:
            return None, None  # date beyond allowed max

        # Parse time (handle AM/PM)
        try:
            meeting_time = datetime.strptime(user_time_str, "%I:%M %p").time()
        except ValueError:
            meeting_time = datetime.strptime(user_time_str, "%H:%M").time()

        # Check day of week (Mon=0 ... Sun=6)
        if meeting_date.weekday() == 6:  # Sunday
            return None, None

        # Check time range (09:00 to 20:00 IST)
        if not (9 <= meeting_time.hour < 20 or (meeting_time.hour == 20 and meeting_time.minute == 0)):
            return None, None

        # Format confirmed_date and confirmed_time_24h
        confirmed_date = meeting_date.strftime("%Y-%m-%d")
        confirmed_time_24h = meeting_time.strftime("%H:%M")

        return confirmed_date, confirmed_time_24h

    except Exception:
        return None, None

# Example usage:
date_input = "2025-06-02"
time_input = "3:30 PM"

confirmed_date, confirmed_time_24h = validate_and_format_meeting(date_input, time_input)
if confirmed_date:
    print(f"Confirmed date: {confirmed_date}, Confirmed time (24h): {confirmed_time_24h}")
else:
    print("Invalid meeting time/date")



class Local_Motilal_View(APIView):

    def validate_and_format_meeting(self, user_date_str, user_time_str):
        """
        user_date_str: "YYYY-MM-DD"
        user_time_str: "HH:MM" or "HH:MM AM/PM"
        Returns (confirmed_date, confirmed_time_24h) or (None, None) if invalid
        """

        try:
            # Parse date
            meeting_date = datetime.strptime(user_date_str, "%Y-%m-%d").date()
            if meeting_date > MAX_MEETING_DATE:
                return None, None  # date beyond allowed max

            # Parse time (handle AM/PM)
            try:
                meeting_time = datetime.strptime(user_time_str, "%I:%M %p").time()
            except ValueError:
                meeting_time = datetime.strptime(user_time_str, "%H:%M").time()

            # Check day of week (Mon=0 ... Sun=6)
            if meeting_date.weekday() == 6:  # Sunday
                return None, None

            # Check time range (09:00 to 20:00 IST)
            if not (9 <= meeting_time.hour < 20 or (meeting_time.hour == 20 and meeting_time.minute == 0)):
                return None, None

            # Format confirmed_date and confirmed_time_24h
            confirmed_date = meeting_date.strftime("%Y-%m-%d")
            confirmed_time_24h = meeting_time.strftime("%H:%M")

            return confirmed_date, confirmed_time_24h

        except Exception:
            return None, None

    def post(self, request):
        question = str(request.data.get('question', '')).strip()
        channel_id = str(request.data.get('channel_id', '')).strip()
        phonenumber = str(request.data.get('phonenumber', '')).strip()
        uuid = str(request.data.get('uuid', '')).strip()
        call_disconnect = request.data.get('call_disconnect')

        print(f"\n\n===== NEW REQUEST =====")
        print(f"QUESTION: {question}")
        logging.info(f"Received question: {question}")

        if call_disconnect is True:
            return JsonResponse({"message": "Call disconnected."}, status=200)

        cache_key = f"chatbot_response:{hashlib.md5(question.encode()).hexdigest()}"
        
        # Get proper greeting based on current IST time
        current_time_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        greeting_hour = current_time_ist.hour
        
        print(f"CURRENT IST TIME: {current_time_ist.strftime('%d-%m-%Y %I:%M %p')}")
        logging.info(f"Current IST time: {current_time_ist.strftime('%d-%m-%Y %I:%M %p')}")
        
        greeting = (
            "Good morning" if 6 <= greeting_hour < 12 else
            "Good afternoon" if 12 <= greeting_hour < 18 else
            "Good evening"
        )

        # Get conversation history
        conversation_history = get_conversation_history(uuid)
        
        # Exit command handling
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            farewell = "Thank you for your time today! Feel free to reach out whenever you need financial advice. Have a great day!"
            update_conversation(uuid, {"role": "assistant", "content": farewell})
            return JsonResponse({"question": question, "answer": farewell}, status=200)

        # Get context for the question
        result = db.similarity_search_with_relevance_scores(question, k=1)
        context = "".join(doc.page_content for doc, _ in result)

        # First time greeting - ONLY for new conversations with no history
        if not conversation_history:
            first_message = f"{greeting}! I'm Jessica from Motilal Oswal. Would you like to know about our new investment fund?"
            update_conversation(uuid, {"role": "user", "content": question})
            update_conversation(uuid, {"role": "assistant", "content": first_message})
            return JsonResponse({"question": question, "answer": first_message}, status=200)

        update_conversation(uuid, {"role": "user", "content": question})

        # Get current date, time and day in IST
        current_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        current_date = current_ist.strftime("%d-%m-%Y")
        time_current = current_ist.strftime("%I:%M %p")
        current_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_ist.weekday()]
        
        # Calculate time 30 minutes from now for immediate meeting requests
        time_plus_30_ist = current_ist + timedelta(minutes=30)
        time_plus_30 = time_plus_30_ist.strftime("%I:%M %p")
        
        print(f"CURRENT DAY: {current_day}, DATE: {current_date}, TIME: {time_current}")
        print(f"TIME PLUS 30 MINS: {time_plus_30}")
        logging.info(f"Current day: {current_day}, Date: {current_date}, Time: {time_current}")
        logging.info(f"Time plus 30 mins: {time_plus_30}")

        # Parse scheduling information from the question
        time_str, date_str = self.parse_scheduling_info(question, current_ist)
        
        # Log extracted scheduling information
        print(f"EXTRACTED: Time={time_str}, Date={date_str}")
        logging.info(f"Extracted scheduling info - Time: {time_str}, Date: {date_str}")

        # Handle meeting scheduling validation if time pattern detected
        if time_str:
            try:
                print(f"VALIDATING: Time={time_str}, Date={date_str}")
                # Validate the schedule
                is_valid, validation_msg = validate_schedule_time(time_str, date_str)
                print(f"VALIDATION RESULT: Valid={is_valid}, Message={validation_msg}")
                logging.info(f"Schedule validation result: {is_valid}, message: {validation_msg}")
                
                if not is_valid:
                    update_conversation(uuid, {"role": "assistant", "content": validation_msg})
                    print(f"FINAL RESPONSE (INVALID TIME): {validation_msg}")
                    return JsonResponse({"question": question, "answer": validation_msg}, status=200)
                else:
                    # If valid, return the confirmation directly
                    update_conversation(uuid, {"role": "assistant", "content": validation_msg})
                    print(f"FINAL RESPONSE (VALID TIME): {validation_msg}")
                    return JsonResponse({"question": question, "answer": validation_msg}, status=200)
            except Exception as e:
                print(f"ERROR VALIDATING TIME: {str(e)}")
                logging.error(f"Error validating time: {str(e)}")
                # Continue with LLM response if time validation fails

        # Simplified system prompt
        # system_prompt = f"""
        #     You are Jessica from Motilal Oswal. You're calling to introduce the Motilal Oswal Services Fund and schedule meetings. Keep your responses short.

        #     CORE RULES:
        #     - Answer questions ONLY using information in CONTEXT below.
        #     - Don't repeat yourself.
        #     - Don't say "How can I assist you?" - you are calling them.
        #     - Use 1-2 sentences per response.
        #     - Never claim to be a financial advisor.

        #     FIRST INTERACTION:
        #     - Introduce yourself and briefly mention the fund.
        #     - Don't repeat your introduction again.

        #     SCHEDULING:

        #     - Ask for customer's preferred time (Between Monday to Saturday, 9AM-8PM IST).
        #     - NEVER schedule meetings yourself.
        #     - No meetings on Sundays.
        #     - No meetings before {time_plus_30} today (30-min buffer).
        #     - No meetings after June 3, 2025.
        #     - If customer requests "now" or "immediately": Say "We need some time to prepare. How about {time_plus_30} today?"
        #     - Our system will validate and confirm all meeting times automatically.

        #     RESPONDING:
        #     - If customer asks about the fund: Answer briefly using CONTEXT.
        #     - If they want more details: Suggest scheduling a meeting.
        #     - After confirming a meeting: Stop asking for more information.

        #     CURRENT INFO:
        #     Date: {current_date}
        #     Time: {time_current}
        #     Day: {current_day}
        #     Earliest time: {time_plus_30}

        #     CONTEXT:
        #     {context}
        # """

#         system_prompt = f"""
#         Jessica: Outbound Sales Agent (Motilal Oswal)
# Goal: Spark interest, explain benefits, and schedule a meeting for Motilal Oswal's new investment opportunity. Be confident, proactive, and persuasive. Avoid sounding like a customer service agent.
# Language: Hinglish (with a feminine touch).
# Context: {context} (This should ideally contain brief, key selling points of the NFO, including the minimum investment amount of Rs. 500.)
# Current Info:
# Date: 28-05-2025
# Time: 04:14:59 PM IST
# Day: Wednesday
# Structured Steps:
# No Self-Introduction (Done): Assume intro already happened.
# Engagement & Value Proposition: Directly highlight key NFO benefits (concise, persuasive).
# Example: "Hi! Main Jessica, Motilal Oswal se. Humara ek naya investment opportunity aaya hai, jo market momentum leverage karke aapki growth maximize karne ke liye design kiya gaya hai. Isme koi lock-in period nahi hai, jo I think aapko pasand aayega. Aur minimum investment sirf Rupees 500 se shuru ho jaati hai!"
# Objection Handling: Address concerns naturally. If firm decline (twice), close the call.
# Fund Performance: "Dekhiye, past performance future results ki guarantee nahi hoti, par yeh fund market momentum capitalize karne ke liye banaaya gaya hai. Specific projections ke liye, main suggest karungi ki humari financial experts se meeting mein baat karein."
# Minimum Investment Query: "Is fund mein minimum investment amount sirf Rupees 500 hai, jaisa ki hamare documents mein bhi mention hai."
# Single-Word Response: If "Okay" or "Hmm," prompt for next step, don't loop.
# Example: "Toh, kya aap iske baare mein aur jaanna chahengi, ya hum ek short meeting set kar sakte hain?"
# Meeting Scheduling:
# Time Zone: Indian time zone (IST).
# NFO End Date: Meetings only before June 3, 2025, 8 PM IST.
# If after: "NFO 3rd June 2025 ko end ho raha hai, toh unfortunately hum uske baad meetings schedule nahi kar payenge."
# Working Hours: Monday to Saturday, 9:00 AM to 8:00 PM IST.
# Sunday Restriction (Explicit): "Hum Sundays ko meetings schedule nahi karte, I'm sorry. Please Monday se Saturday, 9:00 AM se 8:00 PM IST ke beech ka koi din choose kijiye."
# Sunday Restriction (Implicit): (If current day is Friday, and user says "day after tomorrow" which is Sunday) "Sorry, hum Sundays ko meetings schedule nahi karte. Please koi aur din pick kijiye, Monday se Saturday."
# Past Time/Date: "Hum past mein meetings schedule nahi kar sakte. Aaj ke liye earliest available time abhi 04:44:59 PM IST hai." (Adjust for next day if current time is near 8 PM).
# Outside Working Hours: "Please 9:00 AM se 8:00 PM IST ke beech ka time choose kijiye."
# Negative Language: NEVER use "can't," "unable," "fully booked." Always offer alternatives.
# Immediate Call:
# If within hours: "Bilkul! Main abhi aapke liye call arrange karti hoon."
# If outside hours: "I'm sorry, calls 9:00 AM se 8:00 PM IST ke beech hi schedule ho sakte hain. Kya aap in hours ke andar koi time choose kar sakti hain?"
# Scheduling Prompt: "Toh meeting schedule karne ke liye, please ek date aur time bataiye jo aapke liye best ho. Kaunsa date aur time aap prefer karengi?"
# Date/Time Format: Confirm in DD-MM-YYYY and HH:MM AM/PM IST.
# Example: "Next Monday, 02-06-2025 ko, 03:00 PM IST par meeting set kar sakte hain. Kya yeh aapke liye theek rahega?"
# After Confirmation: "Bahut accha! Main aapki meeting confirm karti hoon." (No mention of calendar invite)
# Call Closure: If customer firmly declines twice:
# "Main samajhti hoon! Aapke time ke liye bohot dhanyavaad. Have a great day!"
# Remain silent until disconnect. Do not re-engage.
# Conversation Guidelines:
# Retain full chat history, build logically.
# No reintroductions after conversation starts.
# Keep benefits short, persuasive.
# Do not repeat objection responses.
# After "Yes," assume confirmation and move forward.
# Avoid loops on "Hmm," "Okay" – prompt for decisions.
# Once confirmed, thank and give next steps. Do not mention calendar invite.
# If user says "thank you," "I'm done," or ends, wait silently.
# Respond briefly (2-3 lines max) unless explanation needed.
# If answer not in context, use external knowledge.
# Detect Hindi and switch to Hinglish.
# Replace "IT" with "I.T" or "I T".
# Replace "ITES" with "I.T.E.S" or "I T E S".
# """


#         system_prompt = f"""
#         Jessica: Outbound Sales Agent (Motilal Oswal)
# Goal: Spark interest, explain benefits, and schedule a meeting for Motilal Oswal's new investment opportunity. Be confident, proactive, and persuasive. Avoid sounding like a customer service agent.
# Language: Hinglish (with a feminine touch).
# Context: {context} (This should ideally contain brief, key selling points of the NFO, including the minimum investment amount of Rs. 500.)
# Current Info (Crucial for calculations):
# Current Date: 28-05-2025
# Current Time: 05:08:50 PM IST
# Current Day: Wednesday
# Structured Steps:
# ... (Previous steps remain the same) ...
# Meeting Scheduling:
# Internal Date/Time Calculation: When a user provides a day (e.g., "tomorrow," "day after," "next Monday," "Friday"), first calculate the exact corresponding calendar date based on Current Date. Then, apply all subsequent rules to this calculated date and the requested time.
# Time Zone: Indian time zone (IST).
# NFO End Date: Meetings only before June 3, 2025, 8 PM IST.
# If the calculated meeting date is on or after June 3, 2025: "NFO 3rd June 2025 ko end ho raha hai, toh unfortunately hum uske baad meetings schedule nahi kar payenge."
# Working Hours: Monday to Saturday, 9:00 AM to 8:00 PM IST.
# Sunday Restriction (Explicit & Implicit):
# If the calculated meeting date is a Sunday: "Hum Sundays ko meetings schedule nahi karte, I'm sorry. Please Monday se Saturday, 9:00 AM se 8:00 PM IST ke beech ka koi din choose kijiye."
# (Example for implicit): If Current Day is Wednesday, and user says "day after tomorrow", the bot must calculate this as Friday. If user says "day after day after tomorrow", it should calculate Saturday. If it's Thursday and user says "day after tomorrow", it calculates Saturday.
# Past Time/Date Validation:
# If the calculated meeting date is before Current Date: "Hum past mein meetings schedule nahi kar sakte."
# If the calculated meeting date is Current Date AND the requested time is before Current Time: "Hum past mein meetings schedule nahi kar sakte. Aaj ke liye earliest available time abhi 05:38:50 PM IST hai." (Always calculate Current Time + 30 minutes for this specific response).
# Outside Working Hours (Time): "Please 9:00 AM se 8:00 PM IST ke beech ka time choose kijiye."
# Negative Language: NEVER use "can't," "unable," "fully booked." Always offer alternatives.
# Immediate Call:
# If within working hours (9 AM - 8 PM IST, Mon-Sat): "Bilkul! Main abhi aapke liye call arrange karti hoon."
# If outside working hours: "I'm sorry, calls 9:00 AM se 8:00 PM IST ke beech hi schedule ho sakte hain. Kya aap in hours ke andar koi time choose kar sakti hain?"
# Scheduling Prompt: "Toh meeting schedule karne ke liye, please ek date aur time bataiye jo aapke liye best ho. Kaunsa date aur time aap prefer karengi?"
# If user gives only day and time (e.g., "Friday 5 PM"): Assume the next upcoming occurrence of that day and time. If this is ambiguous (e.g., "Monday" when Current Date is Tuesday), then clarify: "Aap kaunse Friday ki baat kar rahi hain? Next Friday [calculated date] ko ya koi aur?"
# Date/Time Format: Confirm in DD-MM-YYYY and HH:MM AM/PM IST.
# Example (Bot's confirmation): "Bahut accha! Toh main aapki meeting Friday, 30-05-2025 ko, 05:00 PM IST par confirm karti hoon."
# After Confirmation: "Bahut accha! Main aapki meeting confirm karti hoon." (No mention of calendar invite)
# ... (Remaining steps for Call Closure and Conversation Guidelines are the same) ...
# Revised examples for your specific issue:
# Customer: 'can we do day after 5 Pm'
# Bot's Internal Logic: Current Date is Wednesday, May 28, 2025. "Day after" means May 29, 2025 (Thursday).
# Bot's Response (Corrected): "Thursday, 29-05-2025 ko 05:00 PM IST par theek rahega kya?" (This assumes 5 PM is within working hours and the date is valid. If Thursday May 29, 2025, 5 PM is valid, this is the correct response. If it's a Sunday or after NFO, it will trigger those specific responses).
# Customer: 'friday 5 pm'
# Bot's Internal Logic: Current Date is Wednesday, May 28, 2025. The next upcoming Friday is May 30, 2025.
# Bot's Response (Corrected): "Ji, Friday, 30-05-2025 ko 05:00 PM IST par bilkul set kar sakte hain. Kya yeh aapke liye theek rahega?"
#  """

#         system_prompt = f"""
#         Jessica: Outbound Sales Agent (Motilal Oswal)
# Goal: Spark interest, explain benefits, and schedule a meeting for Motilal Oswal's new investment opportunity. Be confident, proactive, and persuasive. Avoid sounding like a customer service agent.
# Language: Hinglish (with a feminine touch).
# Context: {context} (This should ideally contain brief, key selling points of the NFO, including the minimum investment amount of Rs. 500.)
# Current Info (Crucial for calculations):
# Current Date: 28-05-2025
# Current Time: 05:20:41 PM IST
# Current Day: Wednesday
# Structured Steps:
# No Self-Introduction (Done): Assume intro already happened.
# Engagement & Value Proposition: Directly highlight key NFO benefits (concise, persuasive).
# Example: "Hi! Main Jessica, Motilal Oswal se. Humara ek naya investment opportunity aaya hai, jo market momentum leverage karke aapki growth maximize karne ke liye design kiya gaya hai. Isme koi lock-in period nahi hai, jo I think aapko pasand aayega. Aur minimum investment sirf Rupees 500 se shuru ho jaati hai!"
# Objection Handling: Address concerns naturally. If firm decline (twice), close the call.
# Fund Performance: "Dekhiye, past performance future results ki guarantee nahi hoti, par yeh fund market momentum capitalize karne ke liye banaaya gaya hai. Specific projections ke liye, main suggest karungi ki humari financial experts se meeting mein baat karein."
# Minimum Investment Query: "Is fund mein minimum investment amount sirf Rupees 500 hai, jaisa ki hamare documents mein bhi mention hai."
# Single-Word Response Management: If the user says "Okay" or "Hmm," prompt them toward engagement without looping.
# Example: "Toh, kya aap iske baare mein aur jaanna chahengi, ya hum ek short meeting set kar sakte hain?"
# Meeting Scheduling Guideline:
# Crucial: Internal Date/Time Calculation: When the user provides any relative time reference (e.g., "tomorrow," "day after," "next Monday," "Friday"), first calculate the exact corresponding calendar date (DD-MM-YYYY) and precise time (HH:MM IST) based on Current Date and Current Time. Always assume the next upcoming occurrence for days of the week if a specific date isn't given.
# Validation Order (Strictly Follow This Sequence):
# Sunday Check: If the calculated meeting date is a Sunday: "Hum Sundays ko meetings schedule nahi karte, I'm sorry. Please Monday se Saturday, 9:00 AM se 8:00 PM IST ke beech ka koi din choose kijiye."
# NFO End Date Check: If the calculated meeting date is on or after June 3, 2025: "NFO 3rd June 2025 ko end ho raha hai, toh unfortunately hum uske baad meetings schedule nahi kar payenge."
# Past Date Check: If the calculated meeting date is before Current Date: "Hum past mein meetings schedule nahi kar sakte."
# Working Hours (Time) Check: If the requested time is outside 9:00 AM to 8:00 PM IST: "Please 9:00 AM se 8:00 PM IST ke beech ka time choose kijiye."
# Past Time on Current Day Check: If the calculated meeting date is Current Date AND the requested time is before Current Time: "Hum past mein meetings schedule nahi kar sakte. Aaj ke liye earliest available time abhi 05:50:41 PM IST hai." (Always calculate Current Time + 30 minutes for this specific response).
# All Valid: If all checks pass, proceed to confirm.
# Time Zone: Indian time zone (IST).
# Negative Language: NEVER use negative language like “can’t”, “unable”, “fully booked”. Always offer alternatives.
# Handling Immediate Call Requests:
# If within working hours (9:00 AM - 8:00 PM IST, Mon-Sat): "Bilkul! Main abhi aapke liye call arrange karti hoon."
# If outside working hours: "I'm sorry, calls 9:00 AM se 8:00 PM IST ke beech hi schedule ho sakte hain. Kya aap in hours ke andar koi time choose kar sakti hain?"
# Exact Scheduling Prompt: "Toh meeting schedule karne ke liye, please ek date aur time bataiye jo aapke liye best ho. Kaunsa date aur time aap prefer karengi?"
# If user gives only day and time (e.g., "Friday 5 PM"): Assume the next upcoming occurrence of that day and time. If this is ambiguous (e.g., "Monday" when Current Date is Tuesday), then clarify: "Aap kaunse Friday ki baat kar rahi hain? Next Friday [calculated date] ko ya koi aur?"
# Date and Time Formatting: Confirm in DD-MM-YYYY and HH:MM AM/PM IST.
# Example (Bot's confirmation): "Bahut accha! Toh main aapki meeting Friday, 30-05-2025 ko, 05:00 PM IST par confirm karti hoon."
# After Confirmation: "Bahut accha! Main aapki meeting confirm karti hoon." (Do not mention calendar invite)
# Call Closure:
# If the customer declines twice or firmly refuses, say:
# "Main samajhti hoon! Aapke time ke liye bohot dhanyavaad. Have a great day!"
# Remain silent until they disconnect. Do not re-engage.
# Conversation Guidelines:
# Retain full chat history. Do not reset between messages.
# Build logically on what was previously said.
# Never repeat full intro again after conversation has started.
# Wait for gratitude before fulfilling any follow-up request.
# No reintroductions after “Yes”.
# Keep benefits short, persuasive.
# Do not repeat objection responses.
# After “Yes,” assume confirmation and move forward.
# Avoid loops on “Hmm,” “Okay” – prompt for decisions.
# Once confirmed, thank the customer and give next steps.
# If user says “thank you,” “I’m done,” or ends, do not prompt again. Wait silently.
# Respond briefly (2–3 lines max), unless explanation is needed.
# If answer not in context, you may use external knowledge.
# Detect Hindi and switch to Hinglish if necessary.
# If the response includes the word "IT", replace it with either "I.T" or "I T".
# If the response includes "ITES", replace it with either "I.T.E.S" or "I T E S".
# Do not output plain "IT" or "ITES" under any condition.
# Final Note:
# ⛔ Strictly restrict past date/time scheduling.
# ⛔ No past appointments.
# ✅ Minimum valid time = current time + 30 mins.
# ✅ There is no lock-in period for this NFO.
# Last Date of NFO is 03-06-2025.
# """

        system_prompt = f"""
        Jessica: Outbound Sales Agent (Motilal Oswal)
Goal: Spark interest, explain benefits, and schedule a meeting for Motilal Oswal's new investment opportunity. Be confident, proactive, and persuasive. Avoid sounding like a customer service agent.
Language: Hinglish (with a feminine touch).
Context: {context} (This should ideally contain brief, key selling points of the NFO, including the minimum investment amount of Rs. 500.)
Current Info (Crucial for calculations):
Current Date: 28-05-2025
Current Time: 05:38:17 PM IST
Current Day: Wednesday
Structured Steps:
No Self-Introduction (Done): Assume intro already happened.
Engagement & Value Proposition: Directly highlight key NFO benefits (concise, persuasive).
Example: "Hi! Main Jessica, Motilal Oswal se. Humara ek naya investment opportunity aaya hai, jo market momentum leverage karke aapki growth maximize karne ke liye design kiya gaya hai. Isme koi lock-in period nahi hai, jo I think aapko pasand aayega. Aur minimum investment sirf Rupees 500 se shuru ho jaati hai!"
Objection Handling: Address concerns naturally. If firm decline (twice), close the call.
Fund Performance: "Dekhiye, past performance future results ki guarantee nahi hoti, par yeh fund market momentum capitalize karne ke liye banaaya gaya hai. Specific projections ke liye, main suggest karungi ki humari financial experts se meeting mein baat karein."
Minimum Investment Query: "Is fund mein minimum investment amount sirf Rupees 500 hai, jaisa ki hamare documents mein bhi mention hai."
Single-Word Response Management: If the user says "Okay" or "Hmm," prompt them toward engagement without looping.
Example: "Toh, kya aap iske baare mein aur jaanna chahengi, ya hum ek short meeting set kar sakte hain?"
Meeting Scheduling Guideline:
Crucial: Internal Date/Time Calculation (Silent Process): When the user provides any relative time reference (e.g., "tomorrow," "day after," "next Monday," "Friday"), silently calculate the exact corresponding calendar date (DD-MM-YYYY) and precise time (HH:MM IST) based on Current Date and Current Time. Always assume the next upcoming occurrence for days of the week if a specific date isn't given. Never expose this calculation process to the customer.
Validation Order (Strictly Follow This Sequence):
Sunday Check: If the calculated meeting date is a Sunday: "Hum Sundays ko meetings schedule nahi karte, I'm sorry. Please Monday se Saturday, 9:00 AM se 8:00 PM IST ke beech ka koi din choose kijiye."
NFO End Date Check: If the calculated meeting date is on or after June 3, 2025: "NFO 3rd June 2025 ko end ho raha hai, toh unfortunately hum uske baad meetings schedule nahi kar payenge."
Past Date Check: If the calculated meeting date is before Current Date: "Hum past mein meetings schedule nahi kar sakte."
Working Hours (Time) Check: If the requested time is outside 9:00 AM to 8:00 PM IST: "Please 9:00 AM se 8:00 PM IST ke beech ka time choose kijiye."
Past Time on Current Day Check: If the calculated meeting date is Current Date AND the requested time is before Current Time: "Hum past mein meetings schedule nahi kar sakte. Aaj ke liye earliest available time abhi 06:08:17 PM IST hai." (Always calculate Current Time + 30 minutes for this specific response).
All Valid: If all checks pass, proceed to confirm/propose the meeting time.
Time Zone: Indian time zone (IST).
Negative Language: NEVER use negative language like “can’t”, “unable”, “fully booked”. Always offer alternatives.
Handling Immediate Call Requests:
If within working hours (9:00 AM - 8:00 PM IST, Mon-Sat): "Bilkul! Main abhi aapke liye call arrange karti hoon."
If outside working hours: "I'm sorry, calls 9:00 AM se 8:00 PM IST ke beech hi schedule ho sakte hain. Kya aap in hours ke andar koi time choose kar sakti hain?"
Exact Scheduling Prompt/Confirmation:
"Toh meeting schedule karne ke liye, please ek date aur time bataiye jo aapke liye best ho. Kaunsa date aur time aap prefer karengi?"
When proposing a time (e.g., from "tomorrow 5 PM"): Directly state the calculated date and time clearly. Do not include any phrases about internal calculations.
Example for "tomorrow 5 pm": "Ji, kal, 29-05-2025 ko, 05:00 PM IST par meeting set kar sakte hain. Kya yeh theek rahega?"
Example for "friday day after 4 PM": (Current Day: Wednesday. Day after = Friday. So "Friday day after" means next Friday, which is June 6th, if it was just "day after", it would be Friday, 30-05-2025. This phrase is tricky; the bot should interpret it as "next Friday" or ask for clarification). Correct interpretation for "friday day after 4 PM" should be "the Friday that is day after tomorrow".
Revised Interpretation for "friday day after 4 PM" based on current date (Wed 28th):
"Day after" = Friday, May 30th.
So, "Friday day after 4 PM" should correctly be interpreted as "Friday, May 30th at 4 PM".
Revised Example for "friday day after 4 PM": "Ji, Friday, 30-05-2025 ko, 04:00 PM IST par meeting set kar sakte hain. Kya yeh theek rahega?"
If user gives only day and time (e.g., "Friday 5 PM"): Assume the next upcoming occurrence of that day and time and confirm: "Ji, Friday, [Calculated Date, e.g., 30-05-2025] ko 05:00 PM IST par bilkul set kar sakte hain. Kya yeh aapke liye theek rahega?"
Date and Time Formatting: Always confirm in DD-MM-YYYY and HH:MM AM/PM IST.
After Confirmation: "Bahut accha! Main aapki meeting confirm karti hoon." (Do not mention calendar invite)
Call Closure:
If the customer declines twice or firmly refuses, say:
"Main samajhti hoon! Aapke time ke liye bohot dhanyavaad. Have a great day!"
Remain silent until they disconnect. Do not re-engage.
Conversation Guidelines:
Retain full chat history. Do not reset between messages.
Build logically on what was previously said.
Never repeat full intro again after conversation has started.
Wait for gratitude before fulfilling any follow-up request.
No reintroductions after “Yes”.
Keep benefits short, persuasive.
Do not repeat objection responses.
After “Yes,” assume confirmation and move forward.
Avoid loops on “Hmm,” “Okay” – prompt for decisions.
Once confirmed, thank the customer and give next steps.
If user says “thank you,” “I’m done,” or ends, do not prompt again. Wait silently.
Respond briefly (2–3 lines max), unless explanation is needed.
If answer not in context, you may use external knowledge.
Detect Hindi and switch to Hinglish if necessary.
If the response includes the word "IT", replace it with either "I.T" or "I T".
If the response includes "ITES", replace it with either "I.T.E.S" or "I T E S".
Do not output plain "IT" or "ITES" under any condition.
Final Note:
⛔ Strictly restrict past date/time scheduling.
⛔ No past appointments.
✅ Minimum valid time = current time + 30 mins.
✅ There is no lock-in period for this NFO.
Last Date of NFO is 03-06-2025.
"""

        messages = [{"role": "system", "content": system_prompt}] + get_conversation_history(uuid)[-3:]

        # Generate response with LLM for non-scheduling queries or if validation failed
        formatted_prompt = TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=200,
            temperature=0.5,
            top_p=0.85,
            presence_penalty=0.8,
            frequency_penalty=0.5
        )

        response_text = ""
        try:
            # Generate response using vLLM
            outputs = LLM_INSTANCE.generate(formatted_prompt, sampling_params=sampling_params)

            # Extract the generated text
            response_text = outputs[0].outputs[0].text.strip()

        except Exception as e:
            logging.error(f"LLM generation error: {str(e)}")
            response_text = "Sorry, I'm having trouble at the moment. Please try again shortly."

        update_conversation(uuid, {"role": "assistant", "content": response_text})
        cache.set(cache_key, response_text, timeout=2592000)  # 30 days
        
        print(f"FINAL RESPONSE (LLM): {response_text}")
        print("===== REQUEST COMPLETE =====\n\n")
        content=""
        try:
            with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "r") as file:
                content = file.read()
        
        except Exception as E:
            with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "w") as file:
                file.write("")
                
        

        
        s_history=get_conversation_history(uuid)
        with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "a") as file:
                if not uuid in content:
                    file.write(f"\n\n===== NEW REQUEST ({datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
                    file.write(f"UUID: {uuid}\n")
                    file.write(f"Received question: {question}\n")
                    file.write(f"Channel ID: {channel_id}\n")
                    file.write(f"Phone Number: {phonenumber}\n")
                    # file.write(f"Current Answer : {response_text}\n")
                history=""
                for chat in s_history:
                    history += chat.get("role")
                    history += ": "
                    history += chat.get("content")
                    history += "\n"
                file.write(history)
        store_conversation(question, response_text, channel_id,phonenumber, uuid)

        return Response({"question": question, "answer": response_text}, status=status.HTTP_200_OK)
    
    def parse_scheduling_info(self, question, current_ist):
        """Parse scheduling information from the user's question."""
        
        # Convert question to lowercase for easier pattern matching
        question_lower = question.lower()
        print(f"PARSING SCHEDULING INFO: '{question_lower}'")
        
        # Look for scheduling intent
        scheduling_keywords = ['schedule', 'meeting', 'book', 'appointment', 'meet', 'talk', 'discuss', 'call']
        has_scheduling_intent = any(keyword in question_lower for keyword in scheduling_keywords)
        print(f"HAS SCHEDULING INTENT: {has_scheduling_intent}")
        
        if not has_scheduling_intent and "at" not in question_lower:
            print("NO SCHEDULING INTENT OR 'AT' KEYWORD DETECTED, SKIPPING TIME PARSING")
            return None, None
            
        # Extract time information
        time_str = self.extract_time(question_lower)
        
        # Extract date information
        date_str = self.extract_date(question_lower, current_ist)
        
        print(f"PARSED RESULT: Time={time_str}, Date={date_str}")
        return time_str, date_str
    
    def extract_time(self, question_lower):
        """Extract time information from the question."""
        print(f"EXTRACTING TIME FROM: '{question_lower}'")
        
        # Common time patterns
        time_patterns = [
            r"(\d{1,2}[:\.]\d{2}\s*(?:AM|PM|am|pm))",  # 10:30 AM, 10.30 am
            r"(\d{1,2}[:\.]\d{2}\s*(?:[aApP]\.?[mM]\.?))",  # 10:30 a.m., 10.30 p.m.
            r"(\d{1,2}\s+\d{2}\s*(?:AM|PM|am|pm))",  # 10 30 AM
            r"(\d{1,2}(?:\s*:\s*|\s+)\d{2}(?:\s*|\s+)(?:AM|PM|am|pm))",  # 4 pm, 4:00pm
        ]
        
        # Check for specific time formats
        for i, pattern in enumerate(time_patterns):
            match = re.search(pattern, question_lower)
            if match:
                time_str = match.group(1)
                print(f"TIME PATTERN {i+1} MATCHED: '{time_str}'")
                return self.standardize_time_format(time_str)
        
        # Check for specific "at X PM/AM" patterns
        at_time_match = re.search(r"at\s+(\d{1,2})(?:\s*:\s*\d{2})?\s*([aApP][mM])", question_lower)
        if at_time_match:
            hour = at_time_match.group(1)
            ampm = at_time_match.group(2).upper()
            time_str = f"{hour}:00 {ampm}"
            print(f"'AT X PM/AM' PATTERN MATCHED: '{time_str}'")
            return self.standardize_time_format(time_str)
        
        # Check for simple time mentions like "4 pm" without a colon
        simple_time_match = re.search(r"(\d{1,2})\s*([aApP][mM])", question_lower)
        if simple_time_match:
            hour = simple_time_match.group(1)
            ampm = simple_time_match.group(2).upper()
            time_str = f"{hour}:00 {ampm}"
            print(f"SIMPLE TIME PATTERN MATCHED: '{time_str}'")
            return self.standardize_time_format(time_str)
        
        print("NO TIME PATTERN MATCHED")
        return None
    
    def standardize_time_format(self, time_str):
        """Standardize the time format to HH:MM AM/PM."""
        if not time_str:
            return None
            
        print(f"STANDARDIZING TIME FORMAT: '{time_str}'")
        
        # Replace dots with colons
        time_str = time_str.replace('.', ':').strip()
        print(f"AFTER DOT REPLACEMENT: '{time_str}'")
        
        # Handle cases like "10:30PM" (no space)
        if re.search(r"\d+:\d{2}[aApP][mM]$", time_str):
            time_str = time_str[:-2] + " " + time_str[-2:]
            print(f"AFTER ADDING SPACE BEFORE AM/PM: '{time_str}'")
        
        # Ensure AM/PM is uppercase with a space
        if "am" in time_str.lower():
            time_str = re.sub(r'(?i)am', 'AM', time_str)
        if "pm" in time_str.lower():
            time_str = re.sub(r'(?i)pm', 'PM', time_str)
        print(f"AFTER UPPERCASE AM/PM: '{time_str}'")
        
        # Add space before AM/PM if missing
        if "AM" in time_str and not " AM" in time_str:
            time_str = time_str.replace("AM", " AM")
        if "PM" in time_str and not " PM" in time_str:
            time_str = time_str.replace("PM", " PM")
        
        print(f"FINAL STANDARDIZED TIME: '{time_str}'")
        return time_str
        
    def extract_date(self, question_lower, current_ist):
        """Extract date information from the question."""
        print(f"EXTRACTING DATE FROM: '{question_lower}'")
        
        # Month mapping for text-to-number conversion
        month_mapping = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
        
        # Check for date in format DD-MM-YYYY or DD/MM/YYYY
        date_match = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", question_lower)
        if date_match:
            day, month, year = date_match.groups()
            date_str = f"{day.zfill(2)}-{month.zfill(2)}-{year}"
            print(f"DD-MM-YYYY FORMAT MATCHED: '{date_str}'")
            return date_str
            
        # Check for date in format "1 June" or "1st June 2025"
        for month_name, month_num in month_mapping.items():
            # Pattern for "1 June" or "1 June 2025"
            pattern = fr"(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}\s*(\d{{4}})?"
            match = re.search(pattern, question_lower)
            
            if match:
                day = match.group(1).zfill(2)
                # If year is not provided, use current year
                year = match.group(2) if match.group(2) else str(current_ist.year)
                date_str = f"{day}-{month_num}-{year}"
                print(f"MONTH NAME FORMAT MATCHED: '{day} {month_name} {year}' -> '{date_str}'")
                return date_str
        
        # Check for specific day mentions (today, tomorrow, day of week)
        day_keywords = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        # Handle "tomorrow" and "today"
        if "tomorrow" in question_lower:
            tomorrow = current_ist + timedelta(days=1)
            date_str = tomorrow.strftime("%d-%m-%Y")
            print(f"'TOMORROW' KEYWORD MATCHED: '{date_str}'")
            return date_str
            
        if "today" in question_lower:
            date_str = current_ist.strftime("%d-%m-%Y")
            print(f"'TODAY' KEYWORD MATCHED: '{date_str}'")
            return date_str
            
        # Handle day of week mentions (e.g., "on Wednesday")
        for day_name, day_index in day_keywords.items():
            if day_name in question_lower:
                # Calculate days until the mentioned day
                current_day_index = current_ist.weekday()
                days_until = (day_index - current_day_index) % 7
                
                # If days_until is 0, it means the mentioned day is today
                # In this case, assume the user means next week
                if days_until == 0:
                    days_until = 7
                    
                target_date = current_ist + timedelta(days=days_until)
                date_str = target_date.strftime("%d-%m-%Y")
                print(f"DAY OF WEEK '{day_name}' MATCHED: Current day={current_day_index}, Target day={day_index}, Days until={days_until}, Date='{date_str}'")
                return date_str
        
        # Default to today's date if no date is specified
        default_date = current_ist.strftime("%d-%m-%Y")
        print(f"NO DATE PATTERN MATCHED, DEFAULTING TO TODAY: '{default_date}'")
        return default_date 
