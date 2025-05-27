from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework import status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from zoneinfo import ZoneInfo  # Replaces pytz
import re
from datetime import datetime, timedelta
import pytz

import logging
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
import logging

# Import the LLM class with streaming support
from vllm.utils import Counter
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from sales_app.llm_instance import LLM_INSTANCE

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
    print("Storing Conversation")
    try:
        conn = connection()
        
        cur = conn.cursor()
        question = question.replace("'" , " ")
        answer = answer.replace("'" , " ")
        query = f"""INSERT INTO your_table_name (question, answer, channelid, phonenumber, uuid) VALUES('{question}' ,'{answer}' ,'{channelid}' ,'{phonenumber}' ,'{uuid}');"""
        
        cur.execute(query)
        conn.commit()
        print("Conversation Stored")
                
    except Exception as e:
        print(f"Error Database Connection Error : {e}")
    
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



class Motilal_ChatBot_View(APIView):
    def post(self, request):
        question = str(request.data.get('question', '')).strip()
        channel_id = str(request.data.get('channel_id', '')).strip()
        phonenumber = str(request.data.get('phonenumber', '')).strip()
        uuid = str(request.data.get('uuid', '')).strip()
        call_disconnect = request.data.get('call_disconnect')

        print(f"\n\n===== NEW REQUEST =====")
        print(f"QUESTION: {question}")
        logging.info(f"Received question: {question}")
        logging.info(f"Channel ID : {channel_id}")
        logging.info(f"uuid : {uuid}")

        with open("/root/BotHub_llm_model/llm_agent_project/motilal_app/data_check.txt", "a") as file:
            file.write("\n\n===== NEW REQUEST =====")
            file.write(f"Received question: {question}")
            file.write(f"Channel ID : {channel_id}")
            file.write(f"uuid : {uuid}")

        if call_disconnect is True:
            try:
                logging.info("Call Disconnected")
                # Get the conversation
                conn = connection()
                
                conversation = pd.read_sql(
                                    f"""
                                            SELECT 
                                                id,
                                                disposition,
                                                uuid, 
                                                GROUP_CONCAT(CONCAT('Customer: ', question, ' || Bot: ', answer) SEPARATOR ' , ') AS all_conversation,
                                                schedule_date
                                            FROM your_table_name  
                                            WHERE 
                                                uuid = '{uuid}'  
                                            GROUP BY uuid;
                                            """,
                                            conn)
            except Exception as e:
                logging.error(f"Error Database Connection Error : {e}")
            
            finally:
                conn.close()

            if len(conversation) > 0 :
                conversation = conversation.to_dict('records')[0]

                conversation_data = conversation['all_conversation']
                conversation_id = conversation['id']

                prompt = f"""
                    You are an expert at analyzing customer conversations. Given a dialogue between a customer and a bot, your task is to select the most appropriate disposition from the list below that best reflects the customer's intent at the end of the conversation.

                    List of Dispositions:
                    1. Meeting Scheduled  
                    2. Interested  
                    3. Not Interested  
                    4. Call Back  
                    5. Do Not Call Me  
                    6. Remove My Number  
                    7. DND  
                    8. DNC  
                    9. Stop Calling  
                    10. I Will Complain  

                    Instructions:
                    - Analyze the entire conversation carefully.
                    - Focus on the customer’s final intent or sentiment.
                    - Select and return only the disposition name from the list (e.g., "Interested", "Meeting Scheduled").
                    - Do NOT include numbering, quotes, markdown symbols like **, or any explanation — just the disposition name.

                    Conversation:
                    {conversation_data}

                    Your Response:
                """
                response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant for classifying customer call dispositions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7,
                    top_p=0.5
                )

                disposition = response.choices[0].message.content.strip()
                # print(f"3 - get the disposition")
                # print(f"Predicted Disposition: {disposition}\n")
                print("Disposition")
                logging.info(disposition)
                query = """
                    UPDATE your_table_name
                    SET disposition = %s
                    WHERE id = %s;
                """
                query_data = (disposition, conversation_id)

                # Checking Meeting Scheduled or not 
                if "Meeting Scheduled".lower() in disposition.lower():

                    # print(f"4 - Meeting Scheduled condition \n")
                    # Set the current datetime (you can dynamically fetch this as needed)
                    current_datetime = datetime.now().strftime("%d/%m/%Y %I:%M %p")

                    prompt = f"""
                        You are an intelligent assistant skilled at analyzing customer conversations to schedule meetings. Your task is to extract the final confirmed meeting **date and time** between a customer and a bot.

                        Guidelines:
                        - Read the full conversation carefully.
                        - Identify the **final meeting date and time** that both the customer and bot agree on.
                        - If the customer says vague terms like "tomorrow", "day after", "next Monday", or just a time like "4 PM", use the following current date and time as the reference:
                        Current Date and Time: {current_datetime}
                        - Resolve such vague expressions into a full date and time accordingly.
                        - Return only one final confirmed meeting date and time in this exact format:
                        "DD/MM/YYYY hh:mm AM/PM"
                        - If no final date and time is confirmed, return:
                        null

                        Here is the conversation:
                        {conversation_data}

                        Return ONLY one of the following:
                        1. A single line with date and time: "DD/MM/YYYY hh:mm AM/PM"
                        2. null
                        Do NOT include any extra words or explanation.
                    """

                    response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant for classifying customer call dispositions."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7,
                        top_p=0.5
                    )

                    # Extract and print result
                    meeting_time = response.choices[0].message.content.strip()
                    print(f"meeting_time : {meeting_time}")
                    logging.info("Meeting Time")
                    logging.info(meeting_time)
                    query = """
                        UPDATE your_table_name
                        SET disposition = %s, schedule_date = %s
                        WHERE id = %s;
                    """
                    query_data = (disposition, meeting_time ,conversation_id)


                # print(f"5 - Meeting not Scheduled outside condition \n")

                # # Update the if the meeting not schedules 
                try:
                    conn = connection()
                    
                    cur = conn.cursor()

                
                    cur.execute(query, query_data)
                    conn.commit()
                    logging.info("Inserted Disposition, Meeting Details in DB")

                except Exception as e:
                    logging.error(f"Error Database Connection Error : {e}")
                
                finally:
                    cur.close()
                    conn.close()

                delete_conversation(uuid)


                deleted_conversation = get_conversation_history(uuid)

                print(f"deleted_conversation : {deleted_conversation}")
                logging.info(f"deleted_conversation : {deleted_conversation}")
                return Response( {"question": "", "answer": "Call Disconnected"} , status=status.HTTP_200_OK)

            else:
                return Response( {"question": "", "answer": "Call Disconnected"} , status=status.HTTP_200_OK)


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
            # update the conversation in db
            store_conversation(question, farewell, channel_id, phonenumber, uuid)
            return JsonResponse({"question": question, "answer": farewell}, status=200)

        # Get context for the question
        result = db.similarity_search_with_relevance_scores(question, k=1)
        context = "".join(doc.page_content for doc, _ in result)

        # First time greeting - ONLY for new conversations with no history
        # handle the case where the first message might not be hello
        if not conversation_history:
            first_message = f"{greeting}! I'm Jessica from Motilal Oswal. Would you like to know about our new investment fund?"
            update_conversation(uuid, {"role": "user", "content": question})
            update_conversation(uuid, {"role": "assistant", "content": first_message})
            store_conversation(question, first_message, channel_id, phonenumber, uuid)
            return JsonResponse({"question": question, "answer": first_message}, status=200)

        if "hello" in question.lower() and len(conversation_history) > 0:
                answer = "Hi there! I'm here. How can I assist you further?"
                update_conversation(uuid, {"role": "user", "content": question})
                update_conversation(uuid, {"role": "content", "content": answer})
                store_conversation(question, answer, channel_id, phonenumber, uuid)
                return JsonResponse({"question": question, "answer": answer}, status=200)
        
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
#         system_prompt = f"""
# You are Jessica, an outbound calling agent from Motilal Oswal. You NEVER say you’re a financial advisor. Your role is to confidently pitch the Motilal Oswal Services Fund, answer initial questions using ONLY the RELEVANT CONTEXT below, and guide the customer to schedule a follow-up meeting.

# INTRODUCTION:
# - Track internally if you have introduced the fund yet (introduction_done = False initially).
# - On the first "hello" from the customer:
#     - If introduction_done is False:
#         - Say: "Hi, this is Jessica from Motilal Oswal! Our Services Fund targets India’s fastest-growing sectors, aiming for steady long-term growth. Have you had a chance to explore service sector investments before?"
#         - Set introduction_done = True.
# - For any subsequent "hello":
#     - Do NOT reintroduce yourself.
#     - Respond naturally like a human would (e.g., "Yes, I’m here!" or continue conversation).

# SPEAKING STYLE:
# - Be confident, energetic, and firmly lead the conversation.
# - Use brief, sharp sentences (1-2 max).
# - Use natural fillers sparingly, only if natural.
# - NEVER say “How can I assist you?” or similar.

# FLOW & ENGAGEMENT:
# - If customer shows clear interest in scheduling a meeting or explicitly asks to schedule:
#     - Do NOT repeat the fund pitch.
#     - Shift immediately to scheduling, e.g.:
#       "Great, what time works for you? We’re available Monday to Saturday, 9 AM to 8 PM IST."
# - If customer asks about the fund or related questions before showing scheduling intent:
#     - Answer precisely using RELEVANT CONTEXT.
#     - Then prompt for meeting if appropriate:
#       "Sounds good, shall we lock in a time to discuss this in detail?"
# - Avoid repetitive or monotonous phrasing.
# - Maintain control and keep the conversation active and engaging.

# ANSWERING QUESTIONS:
# - Use ONLY the RELEVANT CONTEXT to answer.
# - If the answer is not found in context, say:
#   "Hmm, that’s an insightful question. Based on what I see here, [answer from context]. Does that help clarify things?"
# - If still unclear or outside context:
#   "That’s a great question. Let’s schedule a quick call so I can get that answered for you."

# MEETING SCHEDULING:
# - If user requests meeting "now", "right now", or "asap", say:
#   "We need a bit of time to get things ready. We can do it after 30 minutes from now, around {time_plus_30} IST."
# - Never offer meetings on Sundays.
# - Never schedule meetings after 3rd June 2025 IST.
# - Validate meeting time is Mon-Sat, 9:00–20:00 IST, and date ≤ 3 June 2025 IST.
# - Always ask the customer for their preferred date/time; never schedule by yourself.

# MEETING CONFIRMATION:
# - When customer proposes a valid date and time:
#     - Say once:
#       "Awesome! Just to confirm, we have scheduled the call on {confirmed_date} at {confirmed_time_24h} IST. Is that correct?"
#     - If customer confirms positively:
#         - Say: "Thanks! I’ve noted that time and we’ll confirm it shortly."
#         - Then end the call (no further responses).
#     - If customer denies or suggests a different time:
#         - Ask again for their preferred time and validate.
# - If customer proposes a date beyond 3rd June 2025 IST:
#     - Say: "Sorry, we’re only scheduling meetings up to 3rd June 2025. Could you share an earlier time?"

# NEVER:
# - Repeat your introduction after the first time.
# - Say "How can I assist you?" or similar.
# - Schedule meetings yourself.
# - Speak after confirming a valid meeting time (except the confirmation message).
# - Go outside the provided context to answer questions.

# CURRENT INFO (in IST):
# Date: {current_date}
# Time: {time_current}
# Day: {current_day}
# Earliest allowed meeting time: {time_plus_30}

# RELEVANT CONTEXT:
# {context}
# """


        system_prompt = f'''
                You are Jessica, an outbound calling agent from Motilal Oswal. You NEVER say you’re a financial advisor. Your role is to confidently pitch the Motilal Oswal Services Fund, answer initial questions using ONLY the RELEVANT CONTEXT below, and guide the customer to schedule a follow-up meeting.
                
                ---
                
                ### Introduction
                
                * **Track internally**: `introduction_done = False` (initially).

                * **On first "hello"**:

                    * If `introduction_done` is `False`: "Hi, this is Jessica from Motilal Oswal! Our Services Fund targets India’s fastest-growing sectors, aiming for steady long-term growth. Have you had a chance to explore service sector investments before?"

                    * Set `introduction_done = True`.

                * **On subsequent "hello"**: Do NOT reintroduce. Respond naturally.
                
                ---
                
                ### Speaking Style
                
                * Be **confident, energetic, and firmly lead**.

                * Use **brief, sharp sentences** (1-2 max).

                * Use natural fillers sparingly.

                * **NEVER** say “How can I assist you?” or similar.
                
                ---
                
                ### Flow & Engagement
                
                * **If customer shows clear interest in scheduling or explicitly asks to schedule**:

                    * Do NOT repeat pitch.

                    * Shift immediately to scheduling: "Great, what time works for you? We’re available Monday to Saturday, 9 AM to 8 PM IST."

                * **If customer asks about the fund or related questions before showing scheduling intent**:

                    * Answer precisely using **RELEVANT CONTEXT**.

                    * Then prompt for meeting: "Sounds good, shall we lock in a time to discuss this in detail?"

                * **If customer says "I am not interested" or "doesn't want to continue"**: "Thank you for your time. Goodbye." (Then end the call).

                * Avoid repetitive phrasing. Maintain control and engagement.
                
                ---
                
                ### Answering Questions
                
                * Use **ONLY the RELEVANT CONTEXT**.

                * **If answer not in context**: "Hmm, that’s an insightful question. Based on what I see here, [answer from context]. Does that help clarify things?"

                * **If still unclear or outside context**: "That’s a great question. Let’s schedule a quick call so I can get that answered for you."
                
                ---
                
                ### Meeting Scheduling
                
                * **If user requests meeting "now", "right now", or "asap"**: "We need a bit of time to get things ready. We can do it after 30 minutes from now, around {time_plus_30} IST."

                * Never offer meetings on Sundays.

                * Never schedule meetings after June 3rd, 2025 IST.

                * **Validate meeting time**: Mon-Sat, 9:00 AM – 8:00 PM IST, and date $\le$ June 3rd, 2025 IST.

                * **Always ask the customer for their preferred date/time**; never schedule yourself.
                
                ---
                
                ### Meeting Confirmation
                
                * **When customer proposes a valid date and time**:

                    * Say once: "Awesome! Just to confirm, we have scheduled the call on {{confirmed\_date}} at {{confirmed\_time\_24h}} IST. Is that correct?"

                    * **If customer confirms positively**: (End the call).

                    * **If customer denies or suggests a different time**: Ask again for their preferred time and validate.

                * **If customer proposes a date beyond June 3rd, 2025 IST**: "Sorry, we’re only scheduling meetings up to 3rd June 2025. Could you share an earlier time?"
                
                ---
                
                ### NEVER
                
                * Repeat your introduction after the first time.

                * Say "How can I assist you?" or similar.

                * Schedule meetings yourself.

                * Speak after confirming a valid meeting time.

                * Go outside the provided context to answer questions.
                
                ---
                
                ### CURRENT INFO (in IST)
                
                Date: Tuesday, May 27, 2025

                Time: 07:42:12 PM

                Day: Tuesday

                Earliest allowed meeting time: 08:12:12 PM
                
                ---
                
                ### RELEVANT CONTEXT
                
                {context}
                
                '''



        messages = [{"role": "system", "content": system_prompt}] + get_conversation_history(uuid)

        # Generate response with LLM for non-scheduling queries or if validation failed
        formatted_prompt = LLM_INSTANCE.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.5,
            top_p=0.85,
            presence_penalty=0.8,
            frequency_penalty=0.5
        )

        response_text = ""
        try:
            for new_token, finished in LLM_INSTANCE.generate_stream(formatted_prompt, sampling_params):
                logging.info(new_token)
                response_text += new_token
            logging.info(response_text)
        except Exception as e:
            logging.error(f"LLM generation error: {str(e)}")
            response_text = "Sorry, I'm having trouble at the moment. Please try again shortly."

        update_conversation(uuid, {"role": "assistant", "content": response_text})
        cache.set(cache_key, response_text, timeout=2592000)  # 30 days
        
        print(f"FINAL RESPONSE (LLM): {response_text}")
        print("===== REQUEST COMPLETE =====\n\n")
        store_conversation(question, response_text, channel_id, phonenumber, uuid)
        
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
