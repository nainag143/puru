import os
import time
import torch
import datetime
import logging
import hashlib
import pymysql
import json
import pandas as pd
from datetime import datetime, timedelta

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.cache import cache

# Langchain and other imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2Tokenizer
from openai import OpenAI

# --- Configuration (Move these to Django settings or environment variables) ---
# Ensure PyTorch is using GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {DEVICE}\n")

# File Paths (Ideally, manage these through Django's MEDIA_ROOT or similar)
TEXT_FILE_PATH = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/output.txt"
PDF_FILE_PATH = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund_and_Services_FAQ_2_doc_merged_2025_05_16_V3.pdf"

# Set OpenAI API key (CRITICAL: Use environment variables in production)
api_key = os.getenv("OPENAI_API_KEY", "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")
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Suppress HuggingFace tokenizer warning
openai_client=OpenAI(api_key=api_key)

# Database Configuration (CRITICAL: Use environment variables in production)
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "passwd": "passw0rd",
    "database": "voice_bot"
}

# Chatbot specific configurations
MAX_TOKENS_CONTEXT = 800
CONVERSATION_HISTORY_LIMIT = 5 # Keep last N exchanges (user + assistant)
NFO_LAST_DATE = datetime(2025, 6, 3, 20, 0, 0) # June 3rd, 2025, 8:00 PM IST

# --- Global Initialization (Run once on server start) ---
# This part is fine to run globally as it loads static resources.
try:
    with open(TEXT_FILE_PATH, 'rb') as f:
        content = f.read()
        state_of_the_union = content.decode('utf-8', errors='ignore')
    print("Text document loaded.")
except Exception as e:
    print(f"Error loading text document: {e}")
    state_of_the_union = "" # Ensure it's not None

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.create_documents([state_of_the_union])
print("Text splitter initialized.")

hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)
print("HuggingFace Embeddings initialized.")

try:
    loader = PyPDFLoader(PDF_FILE_PATH)
    docs = loader.load()
    embeddings = hf.embed_documents([doc.page_content for doc in docs])
    print("PDF loaded and embeddings created.")
except Exception as e:
    print(f"Error loading PDF or creating embeddings: {e}")
    docs = []
    embeddings = []

# ChromaDB Setup - ensure 'vdb' directory exists and is writable
CHROMADB_PERSIST_DIR = 'vdb'
try:
    client = chromadb.PersistentClient(CHROMADB_PERSIST_DIR)
    collection = client.get_or_create_collection('test-2')
    if not collection.get()['ids'] and docs: # Only add if collection is empty and docs are available
        print("Populating ChromaDB collection...")
        collection.add(
            ids=[str(uuid4()) for _ in docs],
            documents=[doc.page_content for doc in docs],
            embeddings=embeddings
        )
    db = Chroma(collection_name='test-2', persist_directory=CHROMADB_PERSIST_DIR, embedding_function=hf)
    print("ChromaDB initialized and ready.")
except Exception as e:
    print(f"Error setting up ChromaDB: {e}")
    db = None # Set to None to handle gracefully later

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("GPT2 Tokenizer loaded.")

# --- Utility Functions ---

def truncate_context(context, max_tokens=MAX_TOKENS_CONTEXT):
    tokens = tokenizer.encode(context)
    return tokenizer.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else context

def get_db_connection():
    """Establishes a new database connection."""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as e:
        logging.error(f"Database connection error: {e}")
        return None

# --- Conversation Management with Django Cache ---
# The cache acts as your per-session state store.

def update_conversation(uuid, entry):
    """Adds a new message to a user's conversation history in cache."""
    key = f"conversation:{uuid}"
    conversation = cache.get(key, [])
    conversation.append(entry)
    # Trim conversation history to the last N exchanges (e.g., 5 user-bot pairs)
    # Each entry is a dict, so we consider the total length of the list.
    # If CONVERSATION_HISTORY_LIMIT is 5, it means 5 user messages and 5 bot responses.
    if len(conversation) > CONVERSATION_HISTORY_LIMIT * 2:
        conversation = conversation[-(CONVERSATION_HISTORY_LIMIT * 2):]
    cache.set(key, conversation, timeout=2592000) # 30 days timeout

def get_conversation_history(uuid):
    """Retrieves a user's conversation history from cache."""
    key = f"conversation:{uuid}"
    return cache.get(key, [])

def delete_conversation_history(uuid):
    """Deletes a user's conversation history from cache."""
    key = f"conversation:{uuid}"
    cache.delete(key)
    logging.info(f"Conversation history for UUID {uuid} deleted from cache.")


# --- API Views ---

class ChatBot_View(APIView):
    def post(self, request):
        question = str(request.data.get('question', '')).strip()
        channel_id = str(request.data.get('channel_id', '')).strip()
        phonenumber = str(request.data.get('phonenumber', '')).strip()
        uuid = str(request.data.get('uuid', '')).strip()
        call_disconnect = request.data.get('call_disconnect', False) # Default to False

        if not uuid:
            return Response({"error": "UUID is required for conversation tracking."}, status=status.HTTP_400_BAD_REQUEST)

        # --- Handle Call Disconnect ---
        if call_disconnect:
            logging.info(f"Call disconnected for UUID: {uuid}. Processing disposition.")
            conn = None # Initialize conn to None for finally block
            try:
                conn = get_db_connection()
                if not conn:
                    raise Exception("Failed to connect to database for disposition.")

                # Retrieve conversation from DB (if you store full conversations there)
                # Your SQL query seems to group by UUID, which is good.
                conversation_df = pd.read_sql(
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
                    conn
                )

                if not conversation_df.empty:
                    conversation = conversation_df.iloc[0].to_dict()
                    conversation_data = conversation['all_conversation']
                    conversation_id = conversation['id']

                    # --- Get Disposition from OpenAI ---
                    disposition_prompt = f"""
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
                    response_disposition = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for classifying customer call dispositions."},
                            {"role": "user", "content": disposition_prompt}
                        ],
                        max_tokens=50, # Shorter max_tokens for disposition
                        temperature=0.7,
                        top_p=0.5
                    )
                    disposition = response_disposition.choices[0].message.content.strip()
                    logging.info(f"UUID {uuid} - Predicted Disposition: {disposition}")

                    schedule_date = None
                    # --- Check for Meeting Scheduled and extract time ---
                    if "Meeting Scheduled".lower() in disposition.lower():
                        current_datetime_str = datetime.now().strftime("%d/%m/%Y %I:%M %p")
                        meeting_time_prompt = f"""
                        You are an intelligent assistant skilled at analyzing customer conversations to schedule meetings. Your task is to extract the final confirmed meeting **date and time** between a customer and a bot.

                        Guidelines:
                        - Read the full conversation carefully.
                        - Identify the **final meeting date and time** that both the customer and bot agree on.
                        - If the customer says vague terms like "tomorrow", "day after", "next Monday", or just a time like "4 PM", use the following current date and time as the reference:
                        Current Date and Time: {current_datetime_str}
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
                        response_meeting_time = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant for extracting meeting times."},
                                {"role": "user", "content": meeting_time_prompt}
                            ],
                            max_tokens=50, # Shorter max_tokens for time extraction
                            temperature=0.7,
                            top_p=0.5
                        )
                        meeting_time_raw = response_meeting_time.choices[0].message.content.strip()
                        if meeting_time_raw.lower() != "null":
                            try:
                                # Attempt to parse the date and time, handle errors
                                schedule_date = datetime.strptime(meeting_time_raw, "%d/%m/%Y %I:%M %p").strftime("%Y-%m-%d %H:%M:%S")
                                logging.info(f"UUID {uuid} - Extracted Meeting Time: {meeting_time_raw}")
                            except ValueError:
                                logging.warning(f"UUID {uuid} - Failed to parse meeting time: {meeting_time_raw}")
                                schedule_date = None
                        else:
                            logging.info(f"UUID {uuid} - No specific meeting time extracted.")

                    # --- Update Database with Disposition and Schedule Date ---
                    with conn.cursor() as cur:
                        if schedule_date:
                            update_query = """
                                UPDATE your_table_name
                                SET disposition = %s, schedule_date = %s
                                WHERE id = %s;
                            """
                            cur.execute(update_query, (disposition, schedule_date, conversation_id))
                        else:
                            update_query = """
                                UPDATE your_table_name
                                SET disposition = %s
                                WHERE id = %s;
                            """
                            cur.execute(update_query, (disposition, conversation_id))
                        conn.commit()
                        logging.info(f"UUID {uuid} - Database updated with disposition: {disposition} and schedule_date: {schedule_date}.")

                else:
                    logging.warning(f"UUID {uuid} - No conversation data found in DB to process for disposition.")

            except Exception as e:
                logging.error(f"Error during call disconnect processing for UUID {uuid}: {e}")
                # You might want to log the error to a separate system or dead-letter queue for investigation
            finally:
                if conn:
                    conn.close()
                # Always delete conversation history from cache on disconnect, regardless of DB success
                delete_conversation_history(uuid)
                return Response({"question": "", "answer": "Call Disconnected"}, status=status.HTTP_200_OK)

        # --- Handle Active Conversation ---
        # Get conversation history for the specific UUID from cache
        conversation_history = get_conversation_history(uuid)

        # Basic greeting logic (can be made more sophisticated by LLM)
        if "hello" in question.lower() and conversation_history: # Check if conversation already exists
            answer = "Hi there! I'm here. How can I assist you further?"
        else:
            # Perform similarity search with context
            context_docs_scores = []
            if db:
                context_docs_scores = db.similarity_search_with_relevance_scores(question, k=1)
            
            context = ""
            if context_docs_scores:
                context = "".join(doc.page_content for doc, _ in context_docs_scores)
                context = truncate_context(context) # Truncate context if too long

            # Prepare dynamic prompt variables
            current_date = datetime.now().strftime("%d-%m-%Y")
            time_current = datetime.now().strftime("%I:%M %p")
            current_day = datetime.now().strftime("%A") # Full weekday name

            qna_prompt = f"""
            You are an outbound sales agent named Jessica, calling on behalf of Motilal Oswal to introduce a new investment opportunity. Your goal is to spark interest, explain the benefits of the investment, and schedule a meeting. Your responses should be confident, proactive, and persuasive, aimed at generating interest and moving the conversation towards scheduling a meeting. Do not act like a customer service agent. When asked, "Why are you calling me?" do not provide generic answers. Instead, clearly explain the purpose of your call and how the opportunity benefits the customer.

            Refer this context to answer: {context}

            The conversation should move forward without restarting once the customer confirms a meeting time and date.

            Ensure that responses are naturally aligned with the ongoing discussion, maintaining context and flow. If the conversation exceeds token limits, summarize previous exchanges or trim older messages to retain the most relevant context. Do not prompt the user for information unless necessary, and allow the conversation to flow organically.

            Current Date : {current_date}
            Current Time : {time_current}
            Today is : {current_day}

            Follow these structured steps:

            1️⃣ Warm Introduction: Greet the customer {datetime.now().strftime("%I:%M %p")}, introduce yourself confidently, and directly state the fund’s name and its purpose in the first message.
            Example: "Good afternoon! I’m Jessica, calling from Motilal Oswal to introduce our new investment opportunity. This fund investment to maximize growth by leveraging market momentum. Would you be interested in learning more about its benefits?"
            If the customer interrupts at the beginning of the call, proceed with answering their question and do not repeat your introduction.

            2️⃣ Engagement & Value Proposition: Highlight the fund's key benefits concisely and persuasively.

            3️⃣ Objection Handling: Address concerns in a natural, non-repetitive way. Avoid pushing too hard if they firmly decline.

            4️⃣ Single-Word Response Management: If the user says "Okay" or "Hmm," prompt them toward engagement without looping.

            5️⃣ Meeting Scheduling Guideline:

            Meeting Time Restrictions:
            Meeting Scheduling time zone:
                - Follow the Indian time zone to schedule a meeting.

            Meeting Scheduling Date:
                - Schedule meeting only before {NFO_LAST_DATE.strftime("%dth %B %Y %I:%M %p")} IST only.
                - if the Meeting date after {NFO_LAST_DATE.strftime("%dth %B %Y %I:%M %p")} IST "We do not schedule meetings after the NFO ends, which is {NFO_LAST_DATE.strftime("%dth %B %Y")}."

            Working Hours/Days:
            - We schedule meetings from Monday to Saturday between 9:00 AM and 8:00 PM IST only.

            Explicit Sunday Restriction:
            - If the user requests Sunday: "We do not schedule meetings on Sundays. Please choose a day from Monday to Saturday between 9:00 AM and 8:00 PM IST."

            Implicit Sunday Restriction:
            - If today is Friday and they say "day after tomorrow", and that’s Sunday: "Sorry, we do not schedule meetings on Sundays. Please pick a different day, Monday to Saturday."

            Time Validation Rules:
            - If the **time is before current time on the same day**: "We can’t schedule meetings in the past. The earliest available time is {(datetime.now() + timedelta(minutes=30)).strftime("%I:%M %p")} IST today."

            - If the **time is outside 9:00 AM–8:00 PM IST**: "Please choose a time between 9:00 AM and 8:00 PM IST."


            Fund Performance Queries:
            - "While past performance doesn't guarantee future results, this fund follows a strategy that aims to capitalize on market momentum. For specific projections, I'd recommend discussing with our financial experts in our scheduled meeting."

            ✅ Never use negative language like “can’t”, “unable”, “fully booked”. Always offer alternatives.

            2. Fund Introduction Before Scheduling
            Do not say “if you're interested” before explaining. Instead, introduce directly:
            "This fund is designed to maximize growth by leveraging market momentum."

            3. Date and Time Formatting:
            "Next Monday is 11-03-2025. I can schedule a meeting for you at 03:00 PM IST. Does that work for you?"

            4. Exact Scheduling Prompt:
            "To schedule a meeting, please let me know a date and time that works best for you. What date and time would you prefer?"

            5. Handling Immediate Call Requests:
            - "Sure! I'll arrange a call for you right away."
            - "Got it! I'll schedule a call for you at [current time + 30 minutes] IST."
            - "I'm sorry, but calls can only be scheduled between 9:00 AM and 8:00 PM IST. Could you choose a time within these hours?"

            6️⃣ Call Closure:
            If the customer declines twice or firmly refuses, say:
            "I understand! Thank you for your time. Have a great day!"
            Remain silent until they disconnect. Do not re-engage.

            Conversation Guidelines:
            - Start with introduction only once.
            - No reintroductions after “Yes”.
            - Keep benefits short, persuasive.
            - Do not repeat objection responses.
            - After “Yes,” assume confirmation and move forward.
            - Avoid loops on “Hmm,” “Okay” – prompt for decisions.
            - Always confirm date & time in DD-MM-YYYY and HH:MM AM/PM IST format.
            - Once confirmed, thank the customer and give next steps.
            - If user says “thank you,” “I’m done,” or ends, **do not prompt again. Wait silently.**
            - Respond briefly (2–3 lines max), unless explanation is needed.
            - If answer not in context, you may use external knowledge.
            - Detect Hindi and switch to Hinglish if necessary.
            - If the response includes the word "IT", replace it with either "I.T" or "I T".
            - If the response includes "ITES", replace it with either "I.T.E.S" or "I T E S".
            Do not output plain "IT" or "ITES" under any condition.

            #Contextual Behavior:
            - Retain full chat history. Do not reset between messages.
            - Build logically on what was previously said.
            - Never repeat full intro again after conversation has started.
            - Wait for gratitude before fulfilling any follow-up request.

            Do NOT say or imply that you are sending a calendar invite.
            You are only confirming the meeting verbally in this conversation.
            Do NOT say: "I’ve sent a calendar invite", "You’ll receive a calendar", "Check your email", or similar.
            Instead, confirm with: "Great! I've noted that down. You'll receive a confirmation shortly from our team."

            Final Note:
            ⛔ Strictly restrict past date/time scheduling.
            ⛔ No past appointments.
            ✅ Minimum valid time = current time + 30 mins.
            ✅ There is no lock in period for this NFO.
            Last Date of NFO is {NFO_LAST_DATE.strftime("%d-%m-%Y")}.
            """

            # Append the current user message to the conversation history *before* calling the LLM
            update_conversation(uuid, {"role": "user", "content": question})

            # Retrieve the updated conversation history (last N exchanges for context)
            chat_history_for_llm = get_conversation_history(uuid)
            # You might want to limit the history passed to the LLM to manage token count
            # e.g., chat_history_for_llm = chat_history_for_llm[-CONVERSATION_HISTORY_LIMIT * 2:]

            messages = [{"role": "system", "content": qna_prompt}] + chat_history_for_llm

            try:
                response_llm = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo", # or "gpt-3.5-turbo-1106" if preferred
                    messages=messages,
                    max_tokens=150,
                    temperature=0.7,
                    top_p=0.5,
                )
                answer = response_llm.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"OpenAI API error for UUID {uuid}: {e}")
                answer = "I'm sorry, I'm having trouble connecting right now. Please try again in a moment."
            except Exception as e:
                logging.error(f"Unexpected error during LLM call for UUID {uuid}: {e}")
                answer = "An unexpected error occurred. Please try again."

            # Update conversation history with the assistant's response
            update_conversation(uuid, {"role": "assistant", "content": answer})

        # --- Database Insertion for each turn ---
        # It's better to store each question-answer pair as it happens
        conn = None # Initialize conn to None for finally block
        try:
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    # Sanitize inputs to prevent SQL injection (though prepared statements are better)
                    clean_question = question.replace("'", "''")
                    clean_answer = answer.replace("'", "''")
                    insert_query = f"""
                        INSERT INTO your_table_name (question, answer, channelid, phonenumber, uuid, createddate)
                        VALUES('{clean_question}' ,'{clean_answer}' ,'{channel_id}' ,'{phonenumber}' ,'{uuid}', NOW());
                    """
                    cur.execute(insert_query)
                    conn.commit()
                    logging.info(f"UUID {uuid} - Turn stored in DB. Question: '{question[:50]}...', Answer: '{answer[:50]}...'")
            else:
                logging.warning(f"UUID {uuid} - Could not connect to DB to store conversation turn.")
        except Exception as e:
            logging.error(f"Error inserting conversation turn into DB for UUID {uuid}: {e}")
        finally:
            if conn:
                conn.close()

        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)


class RemoveConversation_View(APIView):
    def post(self, request):
        uuid_to_remove = str(request.data.get('uuid')).strip() # Changed from channel_id to uuid for consistency
        if not uuid_to_remove:
            return Response(
                {"error": "uuid is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        delete_conversation_history(uuid_to_remove) # Call the dedicated function

        # Verify removal (optional, for debugging)
        if not get_conversation_history(uuid_to_remove):
            return Response(
                {"message": f"Conversation for UUID {uuid_to_remove} has been removed from cache."},
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"message": f"Failed to remove conversation for UUID {uuid_to_remove} from cache (might not have existed)."},
                status=status.HTTP_404_NOT_FOUND # Or 500 if removal failed unexpectedly
            )


class MonitorActiveCalls_View(APIView):
    def get(self, request):
        # This approach depends on how Django's cache backend stores keys.
        # For LocalMemCache, _cache is accessible, but for others (Redis, Memcached),
        # you'd need to use `cache.keys()` if available and configured,
        # or directly query the cache backend if it exposes such functionality.
        # This is generally NOT reliable for external cache backends.
        
        # A more robust approach for counting active sessions for external caches:
        # If your cache keys are prefixed like "conversation:{uuid}", you can't
        # directly list all keys unless the cache backend supports pattern matching (like Redis).
        # For generic Django cache, you might need to track active UUIDs in a separate,
        # more persistent store (e.g., a simple Django model or a dedicated Redis set).
        
        # For demonstration with LocalMemCache (not recommended for production scale)
        active_uuids = []
        try:
            # This is specific to `django.core.cache.backends.locmem.LocMemCache`
            # For other backends (Redis, Memcached), cache.keys() might not exist or be efficient.
            all_cache_keys = cache._cache.keys() 
            for key in all_cache_keys:
                if key.startswith("conversation:"):
                    uuid = key.split(":")[1]
                    active_uuids.append(uuid)
        except AttributeError:
            # Handle cases where cache._cache is not directly accessible (e.g., Redis backend)
            # You might need to implement a custom way to track active sessions
            logging.warning("Cannot directly inspect cache keys. MonitorActiveCalls_View may not be accurate for your cache backend.")
            # Fallback: Perhaps try a very broad search if your backend supports it, or
            # maintain a separate list of active UUIDs if precise counting is crucial.
            # Example for Redis: active_uuids = [k.decode().split(":")[1] for k in cache.client.keys("conversation:*")]
            
        active_count = len(active_uuids)

        if 0 <= active_count <= 50:
            in_memory_status = "normal"
        elif 51 <= active_count <= 80:
            in_memory_status = "warning"
        else:
            in_memory_status = "critical"

        response = {
            "active_conversations_count": active_count,
            "cache_memory_status": in_memory_status,
            "active_uuids_list": active_uuids, # Be cautious returning full lists in production
        }

        return Response(response, status=status.HTTP_200_OK)
