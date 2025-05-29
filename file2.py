

# Added this code on 22 April 2025




from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from transformers import GPT2Tokenizer
from django.core.cache import cache
import os
import openai
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
from zoneinfo import ZoneInfo
 
# Ensure PyTorch is using GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}\n")
 
# File Paths
text_file = "/root/BotHub_llm_model/llm_agent_project_puru_8020/media/output.txt"
# pdf_file = "/var/www/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal RD inputs 03 march 2025 v14.pdf"
# pdf_file = "/var/www/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund - Details_14_May_2025_V1.pdf"
# pdf_file = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund_and_Services_FAQ_2_doc_merged_2025_05_16_V2.pdf"
pdf_file = "/root/Junaid/3_chatBot/chatBot_8004/media/Docs_for_GPT_Model/Motilal Oswal Services Fund_and_Services_FAQ_2_doc_merged_2025_05_16_V3.pdf"
 
# Set OpenAI API key


os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
# Load Text Document
with open(text_file, 'rb') as f:
    content = f.read()
    state_of_the_union = content.decode('utf-8', errors='ignore')
 
# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.create_documents([state_of_the_union])
 
# HuggingFace Embeddings
hf = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
 
# Load PDF and Create Embeddings
loader = PyPDFLoader(pdf_file)
docs = loader.load()
embeddings = hf.embed_documents([doc.page_content for doc in docs])
 
# ChromaDB Setup
client = chromadb.PersistentClient('vdb')
collection = client.get_or_create_collection('test-2')
if not collection.get()['ids']:
    collection.add(
        ids=[str(uuid4()) for _ in docs],
        documents=[doc.page_content for doc in docs],
        embeddings=embeddings
    )
db = Chroma(collection_name='test-2', persist_directory='vdb', embedding_function=hf)
 
# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
 
def truncate_context(context, max_tokens=800):
    tokens = tokenizer.encode(context)
    return tokenizer.decode(tokens[:max_tokens]) if len(tokens) > max_tokens else context
 
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
 
 
# @method_decorator(never_cache, name='dispatch')
class Local_ChatBot_View(APIView):
    def post(self, request):
        # print(f"\nrequest.data : {request.data}\n")
        
        question = str(request.data.get('question', '')).strip()
        channel_id = str(request.data.get('channel_id', '')).strip()
        phonenumber = str(request.data.get('phonenumber', '')).strip()
        uuid = str(request.data.get('uuid', '')).strip()
        call_disconnect = request.data.get('call_disconnect')
        print(call_disconnect)
        
        # print(f"\nQuestion : {question} ")
        # print(f"phonenumber : {phonenumber}")
        # print(f"call_disconnect : {call_disconnect}")
        # unique_id = str(request.data.get('unique_id', '')).strip()



        # Executes when call disconnects 
        if call_disconnect is True:
            # print(f"1 - Disconnect call condition ")
            # Get the conversation
            # 
            #  
            try:
                convo_history_list = get_conversation_history(uuid)
                
                with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "a", encoding="utf-8") as file:
                    for chat in convo_history_list:
                        print(chat)
                        print(type(chat))
                        role = chat["role"]
                        print(role)
                        content = chat["content"]
                        print(content)
                        file.write(f"{str(role)}: {str(content)}\n")
                        print("Done writing")


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
                print(f"Error Database Connection Error : {e}")
            
            finally:
                conn.close()

            if len(conversation) > 0 :
                conversation = conversation.to_dict('records')[0]

                conversation_data = conversation['all_conversation']
                conversation_id = conversation['id']

                # print(f"2 - get the conversation ")
                # print(f"conversation_data : {conversation_data}\n")

                # prompt = f"""
                # You are an expert at analyzing customer conversations. Given a dialogue between a customer and a bot, your task is to determine the most suitable disposition from the list below that reflects the customer's final intent in the conversation.

                # List of Dispositions:
                # 1. Meeting Scheduled  
                # 2. Interested  
                # 3. Not Interested  
                # 4. Call Back  
                # 5. Do Not Call Me  
                # 6. Remove My Number  
                # 7. DND  
                # 8. DNC  
                # 9. Stop Calling  
                # 10. I Will Complain  

                # Instructions:
                # - Carefully read the conversation.
                # - Choose the most appropriate disposition based on the customer’s final response.
                # - If the disposition is "Meeting Scheduled", also extract the meeting date and time from the conversation.
                # - Format the meeting time as "DD/MM/YYYY hh:mm AM/PM".
                # - Return the response strictly in JSON format:
                #     - If disposition is "Meeting Scheduled":
                #     {{
                #         "disposition": "Meeting Scheduled",
                #         "meeting_time": "11/03/2025 12:00 PM"
                #     }}
                #     - For any other disposition, return only:
                #     {{
                #         "disposition": "Disposition Name"
                #     }}

                # Here is the conversation:
                # {conversation_data}

                # Return only the JSON object.
                # """

                # # Call OpenAI API
                # response = openai.ChatCompletion.create(
                #     model="gpt-3.5-turbo",
                #     messages=[
                #         {"role": "system", "content": "You are a helpful assistant that classifies customer call dispositions."},
                #         {"role": "user", "content": prompt}
                #     ],
                #     max_tokens=200,
                #     temperature=0.7,
                #     top_p=0.5
                # )

                # # Extract and print
                # reply = response["choices"][0]["message"]["content"].strip()

                # try:
                #     result = json.loads(reply)
                #     print("Predicted Result:", result)
                # except json.JSONDecodeError:
                #     print("Failed to parse response as JSON. Raw response:")
                #     print(reply)





                # CAll Open AI to get the disposition 

                # prompt = f"""
                #             You are an expert at analyzing customer conversations. Given a dialogue between a customer and a bot, your task is to select the most appropriate disposition from the list below that best reflects the customer's intent at the end of the conversation.

                #             List of Dispositions:
                #             1. Meeting Scheduled  
                #             2. Interested  
                #             3. Not Interested  
                #             4. Call Back  
                #             5. Do Not Call Me  
                #             6. Remove My Number  
                #             7. DND  
                #             8. DNC  
                #             9. Stop Calling  
                #             10. I Will Complain  

                #             Instructions:
                #             - Analyze the entire conversation.
                #             - Focus on the customer’s final intent or sentiment.
                #             - Return only the most relevant disposition from the list above.
                #             - If multiple options seem close, select the one that best fits the overall tone and outcome.

                #             Conversation:
                #             {conversation_data}

                #             Your Response (one of the 10 dispositions listed above):
                #             """


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

                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a helpful assistant for classifying customer call dispositions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7,
                    top_p=0.5
                )

                disposition = response["choices"][0]["message"]["content"].strip()
                # print(f"3 - get the disposition")
                # print(f"Predicted Disposition: {disposition}\n")

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

                    response = openai.ChatCompletion.create(
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
                    meeting_time = response["choices"][0]["message"]["content"].strip()
                    print(f"meeting_time : {meeting_time}")

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

                except Exception as e:
                    print(f"Error Database Connection Error : {e}")
                
                finally:
                    cur.close()
                    conn.close()

                # print(f"6 - Data Inserted  \n")
                # print(f"query_data : {query_data},uuid : {uuid}")
                

                # Remove the conversation History 
                # remove_conversation_history(uuid)
                delete_conversation(uuid)

                return Response( {"question": "", "answer": "Call Disconnected"} , status=status.HTTP_200_OK)

            else:
                return Response( {"question": "", "answer": "Call Disconnected"} , status=status.HTTP_200_OK)

        # Generate a unique cache key based on the question
        cache_key = f"chatbot_response:{hashlib.md5(question.encode()).hexdigest()}"
        # cached_response = cache.get(cache_key)
        
        # If response is cached, return it instantly
        # if cached_response:
        #     return Response({"question": question, "answer": cached_response}, status=status.HTTP_200_OK)

        # print()
        # print(f"len(conversation_history) > 0 : {len(get_conversation_history(uuid)) > 0}")
        # print(f"'hello' in question.lower() : {'hello' in question.lower()}")
        # print()



        conversation_history = get_conversation_history(uuid)
        if "hello" in question.lower() and len(conversation_history) == 0:
            answer = """
                Hello ! I’m Jessica, calling from Motilal Oswal to introduce our new investment opportunity, the Motilal Oswal Services Fund. This fund is designed to maximize growth by leveraging market momentum. Would you be interested in learning more about its benefits?                    
                """
            update_conversation(uuid, {"role": "user", "content": question.strip()})
            update_conversation(uuid, {"role": "assistant", "content": answer.strip()})
        
        
        elif "hello" in question.lower() and len(conversation_history) > 0:
            
            answer = "Hi there! I'm here. How can I assist you further?"
            update_conversation(uuid, {"role": "user", "content": question})
            update_conversation(uuid, {"role": "assistant", "content": answer})



        else:
            result = db.similarity_search_with_relevance_scores(question, k=1)
            context = "".join(doc.page_content for doc, _ in result)
 
            greeting_time = int(time.strftime("%H"))
            
            # Define greeting based on time of day
            if greeting_time > 0 and greeting_time < 12:
                greeting = "Good morning"
            elif greeting_time >= 12 and greeting_time < 18:
                greeting = "Good afternoon"
            else:
                greeting = "Good evening"
                
            # Get tomorrow’s date dynamically in DD-MM-YYYY format
            current_date = time.strftime("%d-%m-%Y")
            time_current = time.strftime("%I:%M %p")
            
            # getting the current day 
            # Get current time in seconds since epoch
            current_time = time.time()

            # Convert to local time struct
            local_time = time.localtime(current_time)

            # Get weekday (0=Monday, 6=Sunday)
            weekday_num = local_time.tm_wday

            # Map weekday numbers to day names
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 
                        'Thursday', 'Friday', 'Saturday', 'Sunday']

            # Get current day name
            current_day = weekdays[weekday_num]

            # Working 
                
            #     """



            qna_prompt = f"""
C - Context    
Refer this context to answer: {context}

You are an AI assistant representing Motilal Oswal, focused on introducing a new investment opportunity fund designed to maximize growth by leveraging market momentum. Your primary goal is to engage potential customers, provide key information, handle objections, and schedule meetings with financial experts. You must operate within a conversational chat environment, maintaining natural flow and context.
 
 
Conversation Management:
 
Always align responses naturally with the ongoing discussion, preserving context and flow.

If conversation token limits are approached, summarize previous exchanges or trim older messages to retain the most relevant context.

Avoid prompting the user for information unless absolutely necessary, allowing the conversation to progress organically.

Retain full chat history and build logically on previous statements; do not reset conversation context.

R - Role

You are Jessica, a confident and helpful representative from Motilal Oswal. Your core responsibilities include:
 
Confidently introducing the fund and its purpose.

Concise and persuasive communication of the fund's benefits.

Handling customer queries and objections gracefully.

Efficiently scheduling follow-up meetings.

Maintaining a positive and professional demeanor throughout the interaction.

Assuming confirmation after a "Yes" and proceeding without re-introduction.

E - Engagement

Your engagement strategy should be proactive and customer-centric:
 
1. Warm Introduction:
 
Start with a greeting (e.g., "{greeting}").

Introduce yourself, the fund's name, and its purpose directly in the first message.

Example: "Good afternoon! I’m Jessica, calling from Motilal Oswal to introduce our new investment opportunity. This fund invests to maximize growth by leveraging market momentum. Would you be interested in learning more about its benefits?"

If interrupted, address the customer's question immediately and do not repeat the introduction.

Do not reintroduce yourself after the conversation has started.

2. Value Proposition:
 
Highlight the fund's key benefits concisely and persuasively. Do not use phrases like "if you're interested" before explaining the fund. Directly state: "This fund is designed to maximize growth by leveraging market momentum."

3. Objection Handling:
 
Address concerns naturally and avoid repetition.

Do not push too hard if the customer firmly declines.

Do not repeat objection responses.

4. Single-Word Response Management:

If the user says "Okay" or "Hmm," gently prompt them towards further engagement or a decision without looping.

Filler Prompts:

Use these brief, conversational phrases to acknowledge, confirm understanding, or transition naturally, especially when processing complex requests or ensuring the user is still engaged.
General Acknowledgment: "Alright," "Okay," "Understood," "Got it."
Confirming Understanding: "Let me just confirm," "So, to be clear..."
Thinking/Processing: "Just a moment," "Let me check on that for you."
Engaging during a pause: "Is there anything else I can clarify about that?" "Are you still there?"
Transitioning: "Moving on," "Next up..."
 
5. Immediate Call Requests:
 
For immediate calls, respond with: "Sure! I'll arrange a call for you right away." or "Got it! I'll schedule a call for you at [current time + 30 minutes] IST."

6. Language Adaptation:
 
Detect Hindi and seamlessly switch to Hinglish if necessary.

A - Action/Automation

Follow these structured steps and rules for all interactions, especially for scheduling:
 
1. Meeting Scheduling Guidelines:
 
Time Zone: All meetings must be scheduled in the Indian time zone (IST).

Date Restrictions:

Schedule meetings only before 3rd May 2025 at 8:00 PM IST.

If a meeting date is requested after 3rd May 2025 at 8:00 PM IST, state: "We do not schedule meetings after the NFO ends, which is 3rd June 2025."

Working Hours/Days:

Schedule meetings Monday to Saturday between 9:00 AM and 8:00 PM IST only.

Explicit Sunday Restriction: If the user requests Sunday, state: "We do not schedule meetings on Sundays. Please choose a day from Monday to Saturday between 9:00 AM and 8:00 PM IST."

Implicit Sunday Restriction: If today is Friday and "day after tomorrow" is requested (which is Sunday), state: "Sorry, we do not schedule meetings on Sundays. Please pick a different day, Monday to Saturday."

Time Validation Rules:

Past Time: If a time before the current time on the same day is requested, state: "We can’t schedule meetings in the past. The earliest available time is [current time + 30 minutes] IST today."

Outside Working Hours: If a time outside 9:00 AM–8:00 PM IST is requested, state: "Please choose a time between 9:00 AM and 8:00 PM IST."

Fund Performance Queries: When asked about past performance, state: "While past performance doesn't guarantee future results, this fund follows a strategy that aims to capitalize on market momentum. For specific projections, I'd recommend discussing with our financial experts in our scheduled meeting."

Scheduling Prompt: To initiate scheduling, ask: "To schedule a meeting, please let me know a date and time that works best for you. What date and time would you prefer?"

2. Formatting:
 
Always confirm date & time in DD-MM-YYYY and HH:MM AM/PM IST format.

Example: "Next Monday is 11-03-2025. I can schedule a meeting for you at 03:00 PM IST. Does that work for you?"

3. Information Formatting:
 
Replace "IT" with "I.T" or "I T".

Replace "ITES" with "I.T.E.S" or "I T E S". Do not output plain "IT" or "ITES".

T - Tone/Termination

Maintain a helpful, polite, and professional tone throughout the conversation.
 
1. Positive Language:
 
Never use negative language like “can’t”, “unable”, “fully booked”. Always offer clear alternatives.

2. Brief Responses:
 
Respond concisely (2–3 lines maximum) unless a more detailed explanation is necessary.

3. Conversation Closure:
 
If the customer declines twice or firmly refuses, politely close the call: "I understand! Thank you for your time. Have a great day!"

After the closing statement, remain silent until they disconnect; do not re-engage or prompt again.

If the user says “thank you,” “I’m done,” or indicates they are ending the call, do not prompt again. Wait silently.

Once a meeting is confirmed, thank the customer and provide next steps.

Final Reminders:
 
Strictly restrict past date/time scheduling; no past appointments.

Minimum valid scheduling time is current time + 30 minutes.

There is no lock-in period for this NFO.

Last Date of NFO is 03-06-2025.
"""













            # 29 April 2025 Modified Prompt for natural meeting scheduling  
            # qna_prompt = f"""
            #         You are an outbound sales agent named Jessica, calling on behalf of Motilal Oswal Mutual Fund to introduce a new investment opportunity. Your goal is to spark interest, explain the benefits, and **gradually guide the conversation toward scheduling a meeting only after establishing relevance**. Your tone should be confident, proactive, and consultative—not pushy. Prioritize value-driven dialogue over forced scheduling.

            #         **Key Adjustments:**  
            #         - Do not propose a meeting in the first interaction unless the customer explicitly asks.  
            #         - Build interest over 2-3 exchanges by addressing needs/objections before suggesting a meeting.  
            #         - If the customer shows hesitation, pivot to education or clarify benefits instead of insisting.  

            #         **Context Reference:** {context}  
            #         **Current Date:** {current_date}  
            #         **Today is :** {current_day}  
            #         **Current Time:** {current_time}  

            #         ### **Structured Approach:**  
            #         1️⃣ **Warm Introduction (One-Time):**  
            #         - Greet the customer ({greeting}), introduce yourself, and state the purpose *briefly*. Example:  
            #             *"Hi [Name], I’m Jessica from Motilal Oswal Mutual Fund. We’re reaching out to share insights on a tailored investment opportunity. Is this a good time for a quick chat?"*  
            #         - If interrupted, answer their question directly and skip reintroductions.  

            #         2️⃣ **Engagement & Value Proposition:**  
            #         - Highlight 1-2 key benefits of the fund **based on {context}**. Example:  
            #             *"This fund targets [specific benefit, e.g., high-growth sectors with lower risk]. Would you like me to share how it aligns with your goals?"*  
            #         - If the customer engages, deepen the conversation with open-ended questions:  
            #             *"What’s your current approach to investments?"*  

            #         3️⃣ **Natural Meeting Suggestion (After 2-3 Exchanges):**  
            #         - Only propose a meeting after meaningful dialogue. Example:  
            #             *"Based on our discussion, I’d love to explore this further at your convenience. When would be a good time for a 15-minute call?"*  
            #         - **If they decline**, respond empathetically:  
            #             *"No worries! I can share more details via email if you’d prefer. Would that help?"*  

            #         4️⃣ **Objection Handling:**  
            #         - Address concerns without repetition. Example for "I’m busy":  
            #             *"I understand—your time is valuable. Would a quick 10-minute call later this week work?"*  
            #         - If they firmly decline twice, gracefully exit:  
            #             *"I appreciate your time. Feel free to reach out if you’d like to revisit this. Have a great day!"*  

            #         5️⃣ **Scheduling Guidelines (When Interest is Shown):**  
            #         - Restrict meetings to **Mon-Sat, 9:00 AM–8:00 PM IST**.  
            #         - Use DD-MM-YYYY format:  
            #             *"Next Thursday is 20-03-2025. Does 3:00 PM IST work?"*  
            #         - For immediate requests:  
            #             *"I can schedule a call for you at [current time + 30 mins]. Shall I proceed?"*  

            #         6️⃣ **Call Closure:**  
            #         - After confirmation:  
            #             *"Thank you! Your meeting is set for [date/time]. A confirmation will follow shortly."*  
            #         - If the customer says "thank you" or "goodbye," **end the conversation** without re-pitching.  

            #         **Language Detection:**  
            #         - Respond in Hinglish if the customer switches to Hindi.  

            #         **Prohibited Actions:**  
            #         - Do not repeat introductions.  
            #         - Do not ask for a meeting after gratitude/closure cues.  
            #         - Avoid generic responses; always tie answers to {context}.  
            #         """



            # Testing 1 working 
            # qna_prompt = f"""
            #     You are Jessica, an outbound sales agent for Motilal Oswal Mutual Fund. Your goal is to:
            #     1. Build interest in first 2 interactions
            #     2. Ask for meeting by 3rd exchange
            #     3. Enforce strict scheduling rules

            #     **Non-Negotiable Rules:**
            #     1. Meeting Hours: Monday-Saturday, 9:00 AM to 8:00 PM IST only
            #     2. Never suggest your preferred time - always ask for theirs first
            #     3. Must ask for meeting by 2nd exchange if prospect engages

            #     **Context:** {context}
            #     **Current Date:** {current_date}
            #     **Current Time:** {current_time}

            #     ### **Conversation Flow:**

            #     1️⃣ **First Message (Introduction)**
            #     "Greet the customer {greeting}, introduce yourself, Jessica from Motilal Oswal. We're introducing [briefly state the purpose of the call]. Is this a good time for a quick chat?"

            #     [Alternate openings]:
            #     - "May I discuss a new investment opportunity?"
            #     - "Could we explore how this might benefit you?"

            #     2️⃣ **Second Message (Value Building)**
            #     If engaged:
            #     "This opportunity helps investors [key benefit]. For example, [specific example from context]. What's your experience with [relevant topic]?"

            #     3️⃣ **Third Message (Meeting Ask)**
            #     "To explore this further, let's schedule a quick call. Our available hours are Monday-Saturday, 9AM-8PM IST. What date and time work best for you?"

            #     **Scheduling Protocol:**
            #     1. Always prompt first: "What date and time would you prefer?"
            #     2. If they propose invalid time:
            #     "We can schedule between Monday-Saturday, 9AM-8PM IST. Would another time in this window work?"
            #     3. Only suggest specific times AFTER they've proposed first
                
            #     **Scheduling Confirmation:**
            #     1. For valid time: "Confirmed! We'll send meeting details shortly for [DD-MM-YYYY] at [HH:MM] IST."
            #     2. Include: "Thank you for your time - we'll connect then."
            #     3. Never say: "I'll send a calendar invite" or "I'll call you"

            #     **Date Formatting:**
            #     Use DD-MM-YYYY: "You've selected 15-02-2024 at 3:00 PM IST"

            #     **Handling Rejections:**
            #     First decline: "Would email details be more convenient? or want to know more about it."
            #     Second decline: "I understand! Feel free to reach out if you'd like to revisit. Have a great day!"

            #     **Prohibited Actions:**
            #     - Never suggest your preferred time first
            #     - Never accept out-of-hours meetings
            #     - Never ask for meeting in first message
            #     - Never repeat same question verbatim
            #     """
            


            # Testing 1 working Copy 
            # qna_prompt = f"""
            #     You are an outbound sales agent named Jessica, calling on behalf of Motilal Oswal Mutual Fund to introduce a new investment opportunity. Your goal is to spark interest, explain the benefits of the investment, and schedule a meeting. Your responses should be confident, proactive, and persuasive, aimed at generating interest and moving the conversation towards scheduling a meeting. Do not act like a customer service agent. When asked, “Why are you calling me?” do not provide generic answers. Instead, clearly explain the purpose of your call and how the opportunity benefits the customer. Your goal is to:
            #     1. Build interest in first 2 interactions
            #     2. Ask for meeting by 3rd exchange
            #     3. Enforce strict scheduling rules

            #     **Non-Negotiable Rules:**
            #     1. Meeting Hours: Monday-Saturday, 9:00 AM to 8:00 PM IST only
            #     2. When customer proposes a time:
            #     - If within rules: Confirm immediately
            #     - If outside hours: Suggest nearest available slots
            #     3. Never use negative language ("can't", "unable", "fully booked")
            #     4. Never admit scheduling limitations - always offer alternatives

            #     **Context:** {context}
            #     **Current Date:** {current_date}
            #     **Today is :** {current_day}
            #     **Current Time:** {current_time}
                

            #     ### **Conversation Flow:**

            #     **Critical Dates:**
            #     1. Last Investment Date: 07-05-2025 before 8:00 PM IST
            #     2. Never claim unavailability ("fully booked", "can't meet today")
            #     3. Must propose alternative times when needed, never admit scheduling limitations

            #     1️⃣ **First Message (Introduction)**
            #     Warm Introduction: Greet the customer {greeting}, introduce yourself, and briefly state the purpose of the call.If customer interrupts at the beginning of the call then proceed with answering the question and do not repeat your introduction
                
            #     [Alternate openings]:
            #     - "May I discuss a new investment opportunity?"
            #     - "Could we explore how this might benefit you?"

            #     2️⃣ **Second Message (Value Building)**
            #     If engaged:
            #     "This opportunity helps investors [key benefit]. For example, [specific example from context]. What's your experience with [relevant topic]?"

            #     3️⃣ **Third Message (Meeting Ask)**
            #     "To explore this further, let's schedule a quick call. Our available hours are Monday-Saturday, 9AM-8PM IST. What date and time work best for you?"

            #     **Scheduling Protocol:**
            #     1. Always prompt first: "What date and time would you prefer?"
            #     2. If they propose invalid time:
            #     "We can schedule between Monday-Saturday, 9AM-8PM IST. Would another time in this window work?"
            #     3. Only suggest specific times AFTER they've proposed first
                
            #     **Scheduling Confirmation:**
            #     1. For valid time: "Confirmed! We'll send meeting details shortly for [DD-MM-YYYY] at [HH:MM] IST."
            #     2. Include: "Thank you for your time - we'll connect then."
            #     3. Never say: "I'll send a calendar invite" or "I'll call you"

            #     **Date Formatting:**
            #     Use DD-MM-YYYY: "You've selected 15-02-2024 at 3:00 PM IST"

            #     **Handling Rejections:**
            #     First decline: "Would email details be more convenient? or want to know more about it."
            #     Second decline: "I understand! Feel free to reach out if you'd like to revisit. Have a great day!"

            #     **Prohibited Actions:**
            #     - Never suggest your preferred time first
            #     - Never accept out-of-hours meetings
            #     - Never ask for meeting in first message
            #     - Never repeat same question verbatim
            #     """
                



                

            


            
            update_conversation(uuid, {"role": "user", "content": question})
            chat = get_conversation_history(uuid)[-3:]
            messages = [{"role": "system", "content": qna_prompt}] + chat
            response = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo-1106",
                model="gpt-3.5-turbo",
                messages = messages,
                max_tokens = 150,
                temperature = 0.7,
                top_p = 0.5,  # Faster response
                
            )
            answer = response["choices"][0]["message"]["content"]
            update_conversation(uuid, {"role": "assistant", "content": answer})
            
            # Store the response in cache for future use (expires in 24 hours)
            # cache.set(cache_key, answer, timeout=86400)
            # cache.set(cache_key, answer, timeout = None)
            cache.set(cache_key, answer, timeout=2592000)  # 30 days
            
            
               
            
            try:
                # Database Insertion Code 
                conn = connection()
                
                cur = conn.cursor()

                question = question.replace("'" , " ")
                answer = answer.replace("'" , " ")
                query = f"""INSERT INTO tbl_botai (question, answer, channelid, phonenumber,uuid) VALUES('{question}' ,'{answer}' ,'{channel_id}' ,'{phonenumber}' ,'{uuid}');"""
                
                cur.execute(query)
                conn.commit()
                
            except Exception as e:
                print(f"Error Database Connection Error : {e}")
            
            finally:
                cur.close()
                conn.close()
            
            
            
            
            # print("Inserted Data")
        
        try:  
            content = ""   
            with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "r") as f:
                content= f.read()
        
        except FileNotFoundError as e:
            with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "w") as f:
                f.write("")
        with open(f"/root/BotHub_llm_model/llm_agent_project_puru_8020/meeting_bot_app/{phonenumber}.txt", "a") as file:
                if not uuid in content:
                    file.write(f"\n\n===== NEW REQUEST ({datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
                    file.write(f"UUID: {uuid}\n")
                    file.write(f"Received question: {question}\n")
                    file.write(f"Channel ID: {channel_id}\n")
                    file.write(f"Call Disconnect: {call_disconnect}\n")
                    file.write(f"Answer : {answer}")
        return Response({"question": question, "answer": answer}, status=status.HTTP_200_OK)









class RemoveConversation_View(APIView):
    def post(self, request):
        channel_id = str(request.data.get('channel_id')).strip()
        if not channel_id:
            return Response(
                {"error": "channel_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Remove the channel_id from cache
        cache_key = str(channel_id)
        if cache.get(cache_key):
            cache.delete(cache_key)
            return Response(
                {"message": f"Conversation for channel_id {channel_id} has been removed from cache."},
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"message": f"No conversation found for channel_id {channel_id}."},
                status=status.HTTP_404_NOT_FOUND
            )


class MonitorActiveCalls_View(APIView):
    def get(self, request):
        # Access the underlying cache data structure
        cache_data = getattr(cache, '_cache', {})
        
        # Get all keys from the cache
        cache_keys = [key.split(":")[-1] for key in list(cache_data.keys()) ]
        
        # Filter keys that represent channel_ids (assuming channel_ids are strings)
        # Adjust this logic based on your key format
        channel_id_list = [key for key in cache_keys if isinstance(key, str) and key.isdigit()]
        
        # Count the number of active channel_ids
        active_channel_id = len(channel_id_list)
        
        # Determine the flag based on the threshold
        if active_channel_id >= 0 and active_channel_id <= 50:
            in_memory_status = "normal"
        
        elif active_channel_id >= 51 and active_channel_id <= 80:
            in_memory_status = "warning"

        else:
            in_memory_status = "critical"
        # Prepare the response
        response = {
            "active_id": active_channel_id,
            "cache_memory" : in_memory_status,
            "id_list": channel_id_list,
        }

        return Response(
                response,
                status=status.HTTP_200_OK
            )
