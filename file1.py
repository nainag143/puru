import openai
import time
import random


nlu_template = """
Agnostic NLU with Llama 3.1 Instruct: Intent and Entity Examples
This prompt is designed to extract intents that reflect the conversational action the user is taking, rather than the specific topic.

<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an advanced, domain-agnostic Natural Language Understanding (NLU) system for a conversational AI assistant.
Your sole task is to analyze user utterances, identify their primary **conversational intent**, and extract any relevant **universal entities**.
You must be precise and extract all explicit information.
Return your analysis as a JSON object.

<full NLU instructions go here, don't skip them>

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_query}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def analyze_nlu(user_query):
    # ✅ Now format it safely
    prompt = nlu_template.format(user_query=user_query)

    start_time = time.time()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for classifying customer call dispositions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.3,
        top_p=0.5
    )


    duration = time.time() - start_time

    try:
        result = response['choices'][0]['message']['content'].strip()
        return result, duration  # Return NLU result + time taken
    except Exception as e:
        return {"error": str(e)}, duration

    # try:
    #     result = response['choices'][0]['message']['content'].strip()
    #     return result  # Should be the pure JSON
    # except Exception as e:
    #     return {"error": str(e)}

# 👇 Test call
# question = "Cancel my meeting and reschedule in mUmbai at 5 pm, also inform my secretery"
# output = analyze_nlu(question)
# print("NLU Output:", output)



# ---------------------
# FillerManager Class
# ---------------------
class FillerManager:
    def __init__(self, threshold_seconds=2.0):
        self.threshold = threshold_seconds

        # Original intent-only fillers
        self.intent_specific_fillers = {
            "request_information": [
                "Let me take a moment to gather the information you requested.",
                "I'm retrieving those details now — thanks for your patience.",
                "Give me a second while I look into that for you.",
                "I’m checking the latest updates for you right now.",
                "Please hold on — I’m pulling up the relevant data."
            ],
            "request_action": [
                "Alright, I’m working on carrying out your request as we speak.",
                "Understood. I’ll start processing that for you immediately.",
                "Let me initiate the action you've asked for — it’ll just take a moment.",
                "I’ve got that. Beginning the task now, and I’ll confirm once done.",
                "Sure thing, I’m handling that request right away."
            ],
            "express_frustration": [
                "I completely understand how frustrating this must be — give me a moment to resolve it.",
                "I'm really sorry you're experiencing this. Let me find a solution as quickly as I can.",
                "Thanks for your patience — I know this is frustrating, and I'm here to help.",
                "That doesn’t sound ideal. I’m on it and will do my best to make things right.",
                "I hear you — let me investigate and try to fix this promptly."
            ],
            "express_confusion": [
                "Let me go back and double-check what was said earlier so I can clarify things.",
                "I understand you're confused. Give me a second to rephrase or explain better.",
                "Alright, I’ll take a closer look and make sure everything is clear.",
                "One moment — I’m reviewing the previous context to make sure we’re aligned.",
                "I see there might have been a misunderstanding. Let me sort that out for you."
            ],
            "default": [
                "Just a moment while I figure this out for you.",
                "Thanks for your patience — I’m processing that now.",
                "I’ve received your message. Give me a few seconds to respond appropriately.",
                "Working on that for you — it shouldn’t take long.",
                "Alright, I’m thinking through that and will get back to you shortly."
            ]
        }

        # ✅ New: Sentiment-specific fillers
        self.intent_sentiment_fillers = {
            "request_information": {
                "positive": [
                    "Excellent! Let me quickly find that for you.",
                    "Great! Just a moment while I pull up those details."
                ],
                "neutral": [
                    "Hmm, let me look that up for you.",
                    "One moment, I'm checking the details.",
                    "Please hold while I retrieve the information.",
                    "Just a second, finding that data now."
                ],
                "negative": [
                    "I understand you're looking for information. Please bear with me while I check.",
                    "I hear your query. Let me quickly verify that for you."
                ]
            },
            "request_action": {
                "positive": [
                    "Absolutely! Processing that for you now, just a moment.",
                    "Great! I'm on it. Let me see what I can do."
                ],
                "neutral": [
                    "Alright, processing that request.",
                    "Okay, I'm working on that.",
                    "Let me see what I can do for you."
                ],
                "negative": [
                    "I'm sorry to hear that. Let me quickly work on resolving this for you.",
                    "Understood. I'm taking action on that right now."
                ]
            },
            "express_frustration": {
                "positive": [],
                "neutral": [],
                "negative": [
                    "I truly apologize for the frustration you're experiencing. Please bear with me for a moment.",
                    "I completely understand your frustration. I'm looking into this right now for you.",
                    "I hear your concern. Let me do my best to resolve this quickly."
                ]
            },
            "express_confusion": {
                "positive": [],
                "neutral": [
                    "I see. Let me take a moment to clarify and re-evaluate that.",
                    "Okay, let me think about how to explain this better.",
                    "Just a moment while I gather my thoughts on that."
                ],
                "negative": [
                    "I apologize if that wasn't clear. Let me take a moment to formulate a better response.",
                    "I understand that was confusing. Let me quickly figure this out for you."
                ]
            },
            "default": {
                "positive": [
                    "Wonderful! Just a moment.",
                    "Okay, bear with me for a moment."
                ],
                "neutral": [
                    "Hmm, just a moment.",
                    "Please wait while I process that.",
                    "Right, I'm thinking about that."
                ],
                "negative": [
                    "I understand. Please give me a moment.",
                    "I hear you. Let me quickly check on that."
                ]
            }
        }

        # ✅ Updated long response filler
        self.general_long_response_filler = (
            "This might take a moment, but I'm diligently working on it for you. Please wait."
        )

    def get_filler_for_intent(self, conversational_intent):
        fillers = self.intent_specific_fillers.get(conversational_intent, self.intent_specific_fillers["default"])
        return random.choice(fillers)

    def get_sentiment_filler(self, conversational_intent, sentiment="neutral"):
        sentiment_dict = self.intent_sentiment_fillers.get(conversational_intent, self.intent_sentiment_fillers["default"])
        fillers = sentiment_dict.get(sentiment, self.intent_sentiment_fillers["default"]["neutral"])
        if fillers:
            return random.choice(fillers)
        return self.get_filler_for_intent(conversational_intent)

    def should_play_filler(self, processing_time):
        return processing_time > self.threshold

    def get_long_response_filler(self):
        return self.general_long_response_filler

# ---------------------
# Putting It All Together
# ---------------------
if __name__ == "__main__":
    question = "This is really frustrating. Just reset the network for me — nothing else has worked!"



    nlu_output_str, processing_time = analyze_nlu(question)
    print("\n🔍 NLU Output:\n", nlu_output_str)

    # Convert the string result to a dict (safely)
    import json
    try:
        nlu_data = json.loads(nlu_output_str)
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        nlu_data = {"conversational_intent": "default"}

    intent = nlu_data.get("conversational_intent", "default")

    filler_mgr = FillerManager(threshold_seconds=1.5)

    if filler_mgr.should_play_filler(processing_time):
        print("\n💬 Filler (due to processing time):")
        print(filler_mgr.get_long_response_filler())
    else:
        print("\n💬 Intent-based Filler:")
        print(filler_mgr.get_filler_for_intent(intent))
