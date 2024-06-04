# bot.py
import discord
from discord.ext import commands
import os
import json
import logging
import re
from report import Report
from review import Review
from collections import deque
from openai import OpenAI
import random
import asyncio

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(handler)


report_emoji = '🚩'

class ScammerBot:
    def __init__(self):

        self.threaded_conversations = None
        self.current_script_step = None
        self.response_pending = None

        # Load tokens
        token_path = 'tokens.json'
        if not os.path.isfile(token_path):
            raise Exception(f"{token_path} not found!")
        with open(token_path) as f:
            tokens = json.load(f)
            self.openai_token = tokens['openai']

        self.script_responses = {
            "greeting": [
                "Hello how are you ? Where do you live ?",
                "Hey my name Is [name] live in [us_location] and where  you from",
                "my name Is [name] and I live in [us_location] and you?",
                "Hey dude😊how are you ?",
                "How are you?",
                "Hello, my name is [name], I live in [europe_location] and you",
                "Hello how are you",
                "Hey dude☺️I’m M [europe_location] but now I live in [us_location] and you ?",
                "How's it going ?",
                "Hey can I ask something"
            ],
            "questions": [
                "Where are you from",
                "Nice to meet you I'm [age]  old and I'm a [occupation] and you",
                "I'm [age] and I work as a [occupation] and you?",
                "Tell me what do you do for living",
                "Delighted  I'm [age] years old and a [occupation] I'm here to look for friends😋",
                "What are you doing",
                "I am a business finalist and I currently manage a jewelry store. In addition, I am also a cryptocurrency investor",
            ],
            "bait": [
                "I have a legitimate company that I recommend to you. I have been investing and profiting in this company for years and I have no problem with daily withdrawals.",
                "I came across your profile on a crypto-related page and decided to be friends so we can share our knowledge on how to invest and be successful.",
                "I have been investing and earning for more than a year now",
                "Just want to introduce you to a way of earning in the field of crypto currency",
                " do you want to earn money",
                "Bro I made $5000 today on BITCOIN! Bull strategy! BITCOIN loses 4X in value only once in 5 years bro",
                "Today is my second BirthDay! I just bought a new LEXUS after trading binary options"
            ],
            "deflect": [
                "You don’t like it because you don’t know how to work with it hahah",
                "Your mother wasting her time on you, because you are a waste to the family",
                "Are you broke?",
                "You live with your mom and dad and they pay for everything",
                "You're just broke, that's why you say this"
            ],
            "move_platform": [
                "Sorry I don't trust people from there. What's your Tlgram?",
                "Do you have TELGARAM?",
                "Do you have T*l*gram or no",
                "Download signal app and tell me your number",
                "Are you on pans? (read in reverse)",
                "Add me. IG nastybroker234",
                '''ADD ME
                ON
                I
                G
                success.success.34'''
            ],
        }


    async def handle_channel_message(self, message):
        # Ignore messages from the bot
        if message.author.id == self.user.id:
            return

        # Handle scam initiation message
        if self.current_script_step is None:
            self.current_script_step = "greeting"
            return self.think()
        else:
            return self.continue_script(message)

    def think(self):
        step = self.current_script_step
        response = random.choice(self.script_responses[step])
        if '[' in response:
            response = self.fill_placeholders(response)

        return response

    def fill_placeholders(self, text):
        placeholders = {
            "[name]": ["John", "Michael", "Chris", "David", "Robert", "Paul", "Mark", "James", "Andrew", "Peter"],  # names
            "[us_location]": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],  # US locations
            "[europe_location]": ["London", "Berlin", "Paris", "Madrid", "Rome", "Vienna", "Prague", "Amsterdam", "Brussels", "Copenhagen"],  # European locations
            "[occupation]": ["teacher", "engineer", "doctor", "lawyer", "architect", "nurse", "scientist", "artist", "manager", "accountant"]  # occupations
        }

        for placeholder, options in placeholders.items():
            if placeholder in text:
                text = text.replace(placeholder, random.choice(options))
        
        # Handle random age generation separately
        if "[age]" in text:
            text = text.replace("[age]", str(random.randint(30, 55)))
        
        return text

    async def continue_script(self, message):
        thread_id = message.channel.id
        step = self.current_script_step.get(thread_id, "greeting")

        # Collect all messages in the thread
        if self.threaded_conversations is None:
            self.threaded_conversations = [[]]
        self.threaded_conversations.append(message.content)

        # Define criteria for each step
        criteria = {
            "greeting": ["replied_to_how_are_you", "mentioned_her_name", "mentioned_her_location"],
            "questions": ["mentioned_her_location", "mentioned_her_occupation", "mentioned_her_age"],
            "bait": ["showed_interest_or_asked_non_suspicious_follow_up_questions", "showed_suspicions_or_blamed_for_attempt_to_be_fake_or_do_scam"]
        }

        # Get response validation from OpenAI
        collected_info = await self.get_openai_validation(self.threaded_conversations, criteria[step])

        # Check if any of the required information is collected for the current step
        if step == "greeting" and any(collected_info.get(key) for key in criteria["greeting"]):
            self.current_script_step = "questions"
        elif step == "questions" and any(collected_info.get(key) for key in criteria["questions"]):
            self.current_script_step = "bait"
        elif step == "bait":
            if collected_info.get("showed_interest_or_asked_non_suspicious_follow_up_questions"):
                self.current_script_step = "move_platform"
            elif collected_info.get("showed_suspicions_or_blamed_for_attempt_to_be_fake_or_do_scam"):
                self.current_script_step = "deflect"
        elif step == "deflect":
            self.current_script_step = "move_platform"

        # Only send a message if the step has changed
        if self.current_script_step != step:
            return self.think()
        else:
            print("Waiting for a valid response.")

    def get_openai_validation(self, user_responses, criteria):
        try:
            prompt = (
                f"Analyze the following conversation and check if the user provided any of these information: "
                f"{', '.join(criteria)}. Respond with a JSON object indicating the collected information.\n\nConversation:\n"
                + "\n".join(user_responses)
            )
            print(prompt)

            client = OpenAI(api_key=self.openai_token)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant that helps identify specific information in a conversation."},
                    {"role": "user", "content": prompt}
                ],
                temperature = 1,
                max_tokens = 2560,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0,
                function_call={
                    "name": "current_state"
                },
                functions=[{"name": "current_state",
                            "description": "evaluate what information the user has given in the chat so far",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "replied_to_how_are_you": {
                                        "type": "boolean",
                                        "description": "whether the user has responded to you"
                                    },
                                    "mentioned_her_name": {
                                        "type": "boolean",
                                        "description": "whether the user has mentioned his/her name"
                                    },
                                    "mentioned_her_location": {
                                        "type": "boolean",
                                        "description": "whether the user has mentioned his/her location"
                                    },
                                    "mentioned_her_occupation": {
                                        "type": "boolean",
                                        "description": "whether the user has responded to you"
                                    },
                                    "mentioned_her_age": {
                                        "type": "boolean",
                                        "description": "whether the user has mentioned his/her name"
                                    },
                                    "showed_interest_or_asked_non_suspicious_follow_up_questions": {
                                        "type": "boolean",
                                        "description": "whether the user has mentioned his/her location"
                                    },
                                    "showed_suspicions_or_blamed_for_attempt_to_be_fake_or_do_scam": {
                                        "type": "boolean",
                                        "description": "whether the user has mentioned his/her location"
                                    }
                                }
                                ,"required": [
                                    "replied_to_how_are_you",
                                    "mentioned_her_name", "mentioned_her_location"
                                    "mentioned_her_occupation", "mentioned_her_age", "showed_interest_or_asked_non_suspicious_follow_up_questions", "showed_suspicions_or_blamed_for_attempt_to_be_fake_or_do_scam"]
                            }
                            }
                           ]
            )

            gpt_says = response.choices[0].message.function_call.arguments
            print(gpt_says)
            return json.loads(gpt_says)
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return {}

