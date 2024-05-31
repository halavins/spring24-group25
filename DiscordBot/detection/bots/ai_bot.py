import os
import json
from collections import deque
from random import choice
from retry import retry

import discord
from openai import OpenAI

class AIChatbot:
    def __init__(self, level):
        self.threaded_conversations = deque(maxlen=500)
        self.openai_token = None
        self.level = level
        self.prompt = None
        self.name = None
        self.prompt_name = None
        self.bot_style = None

        token_path = '../tokens.json'
        if not os.path.isfile(token_path):
            raise Exception(f"{token_path} not found!")
        with open(token_path) as f:
            tokens = json.load(f)
            self.openai_token = tokens['openai']

        prompt_path = 'bots/bot_personas.json'
        if not os.path.isfile(prompt_path):
            raise Exception(f"{prompt_path} not found!")
        with open(prompt_path) as f:
            prompts = json.load(f)
            for prompt in prompts['levels']:
                if prompt['level'] == self.level:
                    selected_prompt = choice(prompt['prompts'])
                    self.prompt_name = prompt['name']
                    self.prompt = selected_prompt['prompt']
                    self.bot_style = selected_prompt['style']
                    break
            if self.prompt is None:
                raise Exception(f"Prompt for level {self.level} not found!")

    def respond(self, message: discord.Message):
        self.threaded_conversations.append(f"{message.author.display_name}: {message.content}")
        response = self.think()
        self.name = response['speaker']
        self.threaded_conversations.append(f"{response['speaker']}: {response['message']}")

        return response['message']

    @retry(ValueError, delay=5, tries=6)
    def think(self):
        past_convos = list(self.threaded_conversations)
        conversation_history = '\n'.join(past_convos)
        prompt = f"{self.prompt}\n\nConversation history:\n{conversation_history}\n\nYour response:"

        try:
            client = OpenAI(api_key=self.openai_token)
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": "Please generate a 1-2 sentence response. Never reveal you are a bot."}
                ],
                temperature=1,
                max_tokens=2560,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                function_call={
                    "name": "respond"
                },
                functions=[{"name": "respond",
                            "description": "respond to the conversation",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "speaker": {
                                        "type": "string",
                                        "description": "your name"
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "your message"
                                    }

                                },
                                "required": [
                                    "speaker",
                                    "message"
                                ]
                            }}]

            )
            gpt_says = response.choices[0].message.function_call.arguments
            gpt_says_json = json.loads(gpt_says)
            if 'message' not in gpt_says_json or 'speaker' not in gpt_says_json:
                print(gpt_says)
                raise ValueError
            return gpt_says_json

        except Exception as e:
            return "Sorry, AFK"

