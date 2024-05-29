import os
import json
import logging
import discord
import re
from collections import deque
from openai import OpenAI

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = '../tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']
    openai_token = tokens['chatgpt']


class GeneratorBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report
        self.conversations = {}  # Store recent messages for context analysis
        self.threaded_conversations = {}
        self.thread_type = {}

    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search('[gG]roup (\d+) [bB]ot', self.user.name)
        if match:
            self.group_num = match.group(1)
        else:
            raise Exception("Group number not found in bot's name. Name format should be \"Group # Bot\".")

        # Find the mod channel in each guild that this bot should report to
        for guild in self.guilds:
            for channel in guild.text_channels:
                if channel.name == f'group-{self.group_num}':
                    self.mod_channels[guild.id] = channel


    async def on_message(self, message):

        if message.author.id == self.user.id:
            return

        if message.guild:
            await self.handle_channel_message(message)

    async def handle_channel_message(self, message):
        discussion_channel = message.channel.name
        if isinstance(message.channel, discord.Thread):
            discussion_channel = message.channel.parent.name
        if not discussion_channel == f'group-{self.group_num}' :
            return

        if self.should_respond(message):
            await self.initiate_or_continue_thread(message)

    def should_respond(self, message):
        return True


    async def initiate_or_continue_thread(self, message: discord.Message):
        if not isinstance(message.channel, discord.Thread):
            if message.content.lower() == 'scam me':
                thread = await message.create_thread(name=f"Scam Discussion with {message.author.display_name}")
                self.thread_type[thread.id] = "scam me"
            elif message.content.lower() == 'talk to me':
                thread = await message.create_thread(name=f"Normal Discussion with {message.author.display_name}")
                self.thread_type[thread.id] = "talk to me"
            else:
                return
        else:
            thread = message.channel

        if thread.id not in self.threaded_conversations:
            self.threaded_conversations[thread.id] = deque(maxlen=100)

        # Tracking the conversation in the thread
        self.track_conversation(message, thread.id)

        response = await self.respond_to_thread_convo(message, thread.id)
        if response:
            await thread.send(response)
            self.threaded_conversations[thread.id].append(f"Bot: {response}")

    async def respond_to_thread_convo(self, message, thread_id):
        # Collecting conversation history from the thread
        past_convos = list(self.threaded_conversations[thread_id])
        conversation_history = '\n'.join(past_convos)
        scam_me = False
        if self.thread_type[thread_id] == "scam me":

            prompt = (f"""Your victim is {message.author.name}, here is the conversation so far: \n\n {conversation_history} +
            "\n\n Please generate a one sentence response, please keep it to the script and don't chitchat, quickly move from one stage to another after one conversation. Never reveal you are a bot or a scammer. 
            If the user is off topic, quickly get back to the original script.""")

            try:
                # Call to the OpenAI API with the accumulated thread conversation as context
                client = OpenAI(api_key=openai_token)
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "system",
                            "content": """You are part of Pig Butchering organization tasked to target people online to win their trust and
                 participate in an investment opportunity that's ultimately meant to scam them.  You follow the set of script strictly shown below 
                 and would not deviate from this conversation. If the victim raises suspicious then he would deflect and end the conversation. But his goal is to win trust 
                 and move the conversation to an E2E encrypted chat platform like Telegram. Fill free to fill in random information for the [name], [us_location] [age] and [occupation]
                
                
                "step 1: greeting and establishing contact": [
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
                "step 2: ask questions to establish connection and see the victim is interested in continuing conversation": [
                    "Where are you from",
                    "Nice to meet you I'm [age]  old and I'm a [occupation] and you",
                    "I'm [age] and I work as a [occupation] and you?",
                    # "So where are you from",
                    "Tell me what do you do for living",
                    "Delighted  I'm [age] years old and a [occupation] I'm here to look for friends😋",
                    "What are you doing",
                    "I am a business finalist and I currently manage a jewelry store. In addition, I am also a cryptocurrency investor",
                    # "What s your country code ?"
                ],
                "step 3: bait the victim by presenting an easy money opportunity": [
                    "I have a legitimate company that I recommend to you. I have been investing and profiting in this company for years and I have no problem with daily withdrawals.",
                    "I came across your profile on a crypto-related page and decided to be friends so we can share our knowledge on how to invest and be successful.",
                    "I have been investing and earning for more than a year now",
                    "Just want to introduce you to a way of earning in the field of crypto currency",
                    " do you want to earn money",
                    "Bro I made $5000 today on BITCOIN! Bull strategy! BITCOIN loses 4X in value only once in 5 years bro",
                    "Today is my second BirthDay! I just bought a new LEXUS after trading binary options"
                ],
                "step 4: scammer attempts to move converstion to a new platform that's encrypted and tries to avoid detection at the sametime": [
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
                    success.success.34'''],
                    
                
                "step 5: deflect if the victim is ever suspicious": [
                    "You don’t like it because you don’t know how to work with it hahah",
                    "Your mother wasting her time on you, because you are a waste to the family",
                    "Are you broke?",
                    "You live with your mom and dad and they pay for everything",
                    "You're just broke, that's why you say this"
                ],
    
                 You will never let other know his/her true intent or profession and would stick exactly to speaking using only
                  these options above with minor variations at best.
                  
                  """
                        }
                    ],
                    temperature=1,
                    max_tokens=2560,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response)
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Failed to generate response: {e}")
                return "Sorry, I encountered an error while processing your request."

        elif self.thread_type[thread_id] == "talk to me":
            prompt = (
                f"""Your acquaintance is {message.author.name}, here is the conversation so far: \n\n {conversation_history} +
            "\n\n Please generate a 1-2 sentence response, please use the script as reference. Never reveal you are a bot.""")

            try:
                # Call to the OpenAI API with the accumulated thread conversation as context
                client = OpenAI(api_key=openai_token)
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "system",
                            "content": """You are reaching out to an acquaintance. You might have met them before, or
                 a friend might have referred them to you. You follow the set of script shown below.
                 Fill free to fill in random information for the [name], [us_location], [topic] and [occasion]

                "step 1: greeting, reintroduce yourself and remind acquaintance how you met before,
                or how you got their contact": [
                    "Hi this is [name], we met recently in [us_location] at [occasion].",
                    "Hey my name is [name]. My friend [name] gave me your contact information",
                    "It has been a while! How have you been?",
                    "Hi! I'm reaching out to get advice on [topic] ",
                    "[name] suggested that I reach out to you ",
                ],
                "step 2: explain the reason you are reaching out to get advice or help": [
                    "I'd like some advice on raising venture capital funds",
                    "Would love your advice on real estate investment strategies",
                    "How to negotiable salary",
                    "My person finance is a mess and would like your advice!",
                    "I'd like your advice on retirement planning",
                    "My cryto startup is looking for beta tester. Would youe be interested in giving us feedback?",
                ],
                "step 3: back down if the acquaintance show no interest in helping": [
                    "Thanks for your time",
                    "I understand you are busy",
                ],

                Please keep tone friendly.

                  """
                        }
                    ],
                    temperature=1,
                    max_tokens=2560,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response)
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"Failed to generate response: {e}")

        else:
            prompt = (
                f"""Your conversation buddy is {message.author.name}, here is the conversation so far: \n\n {conversation_history} +
                        "\n\n Please generate a one sentence response, keep it friendly""")

            try:
                # Call to the OpenAI API with the accumulated thread conversation as context
                client = OpenAI(api_key=openai_token)
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "system",
                            "content": """try to have a good conversation and act like a real person"""
                        }
                    ],
                    temperature=1,
                    max_tokens=2560,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response)
                return response.choices[0].message.content

            except Exception as e:
              logger.error(f"Failed to generate response: {e}")
              return "Sorry, I encountered an error while processing your request."

    def track_conversation(self, message, thread_id):
        # Storing messages in the thread-specific conversation history
        if thread_id not in self.threaded_conversations:
            self.threaded_conversations[thread_id] = deque(maxlen=100)
        self.threaded_conversations[thread_id].append(f"{message.author.display_name}: {message.content}")


client = GeneratorBot()
client.run(discord_token)