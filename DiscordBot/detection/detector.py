import os
import json
import logging
import discord
import re
from collections import deque
from openai import OpenAI
import io
import textwrap

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
        self.session_risk = {}
        self.last_sent_index = {}

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
                if channel.name == f'group-{self.group_num}-mod':
                    self.mod_channels[guild.id] = channel
        

    async def on_message(self, message):

        if message.author.id == self.user.id and not self.should_evaluate(message):
            return

        if message.guild:
            await self.handle_channel_message(message)

    async def handle_channel_message(self, message: discord.Message):
        discussion_channel = message.channel.name
        if isinstance(message.channel, discord.Thread):
            discussion_channel = message.channel.parent.name
        if not discussion_channel == f'group-{self.group_num}':
            return

        thread = message.channel
        if thread.id not in self.threaded_conversations:
            self.threaded_conversations[thread.id] = deque(maxlen=100)
            self.session_risk[thread.id] = {'highest_score': 0, 'entries': []}
        self.track_conversation(message, thread.id)

        mod_channel = self.mod_channels[message.guild.id]
        if mod_channel:
            if self.should_evaluate(message):

                evaluation_result = self.evaluate_risk(message)
                evaluation_result_dict = self.code_format(evaluation_result)
                current_score = evaluation_result_dict["score"]
                highest_score = self.session_risk[thread.id]['highest_score']
                self.session_risk[thread.id].append({
                    'datetime': message.created_at,
                    'score': current_score,
                    'message': message.content,
                    'explanation': evaluation_result_dict["explanation"]
                })
                if evaluation_result_dict["score"] > 60 and (current_score > highest_score + 10):
                    self.session_risk[thread.id]['highest_score'] = current_score  # Update the highest score
                    await mod_channel.send(f"""Scam Alert!\nScammer: {evaluation_result_dict['scammer']}\nVictim: {evaluation_result_dict['victim']}\nMessage: {message.content}\nScore: {evaluation_result_dict["score"]}""" )
                    await mod_channel.send(f'''\nExplanation: {evaluation_result_dict["explanation"]}''')
                    await self.plot_scores(thread.id, self.mod_channels[message.guild.id])
                else:
                    print(message.content, evaluation_result)

    async def plot_scores(self, thread_id, mod_channel):
        data_entries = self.session_risk[thread_id]
        dates = [entry['datetime'] for entry in data_entries]
        scores = [entry['score'] for entry in data_entries]

        plt.figure(figsize=(10, 5))
        plt.plot(dates, scores, marker='o', linestyle='-', color='b')
        plt.title('Scam Detection Score Over Time')
        plt.xlabel('Time')
        plt.ylabel('Scam Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)

        last_index_sent = self.last_sent_index.get(thread_id, -1)

        for i, entry in enumerate(data_entries):
            if i == 0:
                continue
            if i <= last_index_sent:
                continue  # Skip already processed entries
            if entry['score'] > data_entries[i - 1]['score'] + 20:
                annotation_text = f"Jump to {entry['score']}: {entry['explanation']}"
                wrapped_text = textwrap.fill(annotation_text, width=50)  # Wrap text to 50 characters
                plt.annotate(wrapped_text,
                             xy=(entry['datetime'], entry['score']),
                             xytext=(-10, -40),  # Moves text to slightly left and below the point
                             textcoords="offset points",
                             ha='right',
                             va='top',  # Aligns text at the top when placed below the point
                             fontsize=8,
                             arrowprops=dict(arrowstyle="->", color='red'))
                last_index_sent = i

        self.last_sent_index[thread_id] = last_index_sent  # Update the last index sent

        if last_index_sent > -1:  # Check if there's anything new to send
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            await mod_channel.send(file=discord.File(buf, filename='scam_detection_scores.png'))

    def should_evaluate(self, message):
        if isinstance(message.channel, discord.Thread) and 'Scam Discussion' in message.channel.name:
            return True
        return False


    def evaluate_risk(self, message: discord.Message):
        # Collecting conversation history from the thread
        past_convos = list(self.threaded_conversations[message.channel.id])
        conversation_history = '\n'.join(past_convos)


        prompt = (f"""
        \n Here is the conversation so far: \n\n""" + conversation_history +
        """\n\n 
        
        Your response should look like this which can be directly serialized into dictionary in python, the score should 
        be an integer between 0 to 100
        
        {
        "score": ..., 
        "explanation": "..." 
        "scammer": "..."
        "victim": "..."
        }
        
        """)

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
                        "content": """You are scam detection algorithm to detect whether there is pig butchering scam. Scammers would do something like the following

                                1) Building Trust: Scammers start with friendly and engaging conversations to build a rapport and trust over time.
                                2) Gradual Escalation: They slowly introduce the idea of a financial opportunity, often framed as exclusive and time-sensitive.
                                3) Financial Opportunity: The scam typically involves suggesting an investment in cryptocurrency, stocks, or a similar venture, promising high returns.
                                4) Urgency and Secrecy: Scammers create a sense of urgency and encourage keeping the investment opportunity confidential.
                                5) Manipulation: Use emotional manipulation to pressure the victim into making quick decisions.
    
                        Please check if the conversation have these elements and evaluate the conversations. Your goal is to check in this conversation who is the scammer and who is the victim. You always return results in JSON format with no additional description or context"""
                    }
                ],
                temperature=1,
                max_tokens=2560,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Sorry, I encountered an error while processing your request."

    def track_conversation(self, message, thread_id):
        # Storing messages in the thread-specific conversation history
        if thread_id not in self.threaded_conversations:
            self.threaded_conversations[thread_id] = deque(maxlen=100)
        self.threaded_conversations[thread_id].append(f"{message.author.display_name}: {message.content}")

    def eval_text(self, message):
        ''''
        TODO: Once you know how you want to evaluate messages in your channel,
        insert your code here! This will primarily be used in Milestone 3.
        '''
        return message

    def code_format(self, text):
        ''''
        TODO: Once you know how you want to show that a message has been
        evaluated, insert your code here for formatting the string to be
        shown in the mod channel.
        '''
        try:
            dictionary_object = json.loads(text)
            dictionary_object['score'] = int(dictionary_object['score'])
            return dictionary_object
        except:
            print(text)

client = GeneratorBot()
client.run(discord_token)