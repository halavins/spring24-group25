import json
import logging
import os
import re

import discord

from bots.ai_bot import AIChatbot

# Set up logging to the console
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# There should be a file called 'tokens.json' inside the same folder as this file
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
    tokens = json.load(f)
    discord_token = tokens['discord']
    openai_token = tokens['openai']


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
        self.user_to_bot = {}

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
            if 'scam me' in message.content.lower() or 'talk to me' in message.content.lower():
                match = re.search('scam me (\d+)', message.content.lower())
                if match:
                    sus_level = int(match.group(1))
                elif message.content.lower() == 'talk to me':
                    sus_level = 0
                elif message.content.lower() == 'scam me':
                    sus_level = 11
                else:
                    sus_level = 3

                bot = AIChatbot(level=sus_level)
                thread = await message.create_thread(name=f"{message.author.display_name}- Scam Level ({bot.level}) / {bot.prompt_name} / {bot.bot_style}")

                self.user_to_bot[thread.id] = bot
            else:
                return
        else:
            thread = message.channel

        response = self.user_to_bot[thread.id].respond(message)
        if response:
            await thread.send(response)


client = GeneratorBot()
client.run(discord_token)