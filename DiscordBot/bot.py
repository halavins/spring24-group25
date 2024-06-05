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

# Load tokens
token_path = 'tokens.json'
if not os.path.isfile(token_path):
    raise Exception(f"{token_path} not found!")
with open(token_path) as f:
    tokens = json.load(f)
    discord_token = tokens['discord']
    openai_token = tokens['openai']

report_emoji = '🚩'

class ModBot(discord.Client):
    def __init__(self): 
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.reactions = True
        intents.members = True
        super().__init__(command_prefix='.', intents=intents)
        self.group_num = None
        self.mod_channels = {} # Map from guild to the mod channel id for that guild
        self.reports = {} # Map from user IDs to the state of their report
        self.reviews = {} # Map from thread IDs to review
    async def on_ready(self):
        print(f'{self.user.name} has connected to Discord! It is these guilds:')
        for guild in self.guilds:
            print(f' - {guild.name}')
        print('Press Ctrl-C to quit.')

        # Parse the group number out of the bot's name
        match = re.search(r'[gG]roup (\d+) Pig Butchering [bB]ot', self.user.name)
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
        '''
        This function is called whenever a message is sent in a channel that the bot can see (including DMs). 
        Currently the bot is configured to only handle messages that are sent over DMs or in your group's "group-#" channel. 
        '''
        # Ignore messages from the bot 
        if message.author.id == self.user.id:
            return

        # Check if this message was sent in a server ("guild") or if it's a DM
        if message.guild:
            await self.handle_channel_message(message)
        else:
            await self.handle_dm(message)

    async def handle_dm(self, message):
        # Handle a help message
        if message.content == Report.HELP_KEYWORD:
            reply =  "Use the `report` command to begin the reporting process.\n"
            reply += "Use the `cancel` command to cancel the report process.\n"
            await message.channel.send(reply)
            return

        author_id = message.author.id
        responses = []

        # Only respond to messages if they're part of a reporting flow
        if author_id not in self.reports and not message.content.startswith(Report.START_KEYWORD):
            return

        # If we don't currently have an active report for this user, add one
        if author_id not in self.reports:
            self.reports[author_id] = Report(self)

        # Let the report class handle this message; forward all the messages it returns to us
        responses = await self.reports[author_id].handle_message(message)
        for r in responses:
            await message.channel.send(r)

        # If the report is complete or cancelled, remove it from our map
        if self.reports[author_id].report_complete():
            current_report = self.reports[author_id]
            self.reports.pop(author_id)

            review = Review(self, current_report)
            await review.initiate_review()
            self.reviews[review.thread.id] = review

    async def handle_channel_message(self, message):
        # Ignore messages from the bot
        if message.author.id == self.user.id:
            return

        # Handle report reviews
        elif message.channel.id in self.reviews:
            await self.reviews[message.channel.id].handle_message(message)

    async def on_raw_reaction_add(self, payload):
        # Check if the reaction is in a guild and not from the bot itself
        if payload.guild_id and payload.user_id != self.user.id:
            guild = discord.utils.find(lambda g: g.id == payload.guild_id, self.guilds)
            if guild is None:
                return
            channel = guild.get_channel(payload.channel_id)
            if channel is None:
                return
            message = await channel.fetch_message(payload.message_id)
            member = guild.get_member(payload.user_id)

            # Check if the reaction equals the predefined emoji for reporting
            if str(payload.emoji) == report_emoji:
                channel = payload.member.dm_channel or await payload.member.create_dm()
                message = await self.get_channel(payload.channel_id).fetch_message(payload.message_id)
                if payload.member.id not in self.reports:
                    self.reports[payload.member.id] = Report(self)
                await self.reports[payload.member.id].initiate_report(channel, message)

    async def initiate_report(self, member, message):
        if member.dm_channel is None:
            await member.create_dm()
        await member.dm_channel.send(f"The message by {message.author.display_name} is about to be reported to the Trust & Safety Team of Stanford's CS152 Group-25: '{message.content}'")
        # Start the reporting process and send options in DM
        if member.id not in self.reports:
            self.reports[member.id] = Report(self)
        await self.reports[member.id].start_new_report(member.dm_channel, message)

    async def start_new_report(self, message):
        self.message = message
        # Start with the first question or confirmation
        reply = self.options_to_string(start_options)
        return [reply]

    def eval_text(self, message):
        '''
        TODO: Once you know how you want to evaluate messages in your channel, 
        insert your code here! This will primarily be used in Milestone 3. 
        '''
        return message

    def code_format(self, text):
        '''
        TODO: Once you know how you want to show that a message has been 
        evaluated, insert your code here for formatting the string to be 
        shown in the mod channel. 
        '''
        return "Evaluated: '" + text + "'"

client = ModBot()
client.run(discord_token)
