import io
import json
import logging
import os
import random
import re
import textwrap
from collections import deque

import discord
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

from rules.evaluator import RuleEngine
from review import Review

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


class DetectorBot(discord.Client):
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
        self.rule_evaluator = RuleEngine()
        self.alert_queue = {}
        self.thread_id_to_review = {}
        self.thread_id_to_review_id = {}
        self.reviews = {}

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
        if message.author.id == self.user.id and not self.should_evaluate(message):
            return

        if message.guild:
            await self.handle_channel_message(message)


    def construct_classification_input(self, message: discord.Message):

        chat_history = ' '.join([z.strip(message.author.display_name + ': ') for z in self.threaded_conversations[message.channel.id]
                                 if message.author.display_name in z])

        # Mocking the input since Discord Bot doesn't have access, though this is important signal to capture
        ip_fraud_score = random.randint(50, 100)
        channel_topic = message.channel.name
        convo_history = f"""Channel: {channel_topic} \nFraud Score: {ip_fraud_score} \nMessage: {chat_history}"""
        return convo_history
    
    async def update_review(self, thread_id, current_score, case_update, new_status):
        self.session_risk[thread_id]['highest_score'] = current_score
        review = self.thread_id_to_review
        review.case_override = case_update
        review_id = self.thread_id_to_review_id[thread_id]
        review_thread = client.get_channel(review_id)
        await review_thread.send(review.report_to_string())
        self.alert_queue[thread_id] = {
            'status': new_status
        }


    async def handle_channel_message(self, message: discord.Message):

        discussion_channel = message.channel.name
        if isinstance(message.channel, discord.Thread):
            discussion_channel = message.channel.parent.name
        
        # Handle report reviews
        elif message.channel.id in self.reviews:
            await self.reviews[message.channel.id].handle_message(message)
        
        # allow mods in mod channel to @bot to be added to highest priority thread
        if discussion_channel == f'group-{self.group_num}-mod':
            if 'Group 25 Bot' not in [mention.name for mention in message.mentions]:
                return
            try:
                await message.delete()
            except Exception as e:
                print(f"Caught exception during delete: {e}")
            
            ####
            # TODO: remove this block of test setup
            # try:
            #     test_thread = await message.channel.create_thread(name="Test thread", type=discord.ChannelType.public_thread)
            # except Exception as e:
            #     print(f"Caught exception during thread creation: {e}")  
            # self.session_risk[test_thread.id] = {
            #     'highest_score': 10
            # }
            # self.alert_queue[test_thread.id] = {
            #     'status': 'alerted'
            # }
            ####

            # get top priority thread in 'alerted' status
            highest_score = 0
            highest_thread_id = None 
            for thread_id in self.session_risk:
                if self.session_risk[thread_id]['highest_score'] > highest_score and self.alert_queue[thread_id] and self.alert_queue[thread_id]['status'] == 'alerted':
                    highest_score = self.session_risk[thread_id]['highest_score']
                    highest_thread_id = thread_id
            
            thread = client.get_channel(highest_thread_id)
            await thread.send(f"<@{message.author.id}>")
            return

        if not discussion_channel == f'group-{self.group_num}':
            return

        thread_id = message.channel.id
        self.track_conversation(message, thread_id)

        mod_channel = self.mod_channels[message.guild.id]
        if mod_channel:
            if self.should_evaluate(message):
                features = self.construct_classification_input(message)
                rule_eval_results = self.rule_evaluator.evaluate(message, features)
                under_monitoring = (thread_id in self.alert_queue and self.alert_queue[thread_id]['status'] not in ['dismissed', 'banned'])

                if not rule_eval_results and not under_monitoring:
                    return
                
                if message.channel.id not in self.alert_queue:
                    # Initialize review
                    self.alert_queue[message.channel.id] = {
                        'status': 'monitored'
                    }
                    case_override = f"""Discussion {thread_id} Added to Watchlist for Advanced Monitoring!\n""" + '\n'.join(['\t\t-' + z for z in rule_eval_results])
                    review = Review(client, "I get overriden", channel_override=mod_channel, case_override=case_override)
                    await review.initiate_review()
                    self.thread_id_to_review[thread_id] = review
                    self.reviews[review.thread.id] = review
                    self.thread_id_to_review_id[thread_id] = review.thread.id

                    # await review_thread.send(
                    #     f"""Discussion {thread_id} Added to Watchlist for Advanced Monitoring!\n""" + '\n'.join(['\t\t-' + z for z in rule_eval_results]))

                # Test end line here
                # return

                evaluation_result = self.evaluate_risk(message)
                evaluation_result_dict = self.code_format(evaluation_result)
                current_score = evaluation_result_dict["score"]
                highest_score = self.session_risk[thread_id]['highest_score']
                self.session_risk[thread_id]['entries'].append({
                    'datetime': message.created_at,
                    'score': current_score,
                    'message': message.content,
                    'explanation': evaluation_result_dict["explanation"],
                })
                # If evaluation result is 0, send message in mod thread indicating conversation no longer monitored
                if current_score == 0:
                    self.update_review(thread_id, current_score, "Case dismissed", "dismissed")
                # If evaluation result is > 90, send message in mod thread indicating user is banned
                if current_score > 90:
                    self.update_review(thread_id, current_score, "User is banned", "banned")

                if evaluation_result_dict["score"] > 60 and (current_score > highest_score + 10):
                    new_case = f"""\nScam Detected\nScammer: {evaluation_result_dict['scammer']}\nVictim: {evaluation_result_dict['victim']}\nMessage: {message.content}\nScore: {evaluation_result_dict["score"]}\n""" 
                    new_case += f'''\nExplanation: {evaluation_result_dict["explanation"]}'''

                    review_thread = self.update_review(thread_id, current_score, new_case, "alerted")

                    await self.plot_scores(review_thread, self.mod_channels[thread_id])
                    await self.plot_radar(review_thread, evaluation_result_dict['deception_risk_factors'])

    async def plot_radar(self, mod_channel, radar_data):
        data = radar_data
        # Extract the labels, scores, and explanations from the data
        labels = list(data.keys())
        scores = [data[label]['score'] for label in labels]
        explanations = [data[label]['explanation'] for label in labels]

        # Number of variables
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        scores += scores[:1]  # To complete the loop
        angles += angles[:1]  # To complete the loop

        # Create the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        ax.fill(angles, scores, color='red', alpha=0.25)
        ax.plot(angles, scores, color='red', linewidth=2)

        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([1, 2, 3], ["1", "2", "3"], color="grey", size=10)
        plt.ylim(0, 3)

        # Add title
        plt.title('Scam Detection Radar Plot', size=20, color='red', y=1.1)

        # Annotate each point with the explanation
        for i, (angle, score, explanation) in enumerate(zip(angles[:-1], scores[:-1], explanations)):
            if score != 0:
                wrapped_text = textwrap.fill(explanation, width=50)
                ax.annotate(f'{score}\n{wrapped_text}', xy=(angle, score), xytext=(angle, score + 0.5),
                            textcoords='data', ha='center', va='center', fontsize=9,
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Send the plot to the Discord channel
        await mod_channel.send(file=discord.File(buf, filename='scam_detection_radar.png'))

    async def plot_scores(self, thread_id, mod_channel):
        data_entries = self.session_risk[thread_id]['entries']
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
        if isinstance(message.channel, discord.Thread) and 'Scam' in message.channel.name:
            return True
        return False


    def evaluate_risk(self, message: discord.Message):
        # Collecting conversation history from the thread
        past_convos = list(self.threaded_conversations[message.channel.id])
        conversation_history = '\n'.join(past_convos)


        prompt = (f"""
        \n Here is the conversation so far: \n\n""" + conversation_history +
                  """\n\n 
                  """)

        try:
            # Call to the OpenAI API with the accumulated thread conversation as context
            client = OpenAI(api_key=openai_token)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "system",
                        "content": """
                        You are a scam detection algorithm to detect whether there is pig butchering scam. In Pig Butchering scam, scammers would do something like the following

                                1) Building Trust: Scammers start with friendly and engaging conversations to build a rapport and trust over time.
                                2) Gradual Escalation: They slowly introduce the idea of a financial opportunity, often framed as exclusive and time-sensitive.
                                3) Financial Opportunity: The scam typically involves suggesting an investment in cryptocurrency, stocks, or a similar venture, promising high returns.
                                4) Urgency and Secrecy: Scammers create a sense of urgency and encourage keeping the investment opportunity confidential.
                                5) Manipulation: Use emotional manipulation to pressure the victim into making quick decisions.
    
                        Please check if the conversation have these elements and evaluate the conversations. Your goal is to check in this conversation who is the scammer and who is the victim. You always return results in JSON format. Do note that it is normal for regular people to have conversations that involve investments, but it usually happens between acquaintances. Pay particular attention to signs of deception.

                        As part of outcome, please also evaluate each of the following deception risk factors on a scale from 0 (None), 1 (low), 2 (medium), 3 (high) and explain why:
                            1) Distraction: The scammer uses tactics to divert the victim's attention away from key facts or inconsistencies.
                            2) Social Compliance: The scammer relies on perceived authority or societal norms to persuade the victim.
                            3) Herd: The scammer suggests that others believe or do something to make the victim feel safer in compliance.
                            4) Dishonesty: The scammer leverages the victim's own dishonest or unethical behavior to manipulate them further.
                            5) Kindness: The scammer exploits the victim's good nature and willingness to help. 
                            6) Need and Greed: The scammer manipulates the victim’s desires or needs to influence their behavior.
                            7) Time Pressure: The scammer imposes urgency to limit the victim’s decision-making time.
                            8) Discretion: The scammer will try to move the conversation to a encrypted chatting app such as telegram, whatsapp, signal, etc. They also know that there are automated systems for finding this, so will try to avoid ways of getting detected. 
                            9) Scripted: The scammer will stick to a script and wouldn't want to waste time, sometimes repeating same information. 

                        """
                    }
                ],
                temperature=1,
                max_tokens=2560,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                function_call={
                    "name": "llm_evaluation"
                },
                functions=[{
                    "name": "llm_evaluation",
                    "description": "Evaluation results from LLM",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": "A risk score between 0 and 100 indicating how suspicious this message sender is"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "An explanation for the risk score"
                            },
                            "scammer": {
                                "type": "string",
                                "description": "Name of the scammer in the conversation"
                            },
                            "victim": {
                                "type": "string",
                                "description": "Name of the victim in the conversation"
                            },
                            "deception_risk_factors": {
                                "type": "object",
                                "properties": {
                                    "distraction": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "social_compliance": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "herd": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "dishonesty": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "kindness": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "need_and_greed": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "time_pressure": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "discretion": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    },
                                    "scripted": {
                                        "type": "object",
                                        "properties": {
                                            "score": {
                                                "type": "number",
                                                "description": "Assign 0 for Absent, 1 for Low, 2 for Medium, 3 for High presence of the risk factor"
                                            },
                                            "explanation": {
                                                "type": "string",
                                                "description": "Explain the magnitude for the score in detail with evidence and critical reasoning"
                                            }
                                        }
                                    }
                                },
                                "required": [
                                    "distraction",
                                    "social_compliance",
                                    "herd",
                                    "dishonesty",
                                    "kindness",
                                    "need_and_greed",
                                    "time_pressure",
                                    "discretion",
                                    "scripted"
                                ]
                            }
                        },
                        "required": ["score", "explanation", "scammer", "victim", "deception_risk_factors"]
                    }
                }]
            )
            return response.choices[0].message.function_call.arguments
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Sorry, I encountered an error while processing your request."

    def track_conversation(self, message, thread_id):
        if thread_id not in self.threaded_conversations:
            self.threaded_conversations[thread_id] = deque(maxlen=1000)
            self.session_risk[thread_id] = {'highest_score': 0, 'entries': []}
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

client = DetectorBot()
client.run(discord_token)