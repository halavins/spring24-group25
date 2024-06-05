from enum import Enum, auto
import discord
from uuid import uuid4
import json

class ModState(Enum):
    MOD_REPORT_AWAITING_REVIEW = auto()
    MOD_REPORT_START = auto()
    MOD_REPORT_ROMANCE_INVESTMENT = auto()
    MOD_REPORT_SCAM = auto()
    MOD_REPORT_SCAM_REPEATED = auto()
    MOD_REPORT_SCAM_FIRST_TIME = auto()
    MOD_REPORT_ESCALATE = auto()
    MOD_REPORT_COMPLETE = auto()
    MOD_REPORT_OTHER = auto()
    MOD_REPORT_MALICIOUS = auto()

class Review:
    def __init__(self, client, report, channel_override=None, case_override=None):
        self.client = client
        self.report = report
        self.modState = ModState.MOD_REPORT_AWAITING_REVIEW
        self.began = False
        self.channel_override = channel_override
        self.case_override = case_override

    async def initiate_review(self):
        # Get the mod channel from the initial message
        channel = self.channel_override if self.channel_override else self.client.mod_channels[self.report.message.guild.id]
        # Create the report message from this Report object
        report_message = self.report_to_string()
        # Send report thread to mod
        # Listen in the mod channel for threads to start a Review
        self.thread = await channel.create_thread(
            name=f"Report {uuid4()}",
            type=discord.ChannelType.public_thread
        )
        await self.thread.send(report_message)

        await self.thread.send(f"Report Status: Awaiting moderator review")
        self.modState = ModState.MOD_REPORT_AWAITING_REVIEW

        # Start the review process and send first option in thread
        await self.thread.send("Start moderation flow by answering the following questions")
        await self.thread.send(">> Is this a malicious user report? (yes/no)")

    async def handle_message(self, message: discord.Message):
        print("enter handle message", message)
        print("current mod state", self.modState)

        if self.modState == ModState.MOD_REPORT_AWAITING_REVIEW:
            if message.content.lower() == "yes":
                self.modState = ModState.MOD_REPORT_MALICIOUS
                await self.thread.send("System Action: Reporter is warned for malicious reports.")
                await self.thread.send("Reporting feature is suspended for the reporter account for 24 hours.")
                return
            if message.content.lower() == "no":
                self.modState = ModState.MOD_REPORT_START
                await self.thread.send(">> Is the reported conversation related to investment or romance? (yes/no)" )
                return

        if self.modState == ModState.MOD_REPORT_START:
            if message.content.lower() == "yes":
                self.modState = ModState.MOD_REPORT_ROMANCE_INVESTMENT
                await self.thread.send(">> Are there clear indicators of a potential scam? Is there evidence of "
                                       "coercion or manipulation that pose risks to user safety or privacy? ("
                                       "yes/no/unsure)")
                return
            if message.content.lower() == "no":
                self.modState = ModState.MOD_REPORT_COMPLETE
                await self.thread.send("Evaluate under standard community guidelines")
                await self.thread.send("No further actions required. Case closed.")
                return

        if self.modState == ModState.MOD_REPORT_ROMANCE_INVESTMENT:
            if message.content.lower() == "yes":
                self.modState = ModState.MOD_REPORT_SCAM
                await self.thread.send(
                    ">> Does this user have a history of scam violation? (yes/no)"
                )
                return
            if message.content.lower() == "no":
                self.modState = ModState.MOD_REPORT_OTHER
                await self.thread.send("No further actions required. Case closed.")
                return
            if message.content.lower() == "unsure":
                self.modState = ModState.MOD_REPORT_OTHER
                await self.thread.send(
                    ">> Flag for further review or investigation? (yes/no)"
                )
                return

        if self.modState == ModState.MOD_REPORT_SCAM:
            if message.content.lower() == "yes":
                self.modState = ModState.MOD_REPORT_SCAM_REPEATED
                await self.thread.send(
                    "System Action: User is permanently banned from the platform."
                )
                return
            if message.content.lower() == "no":
                self.modState = ModState.MOD_REPORT_SCAM_FIRST_TIME
                await self.thread.send(
                    "System Action: User is warned via email and next login. User will be temporarily restricted from "
                    "chat for 7 days."
                )
                return

        if self.modState == ModState.MOD_REPORT_OTHER:
            if message.content.lower() == "yes":
                self.modState = ModState.MOD_REPORT_ESCALATE
                await self.thread.send(
                    "System Action: This case has been escalated to senior members and management team"
                )
                return
            if message.content.lower() == "no":
                self.modState = ModState.MOD_REPORT_COMPLETE
                await self.thread.send(
                    "No further actions required. Case closed."
                )
                return

    def report_to_string(self):
        message = f"Message Report\n"
        message += "```" + json.dumps(self.case_override if self.case_override else self.report.case, indent=2) + "```"
        return message