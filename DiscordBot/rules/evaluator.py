import json
import discord

from model.model_inference import ScamClassifier


class RuleEngine:

    def __init__(self):
        with open("rules/rule_set.json", 'r') as file:
            self.rule_set = json.load(file)

        self.classifier = ScamClassifier()

    def evaluate(self, message: discord.Message, features: str):
        violations = []

        # Keyword matching
        for keyword in self.rule_set.get("keywords", []):
            if keyword in message.content:
                violations.append(f"Message contains prohibited keyword: {keyword}")

        # Website matching
        for website in self.rule_set.get("websites", []):
            if website in message.content:
                violations.append(f"Message contains prohibited website: {website}")

        # IP matching
        for ip in self.rule_set.get("ips", []):
            if ip in message.content:
                violations.append(f"Message contains prohibited IP address: {ip}")


        # Initial screening model risk assessment
        if self.rule_set.get("classifier", False):
            if self.classifier.predict_scammer(features) == 'Scam':
                violations.append(f"Message classified as high risk by screening Model")

        return violations

