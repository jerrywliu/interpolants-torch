import json
import os


class Logger:

    def __init__(self, path):  # TODO JL 10/26/24 config?
        self.path = path
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.data = {}

    def log(self, key, value, iter):
        if key not in self.data:
            self.data[key] = {
                "iter": [],
                "value": [],
            }
        self.data[key]["iter"].append(iter)
        self.data[key]["value"].append(value)

    def log(self, dict, iter):
        for key, value in dict.items():
            if key not in self.data:
                self.data[key] = {
                    "iter": [],
                    "value": [],
                }
            self.data[key]["iter"].append(iter)
            self.data[key]["value"].append(value)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)
        return self.data

    def get(self, key):
        return self.data[key]
