import json
import os
import wandb


class Logger:

    def __init__(self, path, use_wandb=False):  # TODO JL 10/26/24 config?
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = {}
        self.use_wandb = use_wandb

    def log(self, key, value, iter):
        # Store locally
        if key not in self.data:
            self.data[key] = {
                "iter": [],
                "value": [],
            }
        self.data[key]["iter"].append(iter)
        self.data[key]["value"].append(value)

        # Log to wandb if enabled
        if self.use_wandb:
            # Handle different types of values
            if isinstance(value, list):
                # For metrics that return lists
                for i, v in enumerate(value):
                    wandb.log({f"{key}_{i}": v}, step=iter)
            else:
                wandb.log({key: value}, step=iter)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def load(self):
        with open(self.path, "r") as f:
            self.data = json.load(f)
        return self.data

    def get(self, key):
        return self.data[key]

    def get_iters(self, key):
        return self.data[key]["iter"]

    def get_values(self, key):
        return self.data[key]["value"]

    def get_most_recent_value(self, key):
        return self.data[key]["value"][-1]
