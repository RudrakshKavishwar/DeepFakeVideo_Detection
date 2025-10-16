import random

def dummy_action_localization(num_actions=2):
    """Fake action localization for demo"""
    actions = [
        {"action": "walking", "confidence": random.uniform(0.7, 0.95), "start_time": 0, "end_time": 5},
        {"action": "talking", "confidence": random.uniform(0.6, 0.9), "start_time": 6, "end_time": 12}
    ]
    return actions[:num_actions]
