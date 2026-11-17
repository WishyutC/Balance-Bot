from emotion import Emotion
from datetime import datetime
import csv

def emotion_to_scalar(emotion_value: int) -> float:
    mapping = {
        Emotion.SAD.value: 0,
        Emotion.ANGRY.value: 0,
        Emotion.FEAR.value: 0,
        Emotion.DISGUST.value: 0,
        Emotion.NEUTRAL.value: 5,
        Emotion.CALM.value: 5,
        Emotion.HAPPY.value: 10,
        Emotion.SURPRISE.value: 10,
    }
    scalar = mapping.get(emotion_value, 5)
    return scalar

def get_average_mood_from_csv(csv_path: str) -> float:
    moods = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            emotion_str = row.get('user_emotion', '').strip()
            try:
                moods.append(emotion_to_scalar(Emotion[emotion_str].value))
            except (KeyError, ValueError):
                continue
    if not moods:
        return 0.0
    return sum(moods) / len(moods)
