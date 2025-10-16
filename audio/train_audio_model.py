from transformers import Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    num_labels=2
)
