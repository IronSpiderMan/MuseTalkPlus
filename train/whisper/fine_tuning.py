import os.path
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import evaluate
from dataclasses import dataclass
from datasets import load_dataset, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
)


def prepare_dataset_new(batch):
    """Function to preprocess the dataset with the .map method"""
    audio = batch["filepath"]
    transcription = batch["text"]

    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]

    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."

    batch["text"] = transcription
    batch["labels"] = tokenizer(batch["text"]).input_ids
    batch["input_features"] = feature_extractor(audio['array'], sampling_rate=16000).input_features[0]
    return batch


def prepare_dataset(batch):
    """Function to preprocess the dataset with the .map method"""
    audio = batch["audio"]
    transcription = batch["sentence"]

    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]

    if transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."

    batch["sentence"] = transcription
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# 加载模型、tokenizer、processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="chinese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="chinese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.generation_config.language = "chinese"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
dataset_dir = Path(os.path.dirname(__file__)) / "edgetts_zh"
cv11_train = load_dataset('csv', data_files=str(dataset_dir / "train.csv"), split='train')
cv11_train = cv11_train.cast_column('filepath', Audio())
cv11_train = cv11_train.map(
    prepare_dataset_new,
    desc="preprocess dataset",
)
# 查看数据
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-chinese",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=cv11_train,
    eval_dataset=cv11_train,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()
