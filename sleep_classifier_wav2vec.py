import time
import argparse
from pathlib import Path
import torch, torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
from datetime import datetime
from torchmetrics.classification import BinaryF1Score, BinaryCohenKappa
from dataloader import LittleBeatsDataset
from transformers import AutoConfig, Wav2Vec2Processor
from LittleBeatsPrelim.wav2vec_scripts.datacollector import DataCollatorCTCWithPadding
from LittleBeatsPrelim.wav2vec_scripts.wav2vec_model import Wav2Vec2ForSpeechClassification
from transformers import TrainingArguments
from LittleBeatsPrelim.wav2vec_scripts.trainer import CTCTrainer

from datasets import load_dataset, load_metric
import numpy as np
import librosa
from transformers import EvalPrediction
from sklearn.metrics import classification_report

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', default='/home/kcchang3/data/LittleBeats/data_30s')
    parser.add_argument('--manifest_dir', default='/home/kcchang3/data/LittleBeats/manifest')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_path', default='/home/kcchang3/data/LittleBeats/manifest/wav2vec2-xlsr-english-speech-sleep-recognition/checkpoint-300')

    args = parser.parse_args()
    return args

def pred_smooth(pred, n=3):
    kernel = torch.ones(1, 1, n)
    pred = (F.conv1d(pred.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) / n).squeeze().round()
    return pred

# def evaluate_classifier(pred, y):
#     y = y.to(torch.int32)
#     pred = pred.to(torch.int32)
#     conf_matrix = ConfusionMatrix(num_classes=2)(pred, y)
#     accuracy = 1 - (((pred - y) ** 2).sum()) / (pred.shape[0])
#     f1 = BinaryF1Score()(pred, y)
#     kappa = BinaryCohenKappa()(pred, y)
#     return conf_matrix, accuracy, f1, kappa


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label




def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

if __name__ == '__main__':

    args = get_arguments()
    data_dir = Path(args.data_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir = manifest_dir / "wav2vec2-xlsr-english-speech-sleep-recognition"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start_time = time.time()
    # output_file = open(f'./output/{datetime.now().strftime("%d-%m-%Y-%H-%M")}.txt', 'w')
    warnings.filterwarnings("ignore")



    # Load dataset
    data_files = {
        "train": f"{manifest_dir / 'train.csv'}",
        "validation": f"{manifest_dir / 'test.csv'}",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    print(train_dataset)
    print(eval_dataset)

    # print out dataset summaries
    input_column = "path"
    output_column = "class"
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")
    # littlebeats_dataset = LittleBeatsDataset(data_dir)
    # val_portion = int(0.1*len(littlebeats_dataset))
    # subdataset = torch.utils.data.random_split(littlebeats_dataset, [len(littlebeats_dataset)-val_portion, val_portion], generator=torch.Generator().manual_seed(42))
    # train_dataloader = DataLoader(subdataset[0], batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(subdataset[1], batch_size=batch_size, shuffle=False)



    # Model specific setup
    batch_size = 8
    model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    pooling_mode = "mean"
    label_list = ['wake', 'sleep']
    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'pooling_mode', pooling_mode)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, )
    target_sampling_rate = processor.feature_extractor.sampling_rate
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path,config=config,)
    model.freeze_feature_extractor()



    def preprocess_function(examples, processor=processor, label_list=label_list, input_column=input_column, output_column=output_column, target_sampling_rate=target_sampling_rate):
        def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
            speech_array, sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            return speech
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
        target_list = [label_to_id(label, label_list) for label in examples[output_column]]
        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)
        return result
    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batch_size=100,
        batched=True,
        num_proc=4
    )




    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    is_regression = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=100,
        learning_rate=1e-4,
        save_total_limit=2,
    )
    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    if(args.train):
        trainer.train()



    '''
        Evaluation
    '''
    if(args.eval):
        model_name_or_path = args.ckpt_path
        config = AutoConfig.from_pretrained(model_name_or_path)
        # processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


        def speech_file_to_array_fn(batch, processor=processor):
            speech_array, sampling_rate = torchaudio.load(batch["path"])
            speech_array = speech_array.squeeze().numpy()
            speech_array = librosa.resample(np.asarray(speech_array), sampling_rate,
                                            processor.feature_extractor.sampling_rate)
            batch["speech"] = speech_array
            return batch

        def predict(batch, processor=processor):
            features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
            input_values = features.input_values.to(device)
            attention_mask = features.attention_mask.to(device)
            with torch.no_grad():
                logits = model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            batch["predicted"] = pred_ids
            return batch

        eval_dataset = eval_dataset.map(speech_file_to_array_fn)
        result = eval_dataset.map(predict, batched=True, batch_size=8)
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        y_true = [config.label2id[name] for name in result["class"]]
        y_pred = result["predicted"]
        print(classification_report(y_true, y_pred, target_names=label_names))