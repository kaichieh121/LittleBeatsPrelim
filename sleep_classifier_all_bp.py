import time
import argparse
from pathlib import Path
import torch, torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
from datetime import datetime
from dataloader import LittleBeatsDataset
from transformers import AutoConfig, Wav2Vec2Processor
from LittleBeatsPrelim.wav2vec_scripts.datacollector import DataCollatorCTCWithPadding
from LittleBeatsPrelim.wav2vec_scripts.wav2vec_audio_ecg_model import AllModalityModel, OneModalityModel, TwoModalityModel, AllThreeLayerClassifier
from transformers import TrainingArguments
from LittleBeatsPrelim.wav2vec_scripts.trainer import CTCTrainer
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa, MulticlassAccuracy

from datasets import load_dataset, load_metric
import numpy as np
import librosa
from transformers import EvalPrediction
from sklearn.metrics import classification_report

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--manifest_dir', default='/home/kcchang3/data/LittleBeats/manifest')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=64)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_from_ckpt', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_path', default='/home/kcchang3/data/LittleBeats/manifest/wav2vec2-xlsr-english-speech-sleep-recognition/checkpoint-300')
    parser.add_argument('--cache_path', default='~/.cache/huggingface/datasets')
    parser.add_argument('--imu_embedding_dir', default="D:\\Projects\\LittleBeatsPrelim_HAL\\LIMU-BERT-Public\\embed")
    parser.add_argument('--embedding_type', default='uci')

    args = parser.parse_args()
    return args


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
        print(evaluate_classifier(torch.tensor(preds), torch.tensor(p.label_ids)))
        return {"accuracy": MulticlassAccuracy(3)(torch.tensor(preds), torch.tensor(p.label_ids))}


def evaluate_classifier(pred, y):
    y = y.to(torch.int32)
    pred = pred.to(torch.int32)
    conf_matrix = ConfusionMatrix(num_classes=3)(pred, y)
    accuracy = MulticlassAccuracy(3)(pred, y)
    f1 = MulticlassF1Score(3)(pred, y)
    kappa = MulticlassCohenKappa(3)(pred, y)
    return conf_matrix, accuracy, f1, kappa

if __name__ == '__main__':

    args = get_arguments()
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir = manifest_dir / "all-modality-manifest"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "checkpoint-best"
    imu_embedding_dir = Path(args.imu_embedding_dir)

    embedding_type = args.embedding_type
    embedding_dict = {'audio': 0, 'ecg': 1, 'uci': 2}


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start_time = time.time()
    warnings.filterwarnings("ignore")



    # Load dataset
    data_files = {
        "train": f"{manifest_dir / 'train_3.csv'}",
        "validation": f"{manifest_dir / 'test_3.csv'}",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", cache_dir=args.cache_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    print(train_dataset)
    print(eval_dataset)

    # print out dataset summaries
    output_column = "class"
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")



    # Model specific setup
    model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    pooling_mode = "mean"
    # label_list = ['high_positive', 'low_positive', 'neutral', 'low_negative', 'high_negative']
    label_list = ['positive', 'neutral', 'negative']
    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=3,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf"
    )
    setattr(config, 'pooling_mode', pooling_mode)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, )
    target_sampling_rate = processor.feature_extractor.sampling_rate



    if(args.train or args.train_from_ckpt):
        if (args.train_from_ckpt):
            print(f"Training from ckpt {args.ckpt_path}")
            model_name_or_path = args.ckpt_path
            model = AllModalityModel.from_pretrained(model_name_or_path, mode=embedding_dict[embedding_type])
            # model = TwoModalityModel.from_pretrained(model_name_or_path)
            model.load_state_dict(torch.load(Path(model_name_or_path) / "pytorch_model.bin"))
            # model.classifier = AllThreeLayerClassifier(config)
            # model.num_labels = 3
        else:
            print("Training from scratch")
            model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            model = AllModalityModel.from_pretrained(model_name_or_path, config=config, mode=embedding_dict[embedding_type])
            # model = TwoModalityModel.from_pretrained(model_name_or_path, config=config)


        model.freeze_feature_extractor()
        # for param in model.wav2vec2_audio.parameters():
        #     param.requires_grad = False
        # for param in model.wav2vec2_ecg.parameters():
        #     param.requires_grad = False

        def preprocess_function(examples, processor=processor, label_list=label_list,
                                output_column=output_column, target_sampling_rate=target_sampling_rate):
            def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
                speech_array, sampling_rate = torchaudio.load(path)
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech

            speech_list = [speech_file_to_array_fn(path) for path in examples['audio_path']]
            ecg_list = [speech_file_to_array_fn(path) for path in examples['ecg_path']]
            uci_list = [np.load(path) for path in examples['uci_path']]

            audio_list = []
            for i in range(len(speech_list)):
                audio_list.append(np.concatenate((np.expand_dims(speech_list[i], axis=0), \
                                                  np.expand_dims(ecg_list[i], axis=0), \
                                                  np.expand_dims(np.concatenate((uci_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0)),axis=0))
            target_list = [label_to_id(label, label_list) for label in examples[output_column]]
            result = processor(audio_list, sampling_rate=target_sampling_rate)

            result["labels"] = list(target_list)
            return result


        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=2
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=2
        )

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        is_regression = False

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=1,
            evaluation_strategy="steps",
            num_train_epochs=args.num_train_epochs,
            fp16=True,
            save_steps=200,
            eval_steps=200,
            logging_steps=100,
            learning_rate=1e-4,
            # weight_decay=1e-500,
            save_total_limit=10,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            gradient_checkpointing=True,

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
        trainer.train()
        trainer.save_model(best_model_dir)



    '''
        Evaluation
    '''
    if(args.eval):
        model_name_or_path = args.ckpt_path
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AllModalityModel.from_pretrained(model_name_or_path, mode=embedding_dict[embedding_type]).to(device)
        # model = TwoModalityModel.from_pretrained(model_name_or_path).to(device)
        model.load_state_dict(torch.load(Path(model_name_or_path) / "pytorch_model.bin"))

        # def speech_file_to_array_fn(batch, processor=processor):
        #     speech_array, sampling_rate = torchaudio.load(batch['audio_path'])
        #     speech_array = speech_array.squeeze().numpy()
        #     speech_array = librosa.resample(np.asarray(speech_array), sampling_rate,
        #                                     processor.feature_extractor.sampling_rate)
        #     ecg_array, sampling_rate = torchaudio.load(batch['ecg_path'])
        #     ecg_array = ecg_array.squeeze().numpy()
        #     ecg_array = librosa.resample(np.asarray(ecg_array), sampling_rate,
        #                                     processor.feature_extractor.sampling_rate)
        #     audio_array = np.concatenate((np.expand_dims(speech_array, axis=0), np.expand_dims(ecg_array, axis=0)), axis=0)
        #     batch["speech"] = audio_array
        #     return batch

        def preprocess_function(examples, processor=processor, label_list=label_list,
                                output_column=output_column, target_sampling_rate=target_sampling_rate):
            def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
                speech_array, sampling_rate = torchaudio.load(path)
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech

            speech_list = [speech_file_to_array_fn(path) for path in examples['audio_path']]
            ecg_list = [speech_file_to_array_fn(path) for path in examples['ecg_path']]
            uci_list = [np.load(path) for path in examples['uci_path']]


            audio_list = []
            for i in range(len(speech_list)):
                audio_list.append(np.concatenate((np.expand_dims(speech_list[i], axis=0), \
                                                  np.expand_dims(ecg_list[i], axis=0), \
                                                  np.expand_dims(np.concatenate((uci_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0)),axis=0))
            target_list = [label_to_id(label, label_list) for label in examples[output_column]]
            result = processor(audio_list, sampling_rate=target_sampling_rate)

            result["labels"] = list(target_list)
            return result
        def predict(batch, processor=processor):
            features = processor(batch["input_values"], sampling_rate=processor.feature_extractor.sampling_rate,
                                 return_tensors="pt", padding=True)
            input_values = features.input_values.to(device)
            attention_mask = features.attention_mask.to(device)
            with torch.no_grad():
                logits = model(input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            batch["predicted"] = pred_ids
            return batch

        eval_dataset = train_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=2
        )
        result = eval_dataset.map(predict, batched=True, batch_size=8)
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        y_true = [config.label2id[name] for name in result["class"]]
        y_pred = result["predicted"]
        print(classification_report(y_true, y_pred, target_names=label_names))
        conf_matrix, accuracy, f1, kappa = evaluate_classifier(torch.tensor(y_pred), torch.tensor(y_true))
        print(conf_matrix, accuracy, f1, kappa)

        # train_dataset = train_dataset.map(speech_file_to_array_fn)
        # result = train_dataset.map(predict, batched=True, batch_size=8)
        # label_names = [config.id2label[i] for i in range(config.num_labels)]
        # y_true = [config.label2id[name] for name in result["class"]]
        # y_pred = result["predicted"]
        # print(classification_report(y_true, y_pred, target_names=label_names))
        # conf_matrix, accuracy, f1, kappa = evaluate_classifier(torch.tensor(y_pred), torch.tensor(y_true))
        # print(conf_matrix, accuracy, f1, kappa)