import os, sys
sys.path.insert(0, os.path.abspath(".."))
import time
import argparse
from pathlib import Path
import torch, torchaudio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import warnings
import pandas as pd
from datetime import datetime
from dataloader import LittleBeatsDataset
from transformers import AutoConfig, Wav2Vec2Processor
from LittleBeatsPrelim.wav2vec_scripts.datacollector import DataCollatorCTCWithPadding
# from LittleBeatsPrelim.wav2vec_scripts.wav2vec_audio_ecg_model import AllModalityModel, OneModalityModel
from LittleBeatsPrelim.wav2vec_scripts.wav2vec2_stereo_model import AllModalityModel, create_model

from transformers import TrainingArguments
from LittleBeatsPrelim.wav2vec_scripts.trainer import CTCTrainer
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import BinaryF1Score, BinaryCohenKappa

from datasets import load_dataset, load_metric
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import classification_report


def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--manifest_dir', default='D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\manifest')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_from_ckpt', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_path', default='/home/kcchang3/data/LittleBeats/manifest/wav2vec2-xlsr-english-speech-sleep-recognition/checkpoint-300')
    parser.add_argument('--cache_path', default='~/.cache/huggingface/datasets')
    parser.add_argument('--embedding_type', default='uci')
    parser.add_argument('--limu_pretrained_model', default="D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\limu_bert\\saved\\pretrain_base_littlebeats_20_120\\limu_v3.pt")
    parser.add_argument('--mode', choices=['mono', 'stereo', 'stereo+limu', 'triple'])
    args = parser.parse_args()
    return args


def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label

def down_sample(data, window_sample, start, end):
    data = data.numpy()
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(start, end - window, window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window, start, end))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, start, end))
                i += window
    return np.array(result)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        print(evaluate_classifier(torch.tensor(preds), torch.tensor(p.label_ids)))
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def evaluate_classifier(pred, y):
    y = y.to(torch.int32)
    pred = pred.to(torch.int32)
    conf_matrix = ConfusionMatrix(num_classes=2)(pred, y)
    accuracy = 1 - (((pred - y) ** 2).sum()) / (pred.shape[0])
    f1 = BinaryF1Score()(pred, y)
    kappa = BinaryCohenKappa()(pred, y)
    return conf_matrix, accuracy, f1, kappa

if __name__ == '__main__':

    args = get_arguments()
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir = manifest_dir / "w2v-audio-and-ecg"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "checkpoint-best"

    embedding_type = args.embedding_type
    embedding_dict = {'audio': 0, 'ecg': 1, 'uci': 2, 'hhar': 3, 'motion': 4, 'shoaib': 5}


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    start_time = time.time()
    warnings.filterwarnings("ignore")



    # Load dataset
    data_files = {
        "train": f"{manifest_dir / 'train.csv'}",
        "validation": f"{manifest_dir / 'val.csv'}",
        "test": f"{manifest_dir / 'test.csv'}"
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", cache_dir=args.cache_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    print(train_dataset)
    print(eval_dataset)

    # print out dataset summaries
    output_column = "class"
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")



    # Model specific setup
    huggingface_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    model_name_or_path = manifest_dir / 'pretrained_weights'
    pooling_mode = "mean"
    label_list = ['wake', 'sleep']
    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        num_hidden_layers=12
    )
    setattr(config, 'limu_pretrained_model', args.limu_pretrained_model)
    setattr(config, 'pooling_mode', pooling_mode)
    setattr(config, 'mode', args.mode)
    processor = Wav2Vec2Processor.from_pretrained(huggingface_path, )
    target_sampling_rate = processor.feature_extractor.sampling_rate

    lb_audio_pretrained_weights = torch.load(manifest_dir / 'pretrained_weights' / 'lb_lena_4300hr.pt')


    if(args.train or args.train_from_ckpt):
        if (args.train_from_ckpt):
            print(f"Training from ckpt {args.ckpt_path}")
            model_name_or_path = args.ckpt_path
            model = AllModalityModel.from_pretrained(model_name_or_path, embedding_type=embedding_dict[embedding_type])
            model.load_state_dict(torch.load(Path(model_name_or_path) / "pytorch_model.bin"))
        else:
            print("Training from scratch")
            model_name_or_path = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

            model = create_model(config=config,
                                 embedding_type=embedding_dict[embedding_type],
                                 model_name_or_path=model_name_or_path,
                                 lb_audio_pretrained_weights=lb_audio_pretrained_weights)


        model.freeze_feature_extractor()

        def preprocess_function(examples, processor=processor, label_list=label_list,
                                output_column=output_column, target_sampling_rate=target_sampling_rate):
            def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
                speech_array, sampling_rate = torchaudio.load(path)
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech
            def imu_to_limu_format(path, raw_sr=150, target_sr=20):
                pd_file = pd.read_csv(path.__str__(), sep=' ', header=None)
                tmp_data = torch.tensor(pd_file.values)
                tmp_data = down_sample(tmp_data, raw_sr / target_sr, 0, tmp_data.shape[0] - 1)
                tmp_data = np.concatenate((tmp_data, np.expand_dims(tmp_data[-1], 0)), 0)[:,:6]
                tmp_data = tmp_data.reshape(tmp_data.shape[0]*tmp_data.shape[1])
                return tmp_data
            speech_list = [speech_file_to_array_fn(path) for path in examples['audio_path']]
            limu_list = [imu_to_limu_format(path) for path in examples['imu_path']]
            ecg_list = [speech_file_to_array_fn(path) for path in examples['ecg_path']]
            uci_list = [np.load(path) for path in examples['uci_path']]
            hhar_list = [np.load(path) for path in examples['hhar_path']]
            motion_list = [np.load(path) for path in examples['motion_path']]
            shoaib_list = [np.load(path) for path in examples['shoaib_path']]


            audio_list = []
            for i in range(len(speech_list)):
                audio_list.append(np.concatenate((np.expand_dims(speech_list[i], axis=0), \
                                                  np.expand_dims(ecg_list[i], axis=0), \
                                                  np.expand_dims(np.concatenate((uci_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0),
                                                  np.expand_dims(np.concatenate((hhar_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((motion_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((shoaib_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((limu_list[i], np.zeros(speech_list[0].shape[0] - limu_list[0].shape[0]))), axis=0)),axis=0))
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
            save_steps=100,
            eval_steps=100,
            logging_steps=50,
            learning_rate=1e-4,
            save_total_limit=10,
            save_strategy="steps",
            load_best_model_at_end=True,
            # metric_for_best_model='accuracy',
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
        model = create_model(config=config, embedding_type=embedding_dict[embedding_type], model_name_or_path=model_name_or_path, lb_audio_pretrained_weights=lb_audio_pretrained_weights)
        model = model.to(device)
        # model = AllModalityModel.from_pretrained(model_name_or_path, mode=embedding_dict[embedding_type]).to(device)
        model.load_state_dict(torch.load(Path(model_name_or_path) / "pytorch_model.bin"))

        def preprocess_function(examples, processor=processor, label_list=label_list,
                                output_column=output_column, target_sampling_rate=target_sampling_rate):
            def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
                speech_array, sampling_rate = torchaudio.load(path)
                resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
                speech = resampler(speech_array).squeeze().numpy()
                return speech
            def imu_to_limu_format(path, raw_sr=150, target_sr=20):
                pd_file = pd.read_csv(path.__str__(), sep=' ', header=None)
                tmp_data = torch.tensor(pd_file.values)
                tmp_data = down_sample(tmp_data, raw_sr / target_sr, 0, tmp_data.shape[0] - 1)
                tmp_data = np.concatenate((tmp_data, np.expand_dims(tmp_data[-1], 0)), 0)[:,:6]
                tmp_data = tmp_data.reshape(tmp_data.shape[0]*tmp_data.shape[1])
                return tmp_data
            speech_list = [speech_file_to_array_fn(path) for path in examples['audio_path']]
            ecg_list = [speech_file_to_array_fn(path) for path in examples['ecg_path']]
            uci_list = [np.load(path) for path in examples['uci_path']]
            hhar_list = [np.load(path) for path in examples['hhar_path']]
            motion_list = [np.load(path) for path in examples['motion_path']]
            shoaib_list = [np.load(path) for path in examples['shoaib_path']]
            limu_list = [imu_to_limu_format(path) for path in examples['imu_path']]


            audio_list = []
            for i in range(len(speech_list)):
                audio_list.append(np.concatenate((np.expand_dims(speech_list[i], axis=0), \
                                                  np.expand_dims(ecg_list[i], axis=0), \
                                                  np.expand_dims(np.concatenate((uci_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0),
                                                  np.expand_dims(np.concatenate((hhar_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((motion_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((shoaib_list[i], np.zeros(speech_list[0].shape[0] - uci_list[0].shape[0]))), axis=0), \
                                                  np.expand_dims(np.concatenate((limu_list[i], np.zeros(speech_list[0].shape[0] - limu_list[0].shape[0]))), axis=0)),axis=0))
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

        eval_dataset = test_dataset.map(
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