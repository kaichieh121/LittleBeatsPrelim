import os, sys
sys.path.insert(0, os.path.abspath(".."))
import time
import argparse
from pathlib import Path
import torch, torchaudio
import warnings
import pandas as pd
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, PreTrainedTokenizer
from LittleBeatsPrelim.wav2vec_scripts.datacollector import DataCollatorCTCWithPadding
from LittleBeatsPrelim.wav2vec_scripts.wav2vec2_stereo_model import create_model

from transformers import TrainingArguments
from LittleBeatsPrelim.wav2vec_scripts.trainer import CTCTrainer

from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
from sklearn.metrics import classification_report

from torch.utils.data.dataloader import DataLoader
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from LittleBeatsPrelim.wav2vec_scripts.modeling_wav2vec2_custom import Wav2Vec2ForPreTraining
import math
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
import datasets
from huggingface_hub import Repository, create_repo
logger = get_logger(__name__)

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--manifest_dir', default='D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\manifest')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument('--ckpt_path', default='/home/kcchang3/data/LittleBeats/manifest/wav2vec2-xlsr-english-speech-sleep-recognition/checkpoint-300')
    parser.add_argument('--cache_path', default='~/.cache/huggingface/datasets')
    parser.add_argument('--embedding_type', default='audio')
    parser.add_argument('--audio_pretrained_model', default="D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\manifest\\pretrained_weights\\lb_lena_4300hr.pt")
    parser.add_argument('--ecg_pretrained_model', default="D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\manifest\\pretrained_weights\\bp_ecg_500hr.pt")
    parser.add_argument('--limu_pretrained_model', default="D:\\Projects\\LittleBeatsPrelim_HAL\\LittleBeatsPrelim\\manifest\\pretrained_weights\\limu\\limu_v3.pt")
    parser.add_argument('--mode')
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=1,
        help="Percentage of training data that should be used for validation if no validation is present in dataset.",
    )
    parser.add_argument(
        "--mask_time_prob",
        type=float,
        default=None,
        help=(
            "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the"
            " contrastive task. If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--mask_time_length",
        type=int,
        default=None,
        help=(
            "Length of each vector mask span to mask along the time axis in the contrastive task."
            " If omitted, will pull value from model config."
        ),
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help=(
            "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the"
            " use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    args = parser.parse_args()
    return args


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm

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
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                i += window
    return np.array(result)



if __name__ == '__main__':

    args = get_arguments()
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir = manifest_dir / "w2v-audio-and-ecg"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "checkpoint-best"

    embedding_type = args.embedding_type
    embedding_dict = {'audio': 0, 'ecg': 1}


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    start_time = time.time()
    warnings.filterwarnings("ignore")

    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        # set up weights and biases if available
        if is_wandb_available():
            import wandb

            wandb.init(project=args.output_dir.split("/")[-1])
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

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

    raw_datasets = DatasetDict()
    raw_datasets["train"] = concatenate_datasets([train_dataset, eval_dataset]).shuffle(seed=args.seed)

    num_validation_samples = raw_datasets["train"].num_rows * args.validation_split_percentage // 100
    if num_validation_samples == 0:
        raise ValueError(
            "`args.validation_split_percentage` is less than a single sample "
            f"for {len(raw_datasets['train'])} training samples. Increase "
            "`args.num_validation_split_percentage`. "
        )

    raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
    raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    # preprocess datasets

    processor = Wav2Vec2Processor(Wav2Vec2FeatureExtractor(return_attention_mask=True), PreTrainedTokenizer())
    target_sampling_rate = 16000
    def preprocess_function(examples, processor=processor, target_sampling_rate=target_sampling_rate):
        def speech_file_to_array_fn(path, target_sampling_rate=target_sampling_rate):
            speech_array, sampling_rate = torchaudio.load(path)
            resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
            speech = resampler(speech_array).squeeze().numpy()
            return speech

        def imu_to_limu_format(path, raw_sr=150, target_sr=20):
            pd_file = pd.read_csv(path.__str__(), sep=' ', header=None)
            tmp_data = torch.tensor(pd_file.values)
            tmp_data = down_sample(tmp_data, raw_sr / target_sr, 0, tmp_data.shape[0] - 1)
            tmp_data = np.concatenate((tmp_data, np.expand_dims(tmp_data[-1], 0)), 0)[:, :6]
            tmp_data = tmp_data.reshape(tmp_data.shape[0] * tmp_data.shape[1])
            return tmp_data

        speech_list = [speech_file_to_array_fn(path) for path in examples['audio_path']]
        ecg_list = [speech_file_to_array_fn(path) for path in examples['ecg_path']]
        limu_list = [imu_to_limu_format(path) for path in examples['imu_path']]

        audio_list = []
        for i in range(len(speech_list)):
            audio_list.append(np.concatenate((np.expand_dims(speech_list[i], axis=0), \
                                              np.expand_dims(ecg_list[i], axis=0), \
                                              np.expand_dims(np.concatenate((limu_list[i], np.zeros(
                                                  speech_list[0].shape[0] - limu_list[0].shape[0]))), axis=0)), axis=0))
        result = processor(audio_list, sampling_rate=target_sampling_rate)
        return result


    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            preprocess_function,
            # remove_columns=raw_datasets['train'].colomn_names,
            batch_size=100,
            batched=True,
            num_proc=2
        )



    # config
    model_name_or_path = manifest_dir / 'pretrained_weights'

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        finetuning_task="wav2vec2_clf",
    )
    setattr(config, 'limu_pretrained_model', args.limu_pretrained_model)
    setattr(config, 'mode', args.mode)
    setattr(config, 'pretrain', True)







    # Model

    lb_audio_pretrained_weights = torch.load(args.audio_pretrained_model)
    bp_ecg_pretrained_weights = torch.load(args.ecg_pretrained_model)
    model = create_model(config=config,
                         embedding_type=embedding_dict[embedding_type],
                         lb_audio_pretrained_weights=lb_audio_pretrained_weights,
                         bp_ecg_pretrained_weights=bp_ecg_pretrained_weights,
                         pretrain=True)

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Define data collator, optimizer and scheduler

    mask_time_prob = config.mask_time_prob if args.mask_time_prob is None else args.mask_time_prob
    mask_time_length = config.mask_time_length if args.mask_time_length is None else args.mask_time_length

    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=processor.feature_extractor,
        pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )
    train_dataloader = DataLoader(
        vectorized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        pin_memory=False,
    )
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)



    # Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # forward
            outputs = model(**batch)

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather_for_metrics(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)

            # update step
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay ** completed_steps,
                    args.min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1

            # 6. Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()
                    outputs.contrastive_loss = accelerator.gather_for_metrics(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather_for_metrics(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather_for_metrics(percent_masked).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available():
                        wandb.log(train_logs)

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                # if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                #     repo.push_to_hub(
                #         commit_message=f"Training in progress step {completed_steps}",
                #         blocking=False,
                #         auto_lfs_prune=True,
                #     )

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= args.max_train_steps:
                break













    if(args.train or args.train_from_ckpt):

        model.freeze_feature_extractor()


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
        model = create_model(config=config, embedding_type=embedding_dict[embedding_type], lb_audio_pretrained_weights=lb_audio_pretrained_weights,
                                 bp_ecg_pretrained_weights=bp_ecg_pretrained_weights)
        model = model.to(device)
        model.load_state_dict(torch.load(Path(model_name_or_path) / "pytorch_model.bin"))

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