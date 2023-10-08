# Classification of Infant Sleep/Wake States

<p align="center">
        <img src="figures/LB_model.jpg" title="LittleBeats Model" width="48%"> 
        <img src="figures/cross_attention.jpg" title="Cross Attention" width="43.8%">
        <em>The LittleBeats Model Architecture (left) and Cross-attention for the audio branch (right).</em>
</p>

> K. C. Chang, M. Hasegawa-Johnson, N. L. McElwain, and B. Islam, Classification of Infant Sleep/Wake States: Cross-Attention among Large Scale Pretrained Transformer Networks using Audio, ECG, and IMU Data. 2023. [[arXiv]](https://arxiv.org/abs/2306.15808) [IEEE Eplore]

Accepted by 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) 

This project code is based on [![](https://img.shields.io/badge/HuggingFace-wav2vec2-blue)](https://github.com/huggingface/transformers/blob/v4.32.1/src/transformers/models/wav2vec2/modeling_wav2vec2.py) and [![](https://img.shields.io/badge/Github-LIMU_BERT-blue)](https://github.com/dapowan/LIMU-BERT-Public)

The code and instruction are constantly being cleaned and updated. Please contact Kai Chieh Chang (kcchang3@illinois.edu) for any questions.

## Infant Sleep/Wake Classification
Infant sleep is critical to brain and behavioral development. 
Prior studies on infant sleep/wake classification have been largely limited to reliance on expensive and burdensome 
polysomnography (PSG) tests in the laboratory or wearable devices that collect single-modality data. 
To facilitate data collection and accuracy of detection, we aimed to advance this field of study by using a 
multi-modal wearable device, LittleBeats (LB), to collect audio, electrocardiogram (ECG), and inertial 
measurement unit (IMU) data among a cohort of 28 infants. We employed a 3-branch (audio/ECG/IMU) large scale 
transformer-based neural network (NN) to demonstrate the potential of such multi-modal data. We pretrained 
each branch independently with its respective modality, then finetuned the model by fusing the pretrained 
transformer layers with cross-attention. We show that multi-modal data significantly improves sleep/wake 
classification (accuracy = 0.880), compared with use of a single modality (accuracy = 0.732). 
Our approach to multi-modal mid-level fusion may be adaptable to a diverse range of architectures and tasks, 
expanding future directions of infant behavioral research.

## Dataset
Due to privacy, we are unable to provide the dataset used for this paper. Please prepare:
1. synchronized audio (sampling frequency 16000Hz), ecg (sampling frequency 16000Hz), and imu
data (sampling frequency 150Hz, space separated .csv file with 6 entries (acceleration x,y,z + gyroscope x,y,z) per row)
2. manifest/train.csv, manifest/val.csv, manifest/test.csv (each follow the same format as provided manifest/sample.csv)

## Pretraining Audio or ECG branch
1. Followed the pretraining procedure for **wav2vec2_base_librispeech** released by [Facebook](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) for both audio and ecg. The configuration used by us is:
```angular2html
srun --gres=gpu:4 --ntasks=1 fairseq-hydra-train \
    task.data=${MANIFEST_DIR} \
    checkpoint.save_dir=${save_dir} hydra.run.dir=${save_dir} \
    common.fp16=True \
    distributed_training.distributed_world_size=4 \
    +optimization.update_freq='[16]' \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```
2. Or download pretrained weights for [audio](https://drive.google.com/file/d/1bJH7TJ6VGdwYKr6g39lBFJ-r2zTfSkXW/view?usp=drive_link) and [ECG](https://drive.google.com/file/d/1Hg64bSl3tok8zIwJlmruDKruzjqZxPe1/view?usp=drive_link)
3. Copy the weights into manifest/pretrained_weights

## Pretraining IMU branch
This is modified from [![](https://img.shields.io/badge/Github-LIMU_BERT-blue)](https://github.com/dapowan/LIMU-BERT-Public)
1. Prepare dataset/littlebeats_pretrain
   1. we used the default sampling rate of 20Hz and sequence length 120 (6 seconds)
   2. place a "data_20_120.npy" ndarray numpy object with shape (num_data, 120, 6) in dataset/littlebeats_pretrain
   3. place a "label_20_120.npy" ndarray numpy object with shape (num_data, 120, 2) in dataset/littlebeats_pretrain
2. Run the following pretraining command
```
python limu_bert/pretrain.py v4 littlebeats_pretrain 20_120 -s limu_v4_sep12
```
3. Or download pretrained weights for [IMU](https://drive.google.com/file/d/1kP3XqyjM8GRCoHolXIMFVKUuTAXMZwnM/view?usp=drive_link)
4. Copy limu_v4_sep12.pt into manifest/pretrained_weights/limu

## Prepare Environment

This code is tested on Windows 11 with Python 3.9, CUDA 11.6 and Pytorch 1.13


## Training from Scratch or from Checkpoint
```
sleep_classifier_w2v_audio_ecg.py 
--train or --train_from_ckpt
--ckpt_path
"path to checkpoint directory"
--cache_path
"D:\Projects\datasets\.cache\huggingface\datasets"
--embedding_type
"audio"
--audio_pretrained_model
"path to pretrained audio wav2vec2"
--ecg_pretrained_model
"path to pretrained ecg wav2vec2"
--limu_pretrained_model
"path to limu_v4_sep12.pt"
--mode
"triple"
```

## Evaluation
```
sleep_classifier_w2v_audio_ecg.py 
--eval
--ckpt_path
"path to checkpoint directory"
--cache_path
"D:\Projects\datasets\.cache\huggingface\datasets"
--embedding_type
"audio"
--audio_pretrained_model
"path to pretrained audio wav2vec2"
--ecg_pretrained_model
"path to pretrained ecg wav2vec2"
--limu_pretrained_model
"path to limu_v4_sep12.pt"
--mode
"triple"
```

## Official Checkpoint
1. Download [checkpoint](https://drive.google.com/drive/folders/1jWmoQnGA1KASlIPCft76YFpGLRGm3WCv?usp=drive_link) directory
2. Run evaluation using the downloaded path as --ckpt_path