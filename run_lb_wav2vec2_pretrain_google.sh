python3 lb_wav2vec2_pretrain.py --base_dir /mnt/disks/lb-dataset/workspace/LittleBeatsPrelim \
                                --ckpt_path /mnt/disks/lb-dataset/workspace/LittleBeatsPrelim/manifest/w2v-audio-and-ecg \
                                --cache_path /mnt/disks/cache-disk/.cache/huggingface/datasets \
                                --audio_pretrained_model /mnt/disks/lb-dataset/workspace/LittleBeatsPrelim/manifest_pretrain/pretrained_weights/lb_lena_4300hr.pt \
                                --ecg_pretrained_model /mnt/disks/lb-dataset/workspace/LittleBeatsPrelim/manifest_pretrain/pretrained_weights/bp_ecg_500hr.pt \
                                --limu_pretrained_model /mnt/disks/lb-dataset/workspace/LittleBeatsPrelim/manifest_pretrain/pretrained_weights/limu/limu_v3_separate4.pt