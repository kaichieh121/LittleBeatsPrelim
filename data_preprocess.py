import argparse
import torch, torchaudio
from pathlib import Path
from sleep_classifier_hal import align_data
from synchronize import create_chunks

def get_arguments():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--annotation_dir', default='/home/kcchang3/data/LittleBeats/LittleBeats_Sleep annotations/RA sleep annotations completed (for Jeff)')
    parser.add_argument('--data_dir', default='/home/kcchang3/data/LittleBeats/sleep_study_preliminary_recordings')
    parser.add_argument('--output_dir', default='/home/kcchang3/data/LittleBeats/data_30s')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    return args

def main():
    '''
        output_dir
            -audio
                -audio_000001.wav
                -audio_000002.wav
            -ecg
                -ecg_000001.wav
                -ecg_000002.wav
            -accz
                -?
            -avg_hr.csv
            -label.csv (0 non-sleep, 1 sleep)

    '''
    args = get_arguments()

    annotation_dir = Path(args.annotation_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_file = open(output_dir / 'label.csv', mode='w')
    idx = 0
    for dir in annotation_dir.iterdir():
        if dir.is_dir() and not (dir.name == "No Sleep files") and not ('Processed ECG' in dir.name):
            dir_name = dir.name
            orig_audio_folder = data_dir / dir_name / "Audio_cleaned"
            orig_ecg_folder = data_dir / dir_name / "ECG_cleaned"
            orig_imu_folder = data_dir / dir_name / "IMU_cleaned"
            for file in orig_ecg_folder.iterdir():
                if (file.match('*cleaned.wav')):
                    ecg_file = file
                if ('timestamp' in file.name):
                    ecg_timestamp_file = file
            for file in orig_imu_folder.iterdir():
                if (file.match('*cleaned.txt')):
                    imu_file = file
                if ('timestamp' in file.name):
                    imu_timestamp_file = file
            # avg_hr = read_avg_hr(dir)
            for file in dir.iterdir():
                if (file.match('*.wav')):
                    num = file.name.split('cleaned_')[1].split("_")[0]
                    for f in orig_audio_folder.iterdir():
                        if(('_Audio_timestamps_' + num + ".txt") in f.name):
                            audio_timestamp_file = f
                    for f in dir.iterdir():
                        if(f'cleaned_{num}' in f.name and "TextGrid" in f.name):
                            audio_textgrid_file = f

                    audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, y = align_data(imu_file, imu_timestamp_file, ecg_file, ecg_timestamp_file, file, audio_timestamp_file, audio_textgrid_file, interval=args.interval)
                    idx = create_chunks(audio_x, audio_sr, ecg_x, ecg_sr, imu_data, imu_sr, y, output_dir, label_file, idx)
                    print(f'{file.name}')
    label_file.close()
if __name__ == '__main__':
    main()