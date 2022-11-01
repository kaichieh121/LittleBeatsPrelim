from scipy.io.wavfile import read,write
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import glob

PLOT = False

target_wav_rate = 24000

def read_timestamps(file):
    """
    This function reads single timestamps.txt file for each wav file
    outputs utc/wav time lists, line fit function between those two time sequence is available
    input args:
    file(str): timestamp.txt file name
    output:
    utc_times_list(list): UTC timestamps sequence with offset taken into account, starting from 0
    wav_times_list(list): wav file timestamps sequence
    """
    f=open(file,"r")
    content=f.readlines()
    f.close()
    utc_times_list=[]
    wav_times_list=[]

    offset=-1
    for i in range(0,len(content)):
        row=content[i]
        row=row.strip("\n").split()
        utc_times = datetime.datetime.utcfromtimestamp(float(row[0])).strftime("%H:%M:%S").split(":")
        utc_time = float(utc_times[0])*3600+float(utc_times[1])*60+float(utc_times[2])
        if offset<0: offset=utc_time
        wav_time = float(row[1])/target_wav_rate
        utc_times_list.append(utc_time-offset)
        wav_times_list.append(wav_time)
        if i == len(content)-1:
            utc_times = datetime.datetime.utcfromtimestamp(float(row[2])).strftime("%H:%M:%S").split(":")
            utc_time = float(utc_times[0])*3600+float(utc_times[1])*60+float(utc_times[2])            
            wav_time = float(row[3])/target_wav_rate
            utc_times_list.append(utc_time-offset)
            wav_times_list.append(wav_time)

    # time_points=np.asarray(list(range(len(utc_times_list))),dtype=np.int64)
    # z_utc = np.polyfit(time_points,utc_times_list,1)
    # z_wav = np.polyfit(time_points,wav_times_list,1)
    # p_utc = np.poly1d(z_utc)
    # p_wav = np.poly1d(z_wav)
    # est_utc=p_utc(time_points)
    # est_wav=p_wav(time_points)
    # ratio = z_wav[0]/z_utc[0]
    # b_diff = z_utc[1]-z_wav[1]
    print("utc lengths in seconds:", utc_times_list[-1])
    print("wav lengths in seconds:", wav_times_list[-1])
    print("zero inserted in mins:", (utc_times_list[-1]-wav_times_list[-1])/60)

    #utc vs wav time plot index
    if PLOT:
        # est_utc_to_wav = est_utc*ratio - b_diff
        # print("residual utc",np.mean(np.square(est_utc-utc_times_list)))
        # print("residual wav",np.mean(np.square(est_wav-wav_times_list)))
        # print("residual est utc to wav",np.mean(np.square(est_utc_to_wav-est_wav)))

        plt.plot(utc_times_list,'r',label="utc time")
        # plt.plot(est_utc,'g',label="utc est time")
        plt.plot(wav_times_list,'b',label='wav time')
        # plt.plot(est_wav,'y',label="wav est time")
        plt.legend()
        plt.savefig("timestamp_example.png")
        plt.close()
    return utc_times_list,wav_times_list


def insert_samples(audio_wav,utc_times_list,wav_times_list,output_filename=None,apply_ratio=False):
    """
    This function inserts zeros to the LB wav file
    outputs wav files that have zeros inserted
    input args:
    audio_wav(str): LB audio file name
    utc_times_list: utc time sequence computed from read_timestamps function
    wav_times_list: wav time sequence computed from read_timestamps function
    output_filename(str): specify output file name
    apply_ratio: experimental condition, not usable. Please set it to False
    output:
    outputs are for statistical usage. If don't want it, simply ignore it.
    zeros_count(int): number of zeros inserted
    len(audio_realign)(int): number of samples for inserted wav files
    zeros_hist(list): histograms of zeros inserted

    """
    target_wav_rate,wav_data= read(audio_wav)

    audio_realign=[]
    zeros_count=0
    zeros_hist=[]
    for i in range(1,len(utc_times_list)):
        start_idx,end_idx = int(utc_times_list[i-1]*target_wav_rate),int(utc_times_list[i]*target_wav_rate)
        start_wav_idx,end_wav_idx = int(wav_times_list[i-1]*target_wav_rate),int(wav_times_list[i]*target_wav_rate)
        #insert zeros to match up utc time
        wav_dur=end_wav_idx-start_wav_idx
        utc_dur=end_idx-start_idx
        if utc_dur<=wav_dur:
            audio_realign.extend(wav_data[start_wav_idx:min(start_wav_idx+utc_dur,len(wav_data))])
            continue
        new_content=np.zeros((end_idx-start_idx))
        if apply_ratio:
            # scale by ratio
            new_idxes = np.asarray(list(range(0,len(new_content))))
            new_idxes = np.asarray(new_idxes*ratio+start_wav_idx,dtype=np.int64)
            new_idxes = new_idxes[np.where(new_idxes<len(wav_data)-1)]
            new_content[:len(new_idxes)]=wav_data[new_idxes]
        else:
            #no multiply by ratio
            audio_length= min(wav_dur,len(wav_data)-start_wav_idx)
            new_content[:audio_length]=wav_data[start_wav_idx:start_wav_idx+audio_length]

        zeros_hist.append((len(new_content)-wav_dur)/target_wav_rate)
        zeros_count+=max(0,len(new_content)-wav_dur) # TODO: minus audio_length?
        audio_realign.extend(new_content)
    # write(output_filename,target_wav_rate,np.asarray(audio_realign))
    return zeros_count,len(audio_realign),zeros_hist, audio_realign