from pydub import AudioSegment, utils
import torch, torchaudio


class Predictor:

    def __init__(self, sample_rate, threshold, duration=30):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.duration = duration
    def predict(self):
        pass

class AudioEnergyPredictor(Predictor):

    def __init__(self, sample_rate, threshold, duration=30, max_amp=32767):
        super().__init__(sample_rate, threshold, duration)
        self.max_amp = max_amp

    def predict(self, audio_energy):
        silence_thresh = utils.db_to_float(self.threshold) * self.max_amp
        num_data = audio_energy.shape[0]
        res = torch.zeros(num_data)
        for i in range(num_data):
            if audio_energy[i] <= silence_thresh:
                res[i] = 1
        return res

class EcgPredictor(Predictor):

    def __init__(self, sample_rate, threshold, avg_hr, duration=30):
        super().__init__(sample_rate, threshold, duration)
        self.avg_hr = avg_hr

    def predict(self, bpm):
        num_data = bpm.shape[0]
        res = torch.zeros(num_data)
        for i in range(num_data):
            if (bpm[i] <= self.avg_hr+self.threshold):
                res[i] = 1
        return res

class ImuZPredictor(Predictor):

    def __init__(self, sample_rate, acc_z_threshold, acc_var_threshold, duration=30):
        super().__init__(sample_rate, acc_z_threshold, duration)
        self.avg_hr = avg_hr
        self.acc_z_threshold = acc_z_threshold
        self.acc_var_threshold = acc_var_threshold

    def predict(self, acc_z, acc_var):
        num_data = acc_z.shape[0]
        res = torch.zeros(num_data)
        for i in range(num_data):
            if (acc_z[i] >= self.acc_z_threshold*self.sample_rate*self.duration) and acc_var[i]<self.acc_var_threshold:
                res[i] = 1
        return res
