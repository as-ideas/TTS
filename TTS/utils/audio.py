import librosa
import soundfile as sf
import numpy as np
import scipy.io.wavfile
import scipy.signal
import pyworld as pw

from TTS.tts.utils.data import StandardScaler

#pylint: disable=too-many-public-methods
from TTS.utils.io import load_config
from TTS.vocoder.datasets.preprocess import load_wav_data


class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 fft_size=1024,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 spec_gain=20,
                 stft_pad_mode='reflect',
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 do_sound_norm=False,
                 stats_path=None,
                 **_):

        print(" > Setting up Audio Processor...")
        # setup class attributed
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.stats_path = stats_path
        # setup stft parameters
        if hop_length is None:
            # compute stft parameters from given time values
            self.hop_length, self.win_length = self._stft_parameters()
        else:
            # use stft parameters from config file
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        assert self.win_length <= self.fft_size, " [!] win_length cannot be larger than fft_size"
        members = vars(self)
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))
        # create spectrogram utils
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        # setup scaler
        if stats_path:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_stats(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None

    ### setting up the parameters ###
    def _build_mel_basis(self, ):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.fft_size,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def normalize(self, S):
        S = np.clip(S, a_min=1.e-5, a_max=None)
        return np.log(S)

    def denormalize(self, S):
        return np.exp(S)

    def melspectrogram(self, y):
        D = self.stft(y)
        S = self.linear_to_mel(np.abs(D))
        return self.normalize(S)

    def inv_melspectrogram(self, mel, n_iter=32):
        """Uses Griffin-Lim phase reconstruction to convert from a normalized
        mel spectrogram back into a waveform."""
        denormalized = self.denormalize(mel)
        S = librosa.feature.inverse.mel_to_stft(
            denormalized, power=1, sr=self.sample_rate,
            n_fft=self.fft_size, fmin=self.mel_fmin, fmax=self.mel_fmax)
        wav = librosa.core.griffinlim(
            S, n_iter=n_iter,
            hop_length=self.hop_length, win_length=self.win_length)
        return wav

    def raw_melspec(self, y):
        D = self.stft(y)
        S = self.linear_to_mel(np.abs(D))
        return S

    def stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.fft_size, hop_length=self.hop_length, win_length=self.win_length)

    def linear_to_mel(self, spectrogram):
        return librosa.feature.melspectrogram(
            S=spectrogram, sr=self.sample_rate, n_fft=self.fft_size, n_mels=self.num_mels, fmin=self.mel_fmin, fmax=self.mel_fmax)

    def compute_stft_paddings(self, x, pad_sides=1):
        '''compute right padding (final frame) or both sides padding (first and final frames)
        '''
        assert pad_sides in (1, 2)
        pad = (x.shape[0] // self.hop_length + 1) * self.hop_length - x.shape[0]
        if pad_sides == 1:
            return 0, pad
        return pad // 2, pad // 2 + pad % 2

    ### Compute F0 ###
    def compute_f0(self, x):
        f0, t = pw.dio(
            x.astype(np.double),
            fs=self.sample_rate,
            f0_ceil=self.mel_fmax,
            frame_period=1000 * self.hop_length / self.sample_rate,
        )
        f0 = pw.stonemask(x.astype(np.double), f0, t, self.sample_rate)
        return f0

    ### Audio Processing ###
    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.01 sec margin """
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=self.trim_db, frame_length=self.win_length, hop_length=self.hop_length)[0]

    @staticmethod
    def sound_norm(x):
        return x / abs(x).max() * 0.9

    ### save and load ###
    def load_wav(self, filename, sr=None):
        x, sr = librosa.load(filename, sr=None)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f' [!] File cannot be trimmed for silence - {filename}')
        assert self.sample_rate == sr, "%s vs %s"%(self.sample_rate, sr)
        if self.do_sound_norm:
            x = self.sound_norm(x)
        return x

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2 ** qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal,)

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2 ** qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x


    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1


import torch
if __name__ == '__main__':
    print('hallo')
    cfg = load_config('/Users/cschaefe/workspace/MozillaTTS/TTS/tts/configs/config.json')
    ap = AudioProcessor(**cfg.audio)
    wav = ap.load_wav('/Users/cschaefe/datasets/audio_data/Cutted_merged_resampled/01452.wav')
    mel = ap.melspectrogram(wav)
    wav_r = ap.inv_melspectrogram(mel)
    print(f'mel shape {mel.shape}')
    mel_torch = torch.tensor(mel).float().unsqueeze(0)
    torch.save(mel_torch, '/tmp/mozilla/sample.mel')
    ap.save_wav(wav_r, '/tmp/mozilla/sample.wav')
