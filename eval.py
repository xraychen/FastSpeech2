import os
import sys
import librosa
import numpy as np

import torch
from s3prl.hub import mos_wav2vec2, mos_tera, mos_apc
from resemblyzer import preprocess_wav, VoiceEncoder

from tqdm import tqdm


class EvalAgent():
    def __init__(self, root_dir):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.root_dir = root_dir
        self.encoder = VoiceEncoder(verbose=False)
        self.mos_predictor = mos_apc().to(self.device)

    def asv_score(self, dir_01, dir_02):
        files = sorted(os.listdir(os.path.join(self.root_dir, dir_01)))

        embed = []
        for f in files:
            ref_01 = preprocess_wav(os.path.join(self.root_dir, dir_01, f))
            ref_02 = preprocess_wav(os.path.join(self.root_dir, dir_02, f))

            embed_01 = self.encoder.embed_utterance(ref_01)
            embed_02 = self.encoder.embed_utterance(ref_02)

            embed.append((embed_01, embed_02))

        scores = [np.dot(e[0], e[1]) for e in embed]

        return np.mean(scores), np.std(scores)

    def asv_score_s(self, dir_01, ref_audio):
        files = sorted(os.listdir(os.path.join(self.root_dir, dir_01)))
        files = [f for f in files if '.wav' in f]

        embed = []
        for f in tqdm(files):
            ref_01 = preprocess_wav(os.path.join(self.root_dir, dir_01, f))
            ref_02 = preprocess_wav(ref_audio)

            embed_01 = self.encoder.embed_utterance(ref_01)
            embed_02 = self.encoder.embed_utterance(ref_02)

            embed.append((embed_01, embed_02))

        scores = [np.dot(e[0], e[1]) for e in embed]

        return np.mean(scores), np.std(scores)

    def mos_score(self, dir_01):
        files = sorted(os.listdir(os.path.join(self.root_dir, dir_01)))
        files = [f for f in files if '.wav' in f]

        wavs = []
        for f in tqdm(files):
            wav, sr = librosa.load(os.path.join(self.root_dir, dir_01, f), sr=None)
            wav = torch.FloatTensor(wav).to(self.device)
            wavs.append(wav)

        self.mos_predictor.eval()
        with torch.no_grad():
            scores = self.mos_predictor(wavs)['scores'].detach().cpu().numpy()

        return np.mean(scores), np.std(scores)


def eval_all(root_dir):
    ref_audio = 'dataset/M2VoC/TST-Track1-S5-male-Story-100/wavs/000002.wav'
    test_dirs = sorted(os.listdir(root_dir))

    agent = EvalAgent(root_dir=root_dir)
    for test_dir in test_dirs:
        asv_score = agent.asv_score_s(test_dir, ref_audio)
        mos_score = agent.mos_score(test_dir)

        print(f'[ref = {ref_audio}, test_dir = {test_dir:>20}] asv = {asv_score[0]:.4f} +- {asv_score[1]:.4f}, mos = {mos_score[0]:.4f} += {mos_score[1]:.4f}')
        # print(f'[test_dir = {test_dir:>20}] mos = {mos_score[0]:.4f} += {mos_score[1]:.4f}')


if __name__ == '__main__':
    eval_all(sys.argv[1])
