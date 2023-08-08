# 比較による解析 (ANAlysis by CoMParison)
# サンプルレートはすべて44100Hzとすること。

# nnAudio

__doc__ = '[IMPORTANT] You *MUST* run "initialise_model" first for initialisation of the model which will be used for performance analysis.'
from typing import Literal
from collections import deque
from warnings import catch_warnings, simplefilter

import librosa as lb
import numpy as np
import torch

from nnAudio.features.cqt import CQT1992v2
np.float = float

import fastdtw as fd

from dtconv import dc
from modelstack2 import powerconv, ToneEnDv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# モデルの設定 ピッチディテクションモデルを改良した場合はここを変えればよい
MODEL = ToneEnDv2(0).to(DEVICE)
_CQT_LAYER = CQT1992v2(44100, 441, lb.note_to_hz('F0'), n_bins = 120, pad_mode = 'constant', verbose = False).to(DEVICE)

# 予めCQTレイヤを準備しておくので、デバイス変更の際は同時に変える
def designate_device(devicename: Literal['cpu', 'cuda:0', 'cuda:1']):
    global DEVICE, _CQT_LAYER
    DEVICE = devicename
    _CQT_LAYER = CQT1992v2(44100, 441, lb.note_to_hz('F0'), n_bins = 120, pad_mode = 'constant', verbose = False).to(DEVICE)

# モデル初期化
def initialise_model(model_state_dict_path: str): MODEL.load_state_dict(torch.load(model_state_dict_path, map_location = DEVICE))
I = initialise_model

# 高速化用。librosa.cqtの代わり
def _cqt(y:np.ndarray):
    y_tensor = torch.from_numpy(y.astype('f4').copy()).to(DEVICE)
    retval = _CQT_LAYER(y_tensor).to('cpu').numpy().copy()
    return retval[0]

# 音源の読み込み。強制的に44100Hzのレートでとって、トリミングする
def perf_to_array(path:str, threshold:int = 35, start_sec:float = None, end_sec:float = None):
    """Parameters
    -------------
    path ... path to your audio file\n
    threshold ... 'maximum volume - this value' will be used\n
    as a upper threshold of noise during the trimming process\n
    start_sec, end_sec ... you can set start / end point of music in your audio file"""
    base = lb.load(path, sr = 44100)[0]
    if {start_sec, end_sec} != {None}:
        start_frame = 0 if start_sec is None else int(start_sec*44100)
        end_frame = -1 if end_sec is None else int(end_sec*44100)
        base = base[start_frame:end_frame]
    return lb.effects.trim(base, top_db = threshold, hop_length = 441)[0]
pta = perf_to_array

# 音量の抽出
def get_vol(y:np.ndarray):
    rms = lb.feature.rms(y = y, hop_length = 441, center = False)
    return lb.amplitude_to_db(rms, ref = 2e-5)[0]

# ピアノロール抽出
def extract_piano_roll(y:np.ndarray):
    baseamp = lb.amplitude_to_db(_cqt(y), ref = 2e-5)
    # lb.cqtをnnAudioベースに置換 Tensor -> ndarray -> tensor は明らかに非効率なので改善予定
    formatted = np.pad(baseamp, [[0, 12], [0, 0]])
    formatted[formatted < 0] = 0
    inp = powerconv(formatted).to(DEVICE)
    
    MODEL.eval()
    with torch.no_grad(): outp = powerconv(MODEL(inp))
    outp[outp<21] = 0
    return outp
epr = extract_piano_roll

# ノート・オン検出
def note_on_detect(y:np.ndarray):
    return lb.onset.onset_detect(y = y, sr = 44100, hop_length = 441)
nod = note_on_detect

# Derivative Dynamic Time Warping (FastDTW ver.)
def ddtw(x, y, radius = 1, dist = None):
    f'''{fd.fastdtw.__doc__}'''
    x = [(x[i+1]-x[i])/2 + (x[i+2]-x[i])/4 for i in range(len(x)-2)]; x = np.array([x[0]] + x + [x[-1]])
    y = [(y[i+1]-y[i])/2 + (y[i+2]-y[i])/4 for i in range(len(y)-2)]; y = np.array([y[0]] + y + [y[-1]])
    return fd.fastdtw(x, y, radius, dist)

# ゼロ除算防止
def _safe_div(value1, value2): return value1 / value2 if value2 else float('inf') * np.sign(value1)

# 評価 (switch文の代わり)
def _rating(conds:list, return_vals:list, exc_val = None):
    for cond, ret in zip(conds, return_vals):
        if cond: return ret
    if isinstance(exc_val, Exception): raise exc_val
    else: return exc_val

# 比較分析 (総合)
def analyse_by_comparison(perf1:np.ndarray, perf2:np.ndarray,
                          adjustment_option:Literal['max', 'min', 'average', 'none'] = None):
    # 音源1 (perf1) が自分、音源2  (perf2) が模範演奏であるとする
    # 下準備
    print('Extracting volume information...         ', end = '', flush = True)
    vol1, vol2 = get_vol(perf1), get_vol(perf2)
    aj = 0
    if adjustment_option == 'max': aj = np.max(vol1) - np.max(vol2)
    elif adjustment_option == 'min': aj = np.min(vol1) - np.min(vol2)
    elif adjustment_option == 'average': aj = np.average(vol1) - np.average(vol2)
    elif adjustment_option in {None, 'none'}: pass
    else: raise ValueError(f'invalid adjustment option: {adjustment_option}')
    vol2 += aj; perf2 *= dc(aj)
    print('done', flush = True)
    print('Extracting features from spectrograms... ', end = '', flush = True)
    y1, y2 = epr(perf1), epr(perf2); print('done', flush = True)
    print('Extracting onsets...                     ', end = '', flush = True)
    onset1, onset2 = nod(perf1), nod(perf2); print('done', flush = True)

    print('Calculating DDTW...                      ', end = '', flush = True)
    dist, ind = ddtw(y1.T, y2.T, 10) # distは不使用だが曲の類似度比較の指標になる
    reduced_ind = [ind[100*i] for i in range(len(ind) // 100)] + [ind[-1]] # 概ね一秒ごとの指標
    print('done', flush = True)

    # 音量バランスの比較 - 全体
    print('Checking volume balance of the whole...  ', end = '', flush = True)
    vave1, vave2 = np.average(vol1), np.average(vol2) # 平均
    vstd1, vstd2 = np.std(vol1), np.std(vol2) # 標準偏差 (範囲で計算すると曲間に長い休符を含む場合に不適切)
    print('done', flush = True)

    # テンポの比較: 正確に値を算出するにはある程度ユーザーの介入が必要だが、BPMの数値自体はあまり重要でないので、比を出す
    # 音源1のテンポが音源2の何倍かを各対応箇所で出す
    print('Calculating tempo ratio...               ', end = '', flush = True)
    trate_overall = len(perf2) / len(perf1) # 時間データから計算しているので逆数であることに注意
    trate_partial = [_safe_div(reduced_ind[i+1][1]-reduced_ind[i][1], reduced_ind[i+1][0]-reduced_ind[i][0]) \
                     for i in range(len(reduced_ind)-1)]
    print('done', flush = True)

    # 部分音量比較
    print('Comparing partial volume balance...      ', end = '', flush = True)
    with catch_warnings():
        simplefilter('ignore', RuntimeWarning)
        vpo1 = np.array([np.average(vol1[reduced_ind[i][0]:reduced_ind[i+1][0]]) for i in range(len(reduced_ind)-1)])
        vpo2 = np.array([np.average(vol2[reduced_ind[i][1]:reduced_ind[i+1][1]]) for i in range(len(reduced_ind)-1)])
        vpo_rate = [_rating([item<-3, -3<=item and item<-1.5, 1.5<item and item<=3, 3<item], \
                            [-2, -1, 1, 2], 0) for item in vpo1 - vpo2] # -2から、極小、小、適、大、極大
        
        foldedspec1 = np.array([[np.average(y1[8*i:8*i+8, reduced_ind[j][0]:reduced_ind[j+1][0]]) \
                                for i in range(11)] for j in range(len(reduced_ind)-1)]) # 厳密に計算するなら位相も要る
        foldedspec1[np.isnan(foldedspec1)] = 0
        foldedspec2 = np.array([[np.average(y2[8*i:8*i+8, reduced_ind[j][1]:reduced_ind[j+1][1]]) \
                                for i in range(11)] for j in range(len(reduced_ind)-1)])
        foldedspec2[np.isnan(foldedspec2)] = 0
        foldedspec_rate = np.array([[_rating([item<-3, -3<=item and item<-1.5, 1.5<item and item<=3, 3<item], \
                            [-2, -1, 1, 2], 0) for item in _fr] for _fr in foldedspec1-foldedspec2])
    print('done', flush = True)
    
    # もつれ度比較
    # 150ms (15フレーム) 以下の長さで演奏される音の連なりの乱れを検出
    # F. Chopin: Fantaisie-Impromptu みたいなポリリズムには弱く、古典派やベートーヴェン向きの機能
    # 正確な音の位置、ヴェロシティ検出ができると汎用性向上
    
    print('Calculating tottering indicators...      ', end = '', flush = True)
    gap1 = [onset1[i+1]-onset1[i] for i in range(len(onset1)-1)]
    gap2 = [onset2[i+1]-onset2[i] for i in range(len(onset2)-1)]

    tmpval = 0
    gaprate1, gaprate2 = deque(), deque()
    for item in gap1:
        if item < 16 and tmpval:
            tmp = item/tmpval
            if tmp < 2**-.5: tmp *= 2 # 一応70msと140msで弾き分けが利かないこともないので倍の長さに対応
            elif tmp > 2**.5: tmp /= 2
            gaprate1.append(tmp*100)
        tmpval = item

    tmpval = 0
    for item in gap2:
        if item < 16 and tmpval:
            tmp = item/tmpval
            if tmp < 2**-.5: tmp *= 2
            elif tmp > 2**.5: tmp /= 2
            gaprate2.append(tmp*100)
        tmpval = item

    randv1, randv2 = np.std(gaprate1), np.std(gaprate2)
    if np.isnan(randv1): randv1 = 0
    if np.isnan(randv2): randv2 = 0

    print('done', flush = True)

    retdict = {
        'perf1': {
            'sndimg': y1,
            'vo_ave': vave1,
            'vo_std': vstd1,
            'rndnes': randv1
        },
        'perf2': {
            'sndimg': y2,
            'vo_ave': vave2,
            'vo_std': vstd2,
            'rndnes': randv2
        },
        'misc': {
            'mindex': reduced_ind,
            'vprate': vpo_rate,
            'fsrate': foldedspec_rate,
            'torate': trate_overall,
            'tprate': trate_partial
        }
    }
    return retdict
anacmp = analyse_by_comparison