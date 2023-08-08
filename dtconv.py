# 音源 to pre-dataset
# (batch数を固定するとGPU / TPU がメモリ切れするので、dataset化はそこで)
# 鳴らすタイミング、音量操作を行いつつ、データ・ラベルを同時作成
# シェルに書き込んで呼び出す関数の形で表すこととする

from warnings import warn
try: import memo_secret as msc
except ModuleNotFoundError:
    warn('Cannot find memo_secret.py. you cannot make datasets, but you still can use some functions.', ImportWarning, 2)
    class msc: O = S = L = './'

import numpy as np
import librosa as lb
from matplotlib.pyplot import plot, close, show
from pydub.playback import play
from pydub import AudioSegment as au
from pickle import dump, load

# numデシベル倍の音量 (rms) を音源 (y) に適用する際掛ける値の算出
def dc(num: int) -> float: return 10 ** (.05 * num)

# パワー計算 (ショートカット)
def get_pow(y):
    retval = lb.amplitude_to_db(lb.feature.rms(y = y, hop_length = 441, center = False), ref = 2e-5)
    # rmsの際frame分縮むが、padはしない (重要) → 後で説明するところでindex取得の際無音が挟まれているとその分ずれる
    retval[retval < 0] = 0
    return retval.astype('u1')[0]

# 番号→音名
def num_to_name(num: int):
    num += 9
    alpha = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
    return alpha[(num) % 12] + str(num // 12)

# 音源ファイル、音量、始点終点を整形 (データセット作りの前処理の前処理)
def audiodata_prep():
    sr = 44100
    
    longs = [lb.load(item, sr = sr)[0] for item in [msc.L + num_to_name(i) + '.m4a' for i in range(88)]]
    shrts = [lb.load(item, sr = sr)[0] for item in \
             [msc.S + num_to_name(i) + '.m4a' for i in range(16)] + [msc.S + 'Cs2.wav'] + \
             [msc.S + num_to_name(i) + '.m4a' for i in range(17, 67)]] + longs[67:88]
    # ※家のピアノはF6以降ダンパーがついていないのでこのようになっている。
    # ※C#2の弦は丁度切ってしまったので、D2から生成した (lb.effects.pitch_shift())

    lloud = [get_pow(item) for item in longs]
    sloud = [get_pow(item) for item in shrts]
    lonid = [np.argmax(item) for item in lloud] # ノートオン
    sonid = [np.argmax(item) for item in sloud]

    lloud = [lloud[i][lonid[i]:] for i in range(88)] # 始点調節
    sloud = [sloud[i][sonid[i]:] for i in range(88)]
    longs = [longs[i][lonid[i]*441:] for i in range(88)]
    longs = [longs[i][:len(lloud[i])*441] for i in range(88)]
    shrts = [shrts[i][sonid[i]*441:] for i in range(88)]
    shrts = [shrts[i][:len(sloud[i])*441] for i in range(88)]

    lsg, ssg = [None for _ in range(88)], [None for _ in range(88)] # 始点 (1) と終点 (-1) が記された列が入る

    # 長い音 → G5までグラフで判断、残りは40dB未満になった地点がノートオフ
    # 短い音 → 一律50ミリ秒 (演奏時多少残響があっても聞き分けられるため。長さはCh.V.Alkanの曲等を参考に連符の限界長を考慮して決定)

    # まずは長い音
    for i in range(88):
        target = longs[i]
        x = list(range(100, min(361, len(lloud[i]))))
        y = [40] * len(x)
        z = lloud[i][100:min(361, len(lloud[i]))]
        plot(x, y)
        plot(x, z)
        show(block = False)
        while True:
            try:
                _ind = int(input('index (100-360) -> '))
                if not 100 <= _ind <= 360: raise Exception()
            except: print('Enter the number of 100-360.'); continue
            _sound = (target[:_ind*441] * 32768).astype('i2').tobytes()
            play(au(_sound, sample_width = 2, frame_rate = 44100, channels = 1))
            if input('OK? y/n >>>  ') == 'y':
                lsg[i] = np.array([1] + [0] * (_ind-2) + [-1] + [0] * (len(target)//441-_ind), dtype = 'i1'); close(); break
            else: continue
    
    # スタッカート
    for i in range(67): ssg[i] = np.array([1, 0, 0, 0, 0, -1] + [0] * (len(shrts[i])//441-6), dtype = 'i1') # 50ミリ秒
    for i in range(67, 88): ssg[i] = lsg[i].copy() # ダンパーが無いので。

    tempdata = dict()
    tempdata['sound'] = list(zip(longs, shrts))
    tempdata['st_and_ed'] = list(zip(lsg, ssg))
    tempdata['loudness'] = list(zip(lloud, sloud))
    with open('data/sound.pkl', 'wb') as f: dump(tempdata, f)

ap = audiodata_prep

# ベースになる音のパス
bpaths = [msc.O + 'data/' + name + '.wav' for name in \
    {'blank', 'blue_noise', 'brownian_noise', 'pink_noise', 'violet_noise', 'white_noise'}]
# ベースになる音
try: base = [lb.load(item, sr = 44100)[0] for item in bpaths]
except: base = []

# 評価用
def where(ev):
    try: return np.where(ev)[0][0]
    except IndexError: return float('inf')

#データセット1つ分
def _one_data_create(sd, ld, se, pl = False, print_index:int = None):
    if print_index is not None: print(f'\r\033[KProcessing... {str(print_index).zfill(5)}', end = '')
    sr = 44100
    stp = np.random.randint(342*sr)
    # ベース音のカットと音量調整
    b = base[np.random.randint(6)][stp:stp+10*sr] * dc(np.random.randint(-10, 11))
    v = np.zeros((88, 1000))
    o = np.zeros((88, 1000))
    # 音の選定
    notes = [i for i in range(88)]
    np.random.shuffle(notes)
    notes = notes[:np.random.randint(5, 89)]
    # 音密度 (一つの音が4秒程度なので10秒のファイルには同じ音が最大3つ (途中まで) しか入らない)
    density = np.random.randint(1, 2)

    for ind in notes:
        start = np.random.randint(1000//density) # 始点
        ll, sl = len(ld[ind][0]), len(ld[ind][1])
        while start < 1000:
            notetype = np.random.binomial(1, .5)
            tvc = np.random.randint(-19, 1) # 音量調整
            if notetype: # スタッカートの場合
                endp = min(start+sl, 1000)
                b[start*441:endp*441] += sd[ind][1][:(endp-start)*441] * dc(tvc)
                plusl = ld[ind][1][:endp-start] + tvc
                v[ind][start:endp] = plusl
                op = se[ind][1][:endp-start]
                c1 = where(op == -1)
                c2 = where(plusl < 34)
                if c1 > c2: op = np.array([1] + [0] * (c2-1) + [-1] + [0] * (len(op) - c2), dtype = 'i1')[:endp-start]
                o[ind][start:endp] = op
                start += np.random.randint(sl, 1000//density)
            else: # レガートの場合
                endp = min(start+ll, 1000)
                b[start*441:endp*441] += sd[ind][0][:(endp-start)*441] * dc(tvc)
                plusl = ld[ind][0][:endp-start] + tvc
                v[ind][start:endp] = plusl
                op = se[ind][0][:endp-start]
                c1 = where(op == -1)
                c2 = where(plusl < 34)
                if c1 > c2: op = np.array([1] + [0] * (c2-1) + [-1] + [0] * (len(op) - c2), dtype = 'i1')[:endp-start]
                o[ind][start:endp] = op
                start += np.random.randint(ll, 1000//density)
    if pl: play(au((b*32768).astype('i2').tobytes(), sample_width = 2, frame_rate = 44100, channels = 1))

    # 短い音で定Q変換したほうが速い
    c = np.pad(lb.amplitude_to_db(
        np.abs(
        lb.cqt(b, sr = 44100, hop_length = 441, fmin = lb.note_to_hz('F0'), n_bins=120, \
               res_type = 'kaiser_fast')), ref = 2e-5)[:, :-1],
            [[0, 12], [0, 0]]
    )
    c[c < 0] = 0
    c = c.astype('u1') # 軽量化
    v[v < 0] = 0
    v = v.astype('u1') # どこかでfloatになっていたので修正
    o = o.astype('i1')
    return c, v, o


def veloc_dataset_create(datnum:int = 1000): # 一度に10秒×1000データつくる。datnumで調整可能 (大きすぎるとメモリエラーになるので注意)
    with open('data/sound.pkl', 'rb') as f: file = load(f)
    sound, st_and_ed, loudness = file['sound'], file['st_and_ed'], file['loudness']

    return [_one_data_create(sound, loudness, st_and_ed, print_index = i+1) for i in range(datnum)]

vdc = veloc_dataset_create

# 使い方の例: インタラクティブシェルで
# 
# from dtconv import vdc
# from multiprocessing import Pool # cqtするサイズが小さいので古めのPCでも並列処理である程度高速化可能。
# # 但しProcessingの進捗の番号は乱れる。
# # multiprocessing.Value等用いれば整いますが、ここに記すには冗長なので改良はお任せいたします
# from itertools import chain
# pl = Pool(プロセス数) # タスクマネージャと速さ、CPU使用率を見て決定
# res = pl.map(vdc, iter([1プロセスあたり生成数] * プロセス数))
# pre_dataset = list(chain.from_iterable(res)) # これで「1プロセスあたり生成数 * プロセス数」個分のデータになる。
#
# あとはpickle.dumpするなりそのまま使うなりして活用。
# .pyファイルを作って実行する場合はPool前にif __name__ == '__main__': が必要。