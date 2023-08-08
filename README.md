# CTSyMP-1
Computer Tutoring System for Musical Practice (some unnecessary files has been deleted)  
楽曲音源の比較を通じた評価 / 数値データを提供します。(Tutorといいつつ細かいコツの指導まではしません。ご了承ください。)  
まだまだ高精度とは言い切れませんが、改善を目指します。  

## ファイル
※性能は最新の大学や研究機関の各分野の研究の方が勝ると思われますので、一部 (そちらが利用されることを想定して) 他の方の利用をあまり考慮していない「使いづらいファイル」がございます。その点についてご理解いただいた上でご利用いただけますと幸いです。  

data/ 中の諸ファイル  
→モデル情報が入っています。.pthファイルがpytorchの`state_dict`、.trlファイルが学習記録です。後者は`pickle.load`で読み込めます。  

anacmp.py 比較による演奏分析を行います。  
①`initialise_model(.pthファイルのパス)` (※ショートカット: `I`) で推論モデルを使用可能にし、  
②`perf_to_array(音源パス, ノイズが最大音より何dB小さいか, 始点, 終点)` (※ショートカット: `pta`) で音源を1つずつndarrayに変換して、  
③`analyse_by_comparison(演奏1, 演奏2, オプション)` (※ショートカット: `anacmp`) で色々な情報を抽出します。  
得られる情報は以下の通りです。辞書ファイル`retdict`に格納されて出力されます。尚、配列形式で返される値は全て1フレーム10ミリ秒です。  

■`retdict['perf1']` / `retdict['perf2]`: 各演奏の情報。以下`perf`とする  
・`perf['sndimg']`: `librosa.display.specshow` でプロット可能なスペクトログラム解析結果 (88×フレーム数)  
・`perf['vo_ave']`: 平均音量 (スカラー)  
・`perf['vo_std']`: 音量の標準偏差 (スカラー)  
・`perf['rndnes']`: 15ミリ秒以下の音同士の間隔の「不安定」さの指標。間隔の比の100倍の標準偏差 (スカラー)  
■`retdict['misc']`: その他 (2音源の関係が格納されている) 。以下`misc`とする  
・`misc['mindex']`: スペクトログラム解析結果2つをDDTWで比較して対応する箇所をとったもののインデックス。(記録したインデックス数×2)  
・`misc['vprate']`: 区間ごとの演奏2に対する演奏1の音量。-2から2の整数表現で、順に「静かすぎ」「静か」「適切」「うるさい」「うるさすぎ」の5段階。 (記録したインデックス数-1)  
・`misc['fsrate']`: 音高グループ (鍵盤8つ分を1グループとして、11グループ) を分けた上での各グループの区間ごとの演奏2に対する演奏1の音量。指標は同上。(記録したインデックス数-1×11)  
・`misc['torate']`: 演奏2に対する演奏1のテンポ比。(スカラー)  
・`misc['tprate']`: 上記の区間ごとの値。(記録したインデックス数)  

dtconv.py データセットの準備をします。(直接`torch.utils.data.TensorDataset`にはしません)  
※データセット制作時は、元の音源データの格納先等を記す為、同じディレクトリに "memo_secret.py" というファイルを書く必要があります。  
※音源データはファイルサイズの大きさの都合上省いています。ご了承ください。  

"memo_secret.py" 形式 ※全て末尾はスラッシュにしてください：  
```python
O = 下地音の格納先 # Oの中のdata/ (下地音) .wav が参照されます
S = 短い音の格納先 # Sの中の (音名) .拡張子 が参照されます*
L = 長い音の格納先 # Lの中の (音名) .拡張子 が参照されます*
# 拡張子の設定についてはかなり特殊な書き方がされております。詳しくはdtconv.py 34行目-37行目をご覧いただいた上で、必要に応じて書き換えを行ってください。
# 音名はピアノ音をA0-C8で記した場合において、12平均律に基づき、黒鍵部分は全てシャープ (#の代わりにs) を用いて表現してください。
# 例: Cs4.mp3 D6.wav など
# 注意: librosaのヴァージョンによってはaudioreadやffmpegが正しく機能しない (廃止されている) 場合がございます。
# こちらの動作環境としては0.10.0を利用しておりますので、.m4aファイル等を直接扱いたい等のご要望をお持ちの方はこのヴァージョンの利用をおすすめいたします。
```

LICENSE ライセンス文書 (MIT) です。 

modelstack2.py pytorchを利用してモデルや学習関連の事がまとめられています。(modelstack.py は充分なパフォーマンスを得られなかったため削除済)  

README.md このファイルです。  

## 必要なライブラリ
次に挙げるパッケージを使用しますので、お使いのPython実行環境に`pip install`してください。記載のヴァージョンでの動作を保証します:  
<a href="https://github.com/slaypni/fastdtw">fastdtw</a> (0.3.4)  
<a href="https://librosa.org">librosa</a> (0.10.0)  
<a href="https://matplotlib.org">matplotlib</a> (3.5.0)  
<a href="https://www.numpy.org">numpy</a> (1.24.0)  
<a href="http://pydub.com">pydub</a> (0.25.1)  
<a href="https://pytorch.org">torch</a> (2.0.1)  

※8 Aug 2023より適用
<a href="https://github.com/KinWaiCheuk/nnAudio">nnAudio</a> (0.3.2)  

尚、筆者のPythonヴァージョンは3.9.7となっております。機械学習の時のみ、GPUを使用できる<a href="https://colab.research.google.com/?hl=ja">Google Colaboratory</a>や<a href="https://www.kaggle.com/">Kaggle Notebook</a>Python3.10.12を利用しています。  

## 参考文献等
データ処理にはDerivative Dynamic Time Warping[1]を使用しています。  
音高 (と音量の) 推論モデルは[2][3]を参考にしています。[2]の音高並べ替えを取り入れた実装となっています。[3]を参考に音の複数要素を分けて学習させてみようとした影響で推論モデルの初期化オプションが選べる仕様になっていますが、今のところまともに利用可能なのは`endf=0` (`modelstack2.ToneEnDv2`), `datatype=1` (`modelstack2.train`) のみとなっております。モデル初期設定に利用できる.pthファイルもすべてそのタイプのものです。  

[1] Keogh EJ, Pazzani MJ. Derivative Dynamic Time Warping. Proceedings of the 2001 SIAM International Conference on Data Mining [Internet]. 2001 [cited 2023 Aug 6];1–11. Available from: https://doi.org/10.1137/1.9781611972719.1  
[2] Chen K, Yu S, Wang C, Li W, Berg-Kirkpatrick T, Dubnov S. TONet: Tone-Octave Network for Singing Melody Extraction from Polyphonic Music [Internet]. Arxiv. 2022 [cited 2023 Aug 6]. Available from: https://arxiv.org/pdf/2202.00951.pdf  
[3] 柴田 健太郎, 中村 栄太, 錦見 亮, 吉井 和佳. 深層多重音検出を用いた音響信号から楽譜へのピアノ採譜. 情報処理学会研究報告 [Internet]. 2019 [cited 2023 Aug 6];1-6. Available from: https://eita-nakamura.github.io/articles/柴田ら_ピアノ採譜_2019.pdf  

## 連絡等
問題等ございましたら、72.igarashi.temma＠tsukukoma-gafe.org (全角＠を半角@に) までご一報ください。