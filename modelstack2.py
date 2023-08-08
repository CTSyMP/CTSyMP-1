import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pickle import dump, load

def powerconv(item): # 音量のarray/Tensor変換
    if isinstance(item, np.ndarray): return torch.unsqueeze(torch.from_numpy(item.astype('f4')).clone(), dim = 0) / 150
    elif isinstance(item, torch.Tensor): return (item.squeeze().to('cpu').numpy().copy() * 150).astype('u1')
    else: raise TypeError("item's type should be numpy.ndarray or torch.Tensor")

def onoffconv(item): # ノートon/fadeのarray/Tensor変換
    if isinstance(item, np.ndarray): return torch.unsqueeze(torch.from_numpy(item.astype('f4')).clone(), dim = 0)
    elif isinstance(item, torch.Tensor):
        tmp = item.squeeze().to('cpu').numpy().copy()
        tmp[tmp >= .4] = 1
        tmp[tmp <= -.4] = -1
        tmp[np.abs(tmp) < .4] = 0
        return tmp.astype('i1')
    else: raise TypeError("item's type should be numpy.ndarray or torch.Tensor")

def getdataloader(mode, data_path, batch_size): # データローダ生成
    """mode: 1 power | 2 onoff | 3 both"""
    with open(data_path, 'rb') as f: file = load(f)
    c, v, o = zip(*file)
    c = torch.stack([powerconv(item) for item in c])
    ret = []
    if mode & 1:
        v = torch.stack([powerconv(item) for item in v])
        power_set = TensorDataset(c, v)
        del v
        ret += [DataLoader(power_set, batch_size, shuffle = True, num_workers = 2, pin_memory = True)]
    if mode & 2:
        o = torch.stack([onoffconv(item) for item in o])
        onoff_set = TensorDataset(c, o)
        del o
        ret += [DataLoader(onoff_set, batch_size, shuffle = True, num_workers = 2, pin_memory = True)]
    del c
    return ret

### ここから新規 ###

class OctaveSplit(nn.Module): # オクターヴ単位でプールして同音を揃える
    def __init__(self, floor, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spl = nn.Conv2d(channels, channels, (12, 1), (12, 1))
        self.spl.requires_grad_(False)
        kernel = [[0] for _ in range(12)]
        kernel[floor] = [1]
        for i in range(channels):
            self.spl.state_dict()['weight'][i] = torch.Tensor(kernel)
        self.spl.state_dict()['bias'].zero_()
    
    def forward(self, x): return self.spl(x)

class Upsampling(nn.Module): # アップサンプリング時の次元エラーを解消
    def __init__(self, scale_factor, mode, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.up = nn.Upsample(scale_factor = scale_factor, mode =  mode)
        
    def forward(self, x:torch.Tensor):
        d = x.dim()
        if 4-d: x.unsqueeze_(1)
        x = self.up(x)
        if 4-d: x.squeeze_()
        return x

class Construct(nn.Module): # ばらしたオクターヴ単位のものを結合
    def __init__(self, channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enl = nn.ConvTranspose2d(channels, channels, 1, (12, 1))
        self.p = [nn.ConstantPad2d((0, 0, i, 11-i), 0) for i in range(12)]

    def forward(self, *x): # xにはTensor12種セット; 低い方 (F0) から
        # 註: 分解 -> 要素毎計算 -> 結合 とせず並び替えで実装も可能
        out = [self.enl(item) for item in x]
        out = sum([self.p[i](out[i]) for i in range(12)])
        return out

class ToneEnD(nn.Module): # 元版 2023/8/3
    """132 -> 88"""
    def __init__(self, endf:int, *args, **kwargs) -> None:
        """endf: 0 Sigmoid | 1 Tanh"""
        super().__init__(*args, **kwargs)
        self.init_pad = nn.ReplicationPad2d((4, 4, 0, 0))
        self.ovtn_ext = nn.Conv2d(1, 32, (37, 3))
        self.splitter = nn.ModuleList([OctaveSplit(i, 32) for i in range(12)])
        self.conv_st1 = nn.ModuleList([nn.Conv2d(32, 64, 3) for _ in range(12)])
        self.conv_st2 = nn.ModuleList([nn.Conv2d(64, 128, 3) for _ in range(12)])
        self.conv_st3 = nn.ModuleList([nn.Conv2d(128, 256, 3) for _ in range(12)])
        self.conv_st4 = nn.ModuleList([nn.Conv2d(256, 128, 1) for _ in range(12)])
        self.conv_st5 = nn.ModuleList([nn.Conv2d(256, 64, 1) for _ in range(12)])
        self.conv_st6 = nn.ModuleList([nn.Conv2d(64, 32, 1) for _ in range(12)])
        self.conv_st7 = nn.ModuleList([nn.Conv2d(64, 32, 1) for _ in range(12)])
        self.combiner = Construct(32)
        self.pre_endf = nn.Conv2d(64, 1, 1)

        self.upsample = Upsampling(scale_factor = (2, 1), mode = 'bilinear')
        self.relu = nn.ReLU()
        self.endf = [nn.Sigmoid(), nn.Tanh()][endf]

    def forward(self, x: torch.Tensor):
        d = x.dim()
        x = self.init_pad(x)
        x = self.ovtn_ext(x); x = self.relu(x); y = x.clone() # 倍音抽出とコピー作成
        p = [func(x) for func in self.splitter]; q = [item.clone() for item in p] # 分割

        # 各分割領域での操作
        p = [self.relu(self.conv_st1[i](p[i])) for i in range(12)] # 8 -> 6
        p = [self.relu(self.conv_st2[i](p[i])) for i in range(12)]; r = [item.clone() for item in p] # 6 -> 4
        p = [self.relu(self.conv_st3[i](p[i])) for i in range(12)] # 4 -> 2

        p = [self.upsample(item) for item in p] # 2 -> 4
        p = [self.relu(self.conv_st4[i](p[i])) for i in range(12)]
        p = [torch.cat([p[i], r[i][..., 1:-1]], dim = d-3) for i in range(12)] # 結合
        p = [self.relu(self.conv_st5[i](p[i])) for i in range(12)]

        p = [self.upsample(item) for item in p] # 4 -> 8
        p = [self.relu(self.conv_st6[i](p[i])) for i in range(12)]
        p = [torch.cat([p[i], q[i][..., 3:-3]], dim = d-3) for i in range(12)]
        p = [self.relu(self.conv_st7[i](p[i])) for i in range(12)]

        # 合併して出力
        x = self.combiner(*p)
        x = torch.cat([x, y[..., 3:-3]], dim = d-3)
        x = self.pre_endf(x)
        x = self.endf(x)
        return x[:, :, 4:92] if d-3 else x[:, 4:92]

class ToneEnDv2(nn.Module): # 参考モデルの一つTONet (https://arxiv.org/pdf/2202.00951.pdf) に更に倣う部分を増やし普通の畳み込み (配列変更しない) も追加
    """132 → 88""" # Transformerを用いる (TONetの実装) のがよいと考えられるが、一旦FCNで実装
    def __init__(self, endf:int, *args, **kwargs) -> None:
        """endf: 0 Sigmoid | 1 Tanh"""
        super().__init__(*args, **kwargs)
        self.init_pad = nn.ReplicationPad2d((4, 4, 0, 0))
        self.ovtn_ext = nn.Conv2d(1, 32, (37, 3))
        self.splitter = nn.ModuleList([OctaveSplit(i, 32) for i in range(12)])
        self.conv_st1 = nn.ModuleList([nn.Conv2d(32, 64, 3) for _ in range(12)])
        self.conv_st2 = nn.ModuleList([nn.Conv2d(64, 128, 3) for _ in range(12)])
        self.conv_st3 = nn.ModuleList([nn.Conv2d(128, 256, 3) for _ in range(12)])
        self.conv_st4 = nn.ModuleList([nn.Conv2d(256, 128, 1) for _ in range(12)])
        self.conv_st5 = nn.ModuleList([nn.Conv2d(256, 64, 1) for _ in range(12)])
        self.conv_st6 = nn.ModuleList([nn.Conv2d(64, 32, 1) for _ in range(12)])
        self.conv_st7 = nn.ModuleList([nn.Conv2d(64, 32, 1) for _ in range(12)])
        self.combiner = Construct(32)

        # ここから新規: 音高の並び替えをせずそのまま畳む
        self.scnv_st1 = nn.Conv2d(32, 64, 3, (3, 1))
        self.scnv_st2 = nn.Conv2d(64, 128, (2, 3), (2, 1))
        self.scnv_st3 = nn.Conv2d(128, 128, (2, 3), (2, 1))
        self.scnv_st4 = nn.Conv2d(256, 64, 1)
        self.scnv_st5 = nn.Conv2d(256, 32, 1)
        self.scnv_st6 = nn.Conv2d(128, 32, 1)

        self.pre_endf = nn.Conv2d(96, 1, 1)

        self.maxpool3 = nn.MaxPool2d((3, 1), (3, 1))
        self.maxpool2 = nn.MaxPool2d((2, 1), (2, 1))
        self.upsample = Upsampling(scale_factor = (2, 1), mode = 'bilinear')
        self.upsamp_3 = Upsampling(scale_factor = (3, 1), mode = 'bilinear')
        self.relu = nn.ReLU()
        self.endf = [nn.Sigmoid(), nn.Tanh()][endf]

    def forward(self, x: torch.Tensor):
        d = x.dim()
        x = self.init_pad(x)
        x = self.ovtn_ext(x); x = self.relu(x); y = x.clone() # 倍音抽出とコピー作成
        p = [func(x) for func in self.splitter]; q = [item.clone() for item in p] # 分割

        # 各分割領域での操作
        p = [self.relu(self.conv_st1[i](p[i])) for i in range(12)] # 8 -> 6
        p = [self.relu(self.conv_st2[i](p[i])) for i in range(12)]; r = [item.clone() for item in p] # 6 -> 4
        p = [self.relu(self.conv_st3[i](p[i])) for i in range(12)] # 4 -> 2

        p = [self.upsample(item) for item in p] # 2 -> 4
        p = [self.relu(self.conv_st4[i](p[i])) for i in range(12)]
        p = [torch.cat([p[i], r[i][..., 1:-1]], dim = d-3) for i in range(12)] # 結合
        p = [self.relu(self.conv_st5[i](p[i])) for i in range(12)]

        p = [self.upsample(item) for item in p] # 4 -> 8
        p = [self.relu(self.conv_st6[i](p[i])) for i in range(12)]
        p = [torch.cat([p[i], q[i][..., 3:-3]], dim = d-3) for i in range(12)]
        p = [self.relu(self.conv_st7[i](p[i])) for i in range(12)]

        # 合併して出力
        x = self.combiner(*p)

        # 別働隊: そのまま畳み込み
        s = self.scnv_st1(y); s = self.relu(s); t = s.clone(); u = self.maxpool3(y) # 96 -> 32
        s = self.scnv_st2(s); s = self.relu(s); v = s.clone(); w = self.maxpool2(t) # 32 -> 16
        s = torch.cat([self.relu(self.scnv_st3(s)), self.maxpool2(s)[..., 1:-1]], dim = d-3) # 16 -> 8
        s = self.scnv_st4(s); s = self.relu(s); s = self.upsample(s) # 8 -> 16
        s = torch.cat([s, v[..., 1:-1], w[..., 2:-2]], dim = d-3)
        s = self.scnv_st5(s); s = self.relu(s); s = self.upsample(s) # 16 -> 32
        s = torch.cat([s, t[..., 2:-2], u[..., 3:-3]], dim = d-3)
        s = self.scnv_st6(s); s = self.relu(s); s = self.upsamp_3(s) # 32 -> 96

        # 完全に合併
        x = torch.cat([x, y[..., 3:-3], s], dim = d-3)
        x = self.pre_endf(x)
        x = self.endf(x)
        return x[:, :, 4:92] if d-3 else x[:, 4:92]

def train(pretrained:bool, trained_data_path:list, datatype:int, traindata_path:str, \
          validdata_path:str, batch_size:int, epochs:int, lr: int, output_dir: str):
    """trained_data_path = list[state, log]\n
        datatype: 1 power | 2 onoff"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' # 今後の課題はTPU導入
    model = ToneEnDv2(datatype-1).to(device)
    preinfo = {
        'epochs': 0,
        'train_loss': [],
        'eval_loss': []
    }
    if pretrained:
        model.load_state_dict(torch.load(trained_data_path[0]))
        with open(trained_data_path[1], 'rb') as f: preinfo = load(f)
    crit = nn.MSELoss() 
    opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-6)
    train_dataloader = getdataloader(datatype, traindata_path, batch_size)[0]
    valid_dataloader = getdataloader(datatype, validdata_path, batch_size)[0]
    
    tl = len(train_dataloader)
    tloslog, vloslog = [], []
    for i in range(epochs):
        print(f'\nEpoch {i+1} / {epochs}', end = '', flush = True)
        num = 0
        model.train()
        for t, l in train_dataloader:
            num += 1
            print(f'\r\033[KEpoch {i+1} / {epochs} | {num} / {tl}', end = '', flush = True)
            t, l = t.to(device), l.to(device)
            los = crit(model(t), l)
            opt.zero_grad()
            los.backward()
            opt.step()
            if num == tl: tloslog += [los.item()]
            del los

        model.eval()
        with torch.no_grad():
            num, tmp = 0, 0
            for t, l in valid_dataloader:
                num += 1
                print(f'\r\033[KEpoch {i+1} / {epochs} | Evaluating...', end = '', flush = True)
                t, l = t.to(device), l.to(device)
                los = crit(model(t), l)
                tmp += los.item()
                del los
                if num == 5: break
            vloslog += [tmp/5]
            print(f'\r\033[KEpoch {i+1} / {epochs} | loss: {tloslog[-1]} (test), {vloslog[-1]} (eval)', end = '', flush = True)

    preinfo['epochs'] += i+1
    preinfo['train_loss'] += tloslog
    preinfo['eval_loss'] += vloslog
    torch.save(model.state_dict(), output_dir + f'/model-{datatype}-{str(preinfo["epochs"]).zfill(5)}.pth')
    with open(output_dir + f'/model-{datatype}-{str(preinfo["epochs"]).zfill(5)}.trl', 'wb') as f: dump(preinfo, f)
