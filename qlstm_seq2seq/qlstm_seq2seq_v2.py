import os
import sys
import json
import time
import re
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- dynamic import of your classes ---
PKG_DIR = os.path.join(os.path.dirname(__file__), "Batch_Pennylane-main")
if PKG_DIR not in sys.path:
    sys.path.append(PKG_DIR)
try:
    from QLSTM_v0_Batch import VQC, CustomQLSTMCell, CustomLSTM  # type: ignore
except ModuleNotFoundError:
    from batch import VQC, CustomQLSTMCell, CustomLSTM  # type: ignore

class _CustomLSTMAdapter(nn.Module):
    def __init__(self, lstm_core: nn.Module):
        super().__init__()
        self.core = lstm_core
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        out, hc = self.core(x, hidden)
        outputs = out
        h_t, c_t = hc
        if isinstance(h_t, torch.Tensor) and h_t.dim() == 3:
            h_t = h_t[-1]
        if isinstance(c_t, torch.Tensor) and c_t.dim() == 3:
            c_t = c_t[-1]
        if isinstance(outputs, list):
            outputs = torch.stack(outputs, dim=1)
        return outputs, (h_t, c_t)

class QLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vqc_depth: int,
        enc_out_dim: int = 0,
        enc_input_embed_dim: int = None,
        use_adapter: bool = False,
        dtype: torch.dtype = torch.float32,   # default float32 to reduce memory
    ):
        """
        enc_input_embed_dim: project input_dim -> enc_input_embed_dim BEFORE QLSTM.
                            If None, use input_dim directly.
        """
        super().__init__()
        self.dtype = dtype
        if enc_input_embed_dim is None:
            enc_input_embed_dim = input_dim
            self.in_proj = None
        else:
            self.in_proj = nn.Linear(input_dim, enc_input_embed_dim).to(dtype=dtype)

        cell_out = max(1, enc_out_dim)
        qcell = CustomQLSTMCell(input_size=enc_input_embed_dim, hidden_size=hidden_dim,
                                output_size=cell_out, vqc_depth=vqc_depth).to(dtype=dtype)
        core = CustomLSTM(input_size=enc_input_embed_dim, hidden_size=hidden_dim, lstm_cell_QT=qcell)
        self.core = _CustomLSTMAdapter(core) if use_adapter else core
        self.enc_out_dim = enc_out_dim
        self._proj = None  # lazy

    def forward(self, src: torch.Tensor, hidden=None):
        # src: [B, Tin, input_dim]
        src = src.to(dtype=self.dtype)
        if self.in_proj is not None:
            B, T, D = src.shape
            src = self.in_proj(src.view(B*T, D)).view(B, T, -1)
        outputs, (h_t, c_t) = self.core(src, hidden)  # outputs: [B, Tin, cell_out]
        if self.enc_out_dim > 0:
            if outputs.size(-1) != self.enc_out_dim:
                if self._proj is None:
                    self._proj = nn.Linear(outputs.size(-1), self.enc_out_dim).to(outputs.device).to(self.dtype)
                enc_seq_out = self._proj(outputs)
            else:
                enc_seq_out = outputs
        else:
            enc_seq_out = None
        return enc_seq_out, (h_t, c_t)


class QLSTMDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        vqc_depth: int,
        use_adapter: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        qcell = CustomQLSTMCell(input_size=output_dim, hidden_size=hidden_dim,
                                output_size=output_dim, vqc_depth=vqc_depth).to(dtype=dtype)
        core = CustomLSTM(input_size=output_dim, hidden_size=hidden_dim, lstm_cell_QT=qcell)
        self.core = _CustomLSTMAdapter(core) if use_adapter else core
        self.output_dim = output_dim

    def forward(self, tgt_inputs: torch.Tensor, hidden):
        tgt_inputs = tgt_inputs.to(dtype=self.dtype)
        dec_outputs, (h_t, c_t) = self.core(tgt_inputs, hidden)
        return dec_outputs, (h_t, c_t)

class QLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 8,
        vqc_depth: int = 1,
        teacher_forcing_ratio: float = 0.5,
        enc_out_dim: int = 0,
        enc_input_embed_dim: int = 8,        # reduce qubits: e.g., 20 -> 8
        use_adapter: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.encoder = QLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            vqc_depth=vqc_depth,
            enc_out_dim=enc_out_dim,
            enc_input_embed_dim=enc_input_embed_dim,
            use_adapter=use_adapter,
            dtype=dtype,
        )
        self.decoder = QLSTMDecoder(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            vqc_depth=vqc_depth,
            use_adapter=use_adapter,
            dtype=dtype,
        )
        self.output_dim = output_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dtype = dtype

        self.fc = nn.Linear(hidden_dim, 2)
        self.sigmoid = nn.Sigmoid()

        # dimension reduciton
        self.fc_featureExtracor = nn.Sequential(
            self.fc,
        )

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, out_steps: Optional[int] = None):
        device = src.device
        _, hidden = self.encoder(src)

        # dimension reduciton
        h_t, c_t = hidden
        # print(h_t)
        # print(h_t.shape)
        maps = self.fc_featureExtracor(h_t)
        # print(maps)

        if tgt is not None:
            B, Tout, D = tgt.shape
            if D != self.output_dim:
                raise ValueError(f"tgt last dim {D} must equal output_dim {self.output_dim}")
            dec_input = torch.zeros(B, 1, self.output_dim, device=device, dtype=self.dtype)
            outputs = []
            h_t, c_t = hidden
            for t in range(Tout):
                step_out, (h_t, c_t) = self.decoder(dec_input, (h_t, c_t))
                outputs.append(step_out)
                use_teacher = torch.rand(1).item() < self.teacher_forcing_ratio
                next_in = tgt[:, t:t+1, :].to(dtype=self.dtype) if use_teacher else step_out.detach()
                dec_input = next_in
            return torch.cat(outputs, dim=1), maps
        else:
            if out_steps is None or out_steps <= 0:
                raise ValueError("When tgt is None, out_steps must be a positive integer.")
            B = src.size(0)
            dec_input = torch.zeros(B, 1, self.output_dim, device=device, dtype=self.dtype)
            outputs = []
            h_t, c_t = hidden
            for _ in range(out_steps):
                step_out, (h_t, c_t) = self.decoder(dec_input, (h_t, c_t))
                outputs.append(step_out)
                dec_input = step_out.detach()
            return torch.cat(outputs, dim=1), maps

# ===== Public training utilities (callable from external scripts) =====
def train_with_loss(
    model: nn.Module,
    src: torch.Tensor,            # [B, Tin, input_dim]
    tgt: torch.Tensor,            # [B, Tout, output_dim]
    iters: int = 10,
    lr: float = 1e-3,
    teacher_forcing_ratio: float = 0.7,
    print_every: int = 10,
    grad_clip: float = 1.0,
    return_history: bool = False,  # adding params
):
    """
    Train the seq2seq model on (src, tgt) pairs with MSELoss.
    Returns:
      - (model, loss_history) if return_history=True
      - model if return_history=False
    """
    assert src.dtype == tgt.dtype, "src/tgt dtype must match"
    device = next(model.parameters()).device
    src = src.to(device)
    tgt = tgt.to(device)

    model.teacher_forcing_ratio = teacher_forcing_ratio
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()
    for it in range(1, iters + 1):
        pred, _ = model(src, tgt=tgt)          # teacher forcing alignment
        loss = criterion(pred, tgt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_history.append(float(loss.item()))
        #if it == 1 or it % print_every == 0:
        print(f"[Train] iter {it:04d} | MSE: {loss.item():.6f}")

    model.eval()
    return (model, loss_history) if return_history else model


@torch.no_grad()
def evaluate_losses(model: nn.Module, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[float, float]:
    """
    Return (TF_MSE, FreeRun_MSE)
    """
    device = next(model.parameters()).device
    src = src.to(device)
    tgt = tgt.to(device)
    criterion = nn.MSELoss()
    # Teacher forcing (aligned)
    pred_tf, _ = model(src, tgt=tgt)
    tf_mse = criterion(pred_tf, tgt).item()
    # Free-run (no teacher forcing)
    T = tgt.size(1)
    pred_free, _ = model(src, tgt=None, out_steps=T)
    free_mse = criterion(pred_free, tgt).item()
    return tf_mse, free_mse

class stockDataset(Dataset):
  def __init__(self, X):
      self.data = X.to(dtype=torch.float32)  # data could be numpy, pandas, torch tensor
    #   self.label = torch.from_numpy(y).to(dtype=torch.float32)

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      return self.data[idx]   # return a single sample

def data_loader(input_path, batch_size, mode='train', tr_start = None, tr_end = None, ts_start = None, ts_end = None):

    df = pd.read_csv(input_path, index_col=0)
    comp_name = df.columns
    random.seed(10)

    ts_df = df[(ts_start <= df.index) & (df.index < ts_end)]
    # ts_df = ts_df.iloc[:, 0:100]
    test_data = torch.transpose(torch.from_numpy(ts_df.to_numpy()).unsqueeze(2), 0, 1)
    ts_dataset = stockDataset(test_data)
    ts_loader = DataLoader(ts_dataset, batch_size=batch_size, shuffle=False)

    if mode == 'train':
        if tr_start is not None and tr_end is not None:
            tr_df = df[(tr_start <= df.index) & (df.index < tr_end)]
            # tr_df = tr_df.iloc[:, 0:100]
            train_data = torch.transpose(torch.from_numpy(tr_df.to_numpy()).unsqueeze(2), 0, 1)
            tr_dataset = stockDataset(train_data)
            tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False)
            return (tr_loader, ts_loader), (comp_name.to_list(), comp_name.to_list())
        else:
            raise ValueError("tr_start or tr_end cannot be None")

    return (None, ts_loader), (None, comp_name.to_list())
    
def plot_scatter(data, title="Scatter Plot", xlabel="f1", ylabel="f2", save_path="./"):
    x = [point[0] for point in data]
    y = [point[1] for point in data]

    # scatter plot
    plt.scatter(x, y, c='blue', marker='o')

    # add label
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # plt.show()
    plt.savefig(f'{os.path.join(save_path, title.replace(" ", "_"))}.png')

def record_result(data, comps, title, output_path, mode="eval"):
    data = torch.cat(data, dim=0).to("cpu").numpy()
    tr_comps, ts_comps = comps
    result = pd.DataFrame(index=ts_comps, columns=['f1', 'f2'])  # 479筆 index，全是 NaN
    if mode == "eval":
        for i in range(len(data)):
            result.iloc[i] = data[i]
        result.to_csv(f'{os.path.join(output_path,  title.replace(" ", "_"))}.csv', encoding='utf-8')
    elif mode == "train":
        for i in range(len(data)):
            result.iloc[i] = data[i]
        result.to_csv(f'{os.path.join(output_path,  title.replace(" ", "_"))}.csv', encoding='utf-8')

    plot_scatter(data, title=title, save_path=output_path)

def get_latest_model(folder):
    pattern = re.compile(r"^QLSTMSeq2Seq_(\d+)\.pth$")
    latest_file = None
    latest_timestamp = -1

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            timestamp = int(match.group(1))
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = filename
    return latest_file

def ensure_file(path: str, create_empty=True):
    """確保指定檔案的資料夾與檔案存在。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if create_empty and not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({'Seq2seq_models': []}, f, ensure_ascii=False)

def update_model_log(path: str, log: dict):
    with open(path, "r", encoding="utf-8") as f:
        model_logs = json.load(f)
        model_logs['Seq2seq_models'].append(log)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_logs, f, ensure_ascii=False)

if __name__ == "__main__":
    torch.manual_seed(0)
    out_dir = os.path.join(os.path.dirname(__file__), "result/")
    model_dir = "./models/"
    mdoel_log_path = os.path.join(model_dir, "model_logs.json")

    # ===== params =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 8                                # customized Batch size
    input_dim, output_dim = 1, 1         # input dim & output dim should be the same to match the autoencoder architecture
    lr = 1e-3
    iters = 5                            # training steps
    print_every = 10

    # ===== dataset preparation =====
    input_path = "./QLSTM_seq2seq/data/sp500_weekly_return_rate.csv"
    tr_start,  tr_end = '2022-01-01', '2022-07-01'
    ts_start,  ts_end = '2022-07-01', '2022-10-01'

    loaders, comps = data_loader(input_path, batch_size=B, mode='train', tr_start = tr_start, tr_end = tr_end, ts_start = ts_start, ts_end = ts_end)
    tr_loader, ts_loader = loaders

    # ===== model =====
    model = QLSTMSeq2Seq(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=8,
        vqc_depth=1,
        teacher_forcing_ratio=0.7,
        enc_out_dim=0,
        enc_input_embed_dim=8,             # reduce qubits: e.g., 20 -> 8
        dtype=torch.float32,
    ).to(device)

    # ===== Loss & Optimizer =====
    latest_model = get_latest_model(model_dir)
    if latest_model is not None:
        model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ===== training loop =====
    model.train()
    for it in range(1, iters + 1):
        # src, tgt = make_batch(B, Tin, Tout, device)
        loss_lst = []
        for b in tr_loader:
            src = b.to(device)
            pred, _ = model(src, tgt=src)                     # [B, Tout, 2]
            loss = criterion(pred, src)
            loss_lst.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 避免梯度爆炸
            optimizer.step()

        #if it % print_every == 0 or it == 1:
        print(f"[Iter {it:04d}] avg train MSE: {np.array(loss_lst).mean():.6f}")

    # ===== simple inference and eval =====
    tf_mapping_lst = []
    free_mapping_lst = []

    model.eval()
    with torch.no_grad():
        for b in ts_loader:
            src = b.to(device)
            # src, tgt = make_batch(B, Tin, Tout, device)
            pred_tf, tf_mapping = model(src, tgt=src)                  # teacher forcing 下對齊的輸出
            mse_tf = criterion(pred_tf, src).item()
            tf_mapping_lst.append(tf_mapping)

            # # auto-regressive inference eval
            # pred_free, free_mapping = model(src, tgt=None, out_steps=src.size(1))
            # mse_free = criterion(pred_free, src).item()
            # free_mapping_lst.append(free_mapping)

    print(f"[Eval] TF MSE: {mse_tf:.6f}")
    # print(f"[Eval] TF MSE: {mse_tf:.6f} | Free-run MSE: {mse_free:.6f}")
    # ====== Save Model & Result ======

    MODEL_NAME = f"QLSTMSeq2Seq_{int(time.time())}.pth"
    PATH = os.path.join(model_dir, MODEL_NAME)
    torch.save(model.state_dict(), PATH)

    ensure_file(mdoel_log_path)
    log = {
        "name": MODEL_NAME,
        "prev_model": latest_model if latest_model is not None else "",
        "training range": {"start": tr_start, "end": tr_end},
        "testing range": {"start": ts_start, "end": ts_end},
        "epoch": iters,
        "testing mse": f"{mse_tf:.6f}",
    }
    update_model_log(mdoel_log_path, log)

    record_result(tf_mapping_lst, comps, title=f"test data mapping from {ts_start} to {ts_end}", output_path=out_dir)
    # record_result(free_mapping_lst, comps, title=f"test data mapping without teacher forcing with mse {mse_tf:.6f}", output_path=out_dir )
    # plot_scatter(tf_mapping_lst, title="test data mapping with teacher forcing", save_path=out_dir)
    # plot_scatter(free_mapping_lst, title="test data mapping without teacher forcing", save_path=out_dir)