from QLSTM_seq2seq_Jeremy_v2 import QLSTMSeq2Seq, data_loader, record_result, get_latest_model, ensure_file, update_model_log
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import time

if __name__ == "__main__":
    torch.manual_seed(0)
    out_dir = os.path.join(os.path.dirname(__file__), "result/")
    model_dir = "./models"
    model_log_path = os.path.join(model_dir, "model_logs.json")

    # ===== params =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 8                                # customized Batch size
    input_dim, output_dim = 1, 1         # input dim & output dim should be the same to match the autoencoder architecture
    lr = 1e-3
    iters = 5                            # training steps
    print_every = 10

    # ===== dataset preparation =====
    input_path = "./QLSTM_seq2seq/data/sp500_weekly_return_rate.csv"
    # loaders, comps = data_loader(input_path, batch_size=B, split_date='2023-01-20')

    # ===== generate training and testing dataset ======
    # the start and end date
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 9, 30)

    # generate the date of the end of the quater
    Q_start = pd.date_range(start=start_date, end=end_date, freq="QS")
    Q_end = pd.date_range(start=start_date, end=end_date, freq="QE")

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

    # output range
    for i in range(len(Q_end)-1):
        # ===== Loss & Optimizer =====
        latest_model = get_latest_model(model_dir)
        if latest_model is not None:
            model_path = os.path.join(model_dir, latest_model)
            model.load_state_dict(torch.load(model_path, weights_only=True))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        tr_start = Q_start[i].strftime('%Y-%m-%d')
        tr_end = (Q_end[i] + timedelta(days=1)).strftime('%Y-%m-%d')
        ts_start = Q_start[i+1].strftime('%Y-%m-%d')
        ts_end = (Q_end[i+1] + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"tr_start,  tr_end = '{tr_start}', '{tr_end}'")
        print(f"ts_start,  ts_end = '{ts_start}', '{ts_end}'\n")

        loaders, comps = data_loader(input_path, batch_size=B, mode='train', tr_start = tr_start, tr_end = tr_end, ts_start = ts_start, ts_end = ts_end)
        tr_loader, ts_loader = loaders

        # ===== training loop =====
        tr_loss = []
        tr_start_time = time.time()
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # preventing gradient explosion
                optimizer.step()

            #if it % print_every == 0 or it == 1:
            print(f"[Iter {it:04d}] avg train MSE: {np.array(loss_lst).mean():.6f}")
            tr_loss.append(np.array(loss_lst).mean())
        tr_end_time = time.time()
        # ===== simple inference and eval =====
        tf_mapping_lst = []
        free_mapping_lst = []

        model.eval()
        ts_start_time = time.time()
        with torch.no_grad():
            for b in ts_loader:
                src = b.to(device)
                # src, tgt = make_batch(B, Tin, Tout, device)
                pred_tf, tf_mapping = model(src, tgt=src)                  # teacher forcing output alignment
                mse_tf = criterion(pred_tf, src).item()
                tf_mapping_lst.append(tf_mapping)

        print(f"[Eval] TF MSE: {mse_tf:.6f}")
        ts_end_time = time.time()
        # ====== Save Model & Result ======
        MODEL_NAME = f"QLSTMSeq2Seq_{int(time.time())}.pth"
        PATH = os.path.join(model_dir, MODEL_NAME)
        torch.save(model.state_dict(), PATH)

        ensure_file(model_log_path)
        log = {
            "name": MODEL_NAME,
            "prev_model": latest_model if latest_model is not None else "",
            "training range": {"start": tr_start, "end": tr_end},
            "training time": tr_end_time - tr_start_time,
            "testing range": {"start": ts_start, "end": ts_end},
            "testing time": ts_end_time - ts_start_time,
            "epoch": iters,
            "training mse": tr_loss,
            "testing mse": f"{mse_tf:.6f}",
        }
        update_model_log(model_log_path, log)
        # record_result(tf_mapping_lst, comps, title=f"training latent data mapping from {tr_start} to {tr_end}", output_path=out_dir)
        record_result(tf_mapping_lst, comps, title=f"test latent data mapping from {ts_start} to {ts_end}", output_path=out_dir)
