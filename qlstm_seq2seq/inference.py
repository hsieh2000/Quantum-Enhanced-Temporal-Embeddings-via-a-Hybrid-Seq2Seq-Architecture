from qlstm_seq2seq_v2 import QLSTMSeq2Seq, data_loader, record_result
import os
import json
import torch

input_path = "./QLSTM_seq2seq/data/sp500_weekly_return_rate.csv"
out_dir = os.path.join(os.path.dirname(__file__), "result/")
model_dir = "./models"
model_log_path = os.path.join(model_dir, "model_logs.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim, output_dim = 1, 1         # input dim & output dim should be the same to match the autoencoder architecture
B = 8                                # customized Batch size

# ===== model =====
model = QLSTMSeq2Seq(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dim=8,
    vqc_depth=1,
    teacher_forcing_ratio=0.7,
    enc_out_dim=0,
    enc_input_embed_dim=8,            # reduce qubits: e.g., 20 -> 8
    dtype=torch.float32,
).to(device)

with open(model_log_path, "r", encoding="utf-8") as f:
    mdoel_log = json.load(f)

for i in mdoel_log["Seq2seq_models"]:
    ts_start, ts_end = i["training range"]["start"], i["training range"]["end"]
    loaders, comps = data_loader(input_path, batch_size=B, mode='eval', ts_start = ts_start, ts_end = ts_end)
    _, ts_loader = loaders

    model.load_state_dict(torch.load(os.path.join(model_dir, i["name"]), weights_only=True))
    model.eval()
    tf_mapping_lst = []

    with torch.no_grad():
        for b in ts_loader:
            src = b.to(device)
            pred_tf, tf_mapping = model(src, tgt=src)                  # teacher forcing output alignment
            tf_mapping_lst.append(tf_mapping)
    print(f"ts_start,  ts_end = '{ts_start}', '{ts_end}'")
    record_result(tf_mapping_lst, comps, title=f"training latent data mapping from {ts_start} to {ts_end}", output_path=out_dir)



