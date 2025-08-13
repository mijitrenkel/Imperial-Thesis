import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from LSTM import FEATURE_COLS, PathDataset, collate_batch, LSTMMultiTask, train_epoch, eval_epoch

def train_full(model, loader, optimizer, device, epochs, patience, val_loader=None):
    best_val = float("inf")
    no_imp = 0
    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, loader, optimizer, device)
        if val_loader:
            va_loss = eval_epoch(model, val_loader, device)
            print(f"epoch {ep}: train {tr_loss:.4f}, val {va_loss:.4f}")
            if va_loss < best_val:
                best_val, no_imp = va_loss, 0
                torch.save(model.state_dict(), "best.pth")
            else:
                no_imp += 1
                if no_imp >= patience:
                    print("  early stopping")
                    break
        else:
            print(f"epoch {ep}: train {tr_loss:.4f}")

def driver(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = json.loads(Path(args.params_json).read_text())
    model = LSTMMultiTask(
        in_dim=len(FEATURE_COLS),
        hidden_dim=best["hidden_dim"],
        n_layers=best["n_layers"],
        dropout= best.get("dropout",0.0)
    ).to(device)
    df_real_tr = pd.read_csv(args.real_train)
    df_real_va = pd.read_csv(args.real_val)
    df_real_te = pd.read_csv(args.real_test)
    ds_tr = PathDataset(df_real_tr)
    ds_va = PathDataset(df_real_va)
    ds_te = PathDataset(df_real_te)
    dl_tr = DataLoader(ds_tr, batch_size=best["batch_size"], shuffle=True,  collate_fn=collate_batch)
    dl_va = DataLoader(ds_va, batch_size=best["batch_size"], shuffle=False, collate_fn=collate_batch)

    if args.mode=="synth+finetune":
        #STEP 1: pretrain on synthetic
        df_syn = pd.read_csv(args.synth_csv)
        ds_syn = PathDataset(df_syn)
        dl_syn = DataLoader(ds_syn, batch_size=best["batch_size"], shuffle=True, collate_fn=collate_batch)
        opt = Adam(model.parameters(), lr=best["lr_pre"])
        print("Pretraining on synthetic for", best["epochs_pre"], "epochs")
        train_full(model, dl_syn, opt, device, best["epochs_pre"], patience=0, val_loader=None)

        #STEP 2: finetune on real
        opt = Adam(model.parameters(), lr=args.lr_finetune)
        print("Finetuning on real for", args.epochs_finetune, "epochs")
        train_full(model, dl_tr, opt, device, args.epochs_finetune, args.patience, val_loader=dl_va)

    else:  #real-only
        opt = Adam(model.parameters(), lr=best["lr_pre"])
        print("Training on real for", best["epochs_pre"], "epochs")
        train_full(model, dl_tr, opt, device, best["epochs_pre"], args.patience, val_loader=dl_va)

    outs = []
    for pid, sub in df_real_te.groupby("path_id", sort=False):
        sub = sub.sort_values("step")
        X = torch.from_numpy(sub[FEATURE_COLS].to_numpy(dtype=np.float32))[None].to(device)
        L = torch.tensor([len(sub)], dtype=torch.long)
        with torch.no_grad():
            pred = model(X, L).cpu().numpy().squeeze()
        sub = sub.copy()
        sub[args.pred_col] = pred
        outs.append(sub)
    pd.concat(outs).to_csv(args.output_dir/f"{args.mode}_{args.pred_col}.csv", index=False)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["synth+finetune","real-only"])
    p.add_argument("--params_json", type=Path, required=True)
    p.add_argument("--synth_csv",type=Path,help="synthetic CSV (only for synth+finetune mode)")
    p.add_argument("--real_train",type=Path, required=True)
    p.add_argument("--real_val", type=Path, required=True)
    p.add_argument("--real_test",type=Path, required=True)
    p.add_argument("--output_dir",type=Path, default=Path("out"))
    p.add_argument("--lr_finetune",type=float, default=1e-4)
    p.add_argument("--epochs_finetune",type=int, default=5)
    p.add_argument("--patience",type=int,   default=2)
    p.add_argument("--pred_col", type=str,   default="delta_pred")
    args = p.parse_args()
    args.output_dir.mkdir(exist_ok=True)
    driver(args)
