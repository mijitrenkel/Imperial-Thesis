import argparse, gc, json, random
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from LSTM import LSTMMultiTask, train_epoch, eval_epoch, FEATURE_COLS, collate_batch

class SynthDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.groups = []
        for _, sub in df.groupby("path_id", sort=False):
            sub = sub.sort_values("step")
            X  = sub[FEATURE_COLS].values.astype(np.float32)
            yd = sub["delta"].values.astype(np.float32)
            self.groups.append((X, yd))
    def __len__(self): return len(self.groups)
    def __getitem__(self, idx): return self.groups[idx]

def split_paths(df: pd.DataFrame, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    paths = df["path_id"].unique().tolist()
    rng.shuffle(paths)
    n_val = int(len(paths) * val_frac)
    val_set = set(paths[:n_val])
    train_mask = df["path_id"].apply(lambda p: p not in val_set).values
    return df[train_mask].reset_index(drop=True), df[~train_mask].reset_index(drop=True)
def objective_builder(anchor_csv: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsample = 250

    full_df = pd.read_csv(anchor_csv)
    df_tr, df_va = split_paths(full_df)
    if len(df_tr) > subsample:
        keep_tr = np.random.choice(df_tr["path_id"].unique(), subsample, False)
        df_tr = df_tr[df_tr["path_id"].isin(keep_tr)].reset_index(drop=True)
        keep_va = np.random.choice(df_va["path_id"].unique(), int(subsample*0.25), False)
        df_va = df_va[df_va["path_id"].isin(keep_va)].reset_index(drop=True)

    ds_tr = SynthDataset(df_tr)
    ds_va = SynthDataset(df_va)

    def objective(trial: optuna.Trial):
        seed = 10_000 + trial.number
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        h_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        n_layers= trial.suggest_int("n_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batch = trial.suggest_categorical("batch_size", [16, 32, 64])
        lr_pre = trial.suggest_float("lr_pre", 1e-4, 3e-3, log=True)
        epochs = trial.suggest_int("epochs_pre", 5, 10)

        model = LSTMMultiTask(in_dim=len(FEATURE_COLS),hidden_dim=h_dim,n_layers=n_layers,dropout=dropout,).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr_pre)

        dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, collate_fn=collate_batch, num_workers=4)
        dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, collate_fn=collate_batch, num_workers=4)

        best_val = np.inf
        for ep in range(epochs):
            train_epoch(model, dl_tr, opt, device)
            val = eval_epoch(model, dl_va, device)
            best_val = min(best_val, val)
            trial.report(best_val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        del model, opt, dl_tr, dl_va; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return best_val

    return objective


def tune_anchor(anchor_csv: Path, n_trials=10, out_dir="tuning_out"):
    study = optuna.create_study(direction="minimize",
                                study_name=anchor_csv.stem,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    TARGET_MSE = 0.009
    def stop_on_target(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if study.best_value is not None and study.best_value < TARGET_MSE:
            print(f"best MSE {study.best_value:.6f} < target {TARGET_MSE:.6f}, stopping early.")
            study.stop()

    study.optimize(objective_builder(anchor_csv),n_trials=n_trials,show_progress_bar=True,callbacks=[stop_on_target],)
    outdir = Path(out_dir)/anchor_csv.stem
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"best_params.json","w") as f:
        json.dump(study.best_params, f, indent=2)
    study.trials_dataframe().to_csv(outdir/"trials.csv", index=False)
    print(f"{anchor_csv.stem}: best val = {study.best_value:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", default="done_py")
    parser.add_argument("--n_trials",    type=int, default=10)
    args = parser.parse_args()

    anchors = ["df_train_clean.csv"]
    for csv_name in anchors:
        tune_anchor(Path(args.synthetic_dir)/csv_name, n_trials=args.n_trials)
