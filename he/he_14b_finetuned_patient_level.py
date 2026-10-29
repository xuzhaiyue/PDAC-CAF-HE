# A8 completion (included in the rules; saves per-patient predictions for CIs and paired tests): Spearman and AUC at TCGA / CPTAC patient level for the pan-cancer pretrained model fine-tuned on all 8 PDAC samples
import os, sys, numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.metrics import roc_auc_score
sys.argv = ["x"]
import he_14_pancancer_pretrain as P
M, A = P.M, P.A
os.nice(19); torch.set_num_threads(2)
sel = pd.read_csv("/Volumes/INTENSO_BIG/HEST1k/_manifest/selection.tsv", sep="\t"); oc = sel.set_index("id").organ
samples = []
for sid in sel.id:
    s = P.load_sample(sid); s["organ"] = str(oc[sid])
    if s["gp"] >= 7 and s["gc"] == 4 and s["caf"].sum() >= 100: samples.append(s)
pre = [s for s in samples if s["organ"] != "Pancreas"]; pdh = [s for s in samples if s["id"] in P.PDAC_HEST]
hd = M.load_hd(); mu, sd = P.feat_stats(pre)
ft_own = [{"X": hd[o]["X"][hd[o]["ok"]], "idx": torch.arange(int(hd[o]["ok"].sum()))[:, None],
           "valid": np.ones(int(hd[o]["ok"].sum()), bool), "y": hd[o]["yloc"][hd[o]["ok"]]} for o in P.OWN]
models = []
for k, s_ in enumerate(P.SEEDS):
    base = P.train(pre, A.SEED + 500 + s_, mu, sd)                       # same seeds as the zero-shot models
    models.append(P.train(pdh + ft_own, A.SEED + 950 + s_, mu, sd, model=base, epochs=P.FT_EPOCHS, lr=P.FT_LR))
pats, Et, bt = M.load_tcga(); cp, yc, bc = M.load_cptac()
ytr, _ = A.make_targets(Et, Et.iloc[:2]); ytr.index = pats
rows = []
for cohort, bags, ids, lab in [("TCGA", bt, pats, ytr), ("CPTAC", bc, cp, yc)]:
    st = []
    for b in bags:
        L = np.mean([P.local_out(m, b, mu, sd) for m in models], axis=0)
        st.append(np.log(L[:, 0].mean()) - np.log(L[:, 1].mean()))
    st = np.array(st)
    pd.DataFrame({"model": "pancancer_finetuned_all8", "cohort": cohort, "patient": ids, "pred_state": st}).to_csv(
        A.HE / f"A8/patient_preds_finetuned_{cohort}.tsv", sep="\t", index=False)
    for t in ["program_resid", "program_sub"]:
        y = lab.loc[ids, t].to_numpy()
        rows.append({"model": "pancancer_finetuned_all8", "cohort": cohort, "target": t,
                     "spearman": stats.spearmanr(y, st)[0], "auc": roc_auc_score(y > np.median(y), st)})
r = pd.DataFrame(rows)
old = pd.read_csv(A.HE / "A8/patient_level.tsv", sep="\t")
pd.concat([old[old.model != "pancancer_finetuned_all8"], r]).to_csv(A.HE / "A8/patient_level.tsv", sep="\t", index=False)
print(pd.concat([old, r]).round(3).to_string(index=False))
