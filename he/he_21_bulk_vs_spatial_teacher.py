"""Frozen post-hoc A10: paired transfer evaluation of bulk and spatial training."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import torch
from scipy import stats
import yaml


def gate():
    while os.getloadavg()[0] > 8:
        print(f'{dt.datetime.now().isoformat()} load={os.getloadavg()[0]:.2f} >8; pause 15s', flush=True)
        time.sleep(15)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
    cfgpath = root/'config/fig1.yaml'
    cfg = yaml.safe_load(cfgpath.read_text())
    rule = cfg['he_amendments']['A10_bulk_vs_spatial_teacher']
    assert rule['status'].startswith('frozen_')
    out = args.output.resolve()
    assert out.is_relative_to(root/'results/he/A10')
    out.mkdir(parents=True, exist_ok=args.resume)
    (out/'models').mkdir(exist_ok=args.resume)
    if args.resume:
        saved = yaml.safe_load((out/'run_config.yaml').read_text())
        assert saved['rule'] == rule, 'Frozen rule differs from run snapshot'
    else:
        (out/'run_config.yaml').write_text(yaml.safe_dump(dict(rule=rule,command=sys.argv,date=dt.datetime.now(dt.timezone.utc).isoformat(),seed=20261021,versions=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__)),allow_unicode=True,sort_keys=False))
    os.nice(19)
    import he_14_pancancer_pretrain as P
    M, N, A = P.M, P.N, P.A
    assert A.PROJ.resolve() == root, 'Legacy root differs from PROJECT_ROOT'
    torch.set_num_threads(2)
    dev = P.DEV
    seed = 20261021
    start = time.monotonic()
    a9 = root/rule['paths']['a9']; a8 = root/rule['paths']['a8']; a5 = root/rule['paths']['a5']
    sources = [cfgpath,Path(__file__).resolve(),root/'he/he_14_pancancer_pretrain.py',root/'he/he_10_spatial_mixed.py',root/'he/he_06_nested_nn.py',root/'he/he_03_ridge_gc_protocol.py']

    def save_table(d,name):
        d.to_csv(out/name,sep='\t',index=False)

    def manifest():
        save_table(pd.DataFrame([dict(path=str(f),size=f.stat().st_size,sha256=sha(f)) for f in dict.fromkeys(sources)]),'input_manifest.tsv')

    @torch.no_grad()
    def local(model,X,mu,sd):
        vals = []
        for chunk in X.split(256):
            gate()
            vals.append((torch.nn.functional.softplus(model.tile((chunk.to(dev).float()-mu)/sd))+1e-4).cpu().numpy())
            time.sleep(.01)
        return np.concatenate(vals)

    def paired_stats(patient):
        rows = []
        for first,second in [('S1','S0'),('S2','S0'),('S2','S1')]:
            d = patient[first].to_numpy()-patient[second].to_numpy()
            rng = np.random.default_rng(seed)
            boot = np.median(d[rng.integers(0,len(d),size=(2000,len(d)))],axis=1)
            p = 1. if np.all(d==0) else stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='approx').pvalue
            rows.append(dict(contrast=f'{first}-{second}',n_patients=len(d),median_paired_delta=np.median(d),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),p_wilcoxon=p,median_first=patient[first].median(),median_second=patient[second].median()))
        return pd.DataFrame(rows)

    if args.quick:
        torch.manual_seed(seed)
        model=N.I1().to(dev).eval();x=torch.randn(17,1536)
        got=local(model,x,torch.zeros(1536,device=dev),torch.ones(1536,device=dev))
        assert got.shape==(17,len(N.HEADS)) and np.isfinite(got).all() and (got>0).all()
        q=pd.DataFrame({'S0':[.1,.2,.3,.4], 'S1':[.2,.25,.4,.6], 'S2':[.3,.3,.5,.7]})
        save_table(paired_stats(q),'synthetic_paired_tests.tsv')
        (out/'README.md').write_text('Synthetic model and paired-statistics smoke check passed. No real outcome evaluated.\n')
        print('A10 synthetic quick passed',flush=True);return

    sources.extend([a9/'lopo_spatial.tsv',a8/'pdac.tsv',a5/'spatial.tsv'])
    ref=pd.read_csv(a9/'lopo_spatial.tsv',sep='\t')
    assert len(ref)==70 and ref.patient.nunique()==25 and not ref['sample'].duplicated().any()
    hd=M.load_hd();labels={}
    sourcecfg=M.C.load_config();sources.extend([Path(sourcecfg['_path']),Path(sourcecfg['paths']['gene_programs'])])
    for smp in M.C.samples(sourcecfg):
        if smp.sample_id in P.OWN:
            sources.extend([smp.seg/'filtered_feature_cell_matrix.h5',smp.seg/'nucleus_segmentations.geojson'])
            sources.extend([f for f in [smp.seg/'spatial/scalefactors_json.json',smp.bins2um/'spatial/scalefactors_json.json',smp.outs/'spatial/scalefactors_json.json'] if f.exists() and f.stat().st_size>0])
    for own in P.OWN:
        gate();print('Loading HD region labels',own,flush=True)
        cache=out/f'labels_{own}_150.tsv'
        if args.resume and cache.exists(): labels[own]=pd.read_csv(cache,sep='\t').set_index('key')
        else:
            labels[own]=M.region_labels(own,150);labels[own].to_csv(cache,sep='\t')
        sources.append(root/rule['paths']['hd']/f'{own}.npz')
    public={}
    for _,r in ref.iterrows():
        if r['sample'] in P.OWN: continue
        gate();sid=r['sample'];base=a8/'hest' if r.source=='HEST' else a9/'geo'
        path=base/f'{sid}.npz';sources.append(path)
        P.FEAT=base;s=P.load_sample(sid)
        with np.load(path,allow_pickle=True) as z:
            patient=str(z['patient']) if 'patient' in z.files and str(z['patient']) not in ('','nan','None') else sid
        if r.source=='HEST': patient=sid
        assert re.sub(r'_Pri\d+$','',patient)==r.patient,(sid,patient,r.patient)
        assert s['gp']>=7 and s['gc']==4 and s['caf'].sum()>=100
        public[sid]=s
    norm=a9/'models/feature_stats.pt';sources.append(norm)
    stat=torch.load(norm,map_location='cpu',weights_only=True);mu,sd=(stat[k].to(dev) for k in ['mu','sd'])
    spatial=[]
    for k in [0,1,2]:
        path=a9/'models'/f'pretrain_seed{k}.pt';sources.append(path)
        m=P.Head().to(dev);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval();spatial.append(m)

    def eval_one(models,sid,mu_,sd_,prog=0,pan=1):
        X=hd[sid]['X'] if sid in P.OWN else public[sid]['X']
        L=np.mean([local(m,X,mu_,sd_)[:,[prog,pan]] for m in models],axis=0)
        if sid in P.OWN:
            z=hd[sid];ix,iy=M.E.region_index(z['x_px']+64/float(z['mpp']),z['y_px']+64/float(z['mpp']),float(z['mpp']),150)
            pred=pd.DataFrame({'key':ix.astype(np.int64)*1_000_000+iy,'p':L[:,0],'c':L[:,1]}).groupby('key').mean()
            d=labels[sid].join(pd.DataFrame({'pred':np.log(pred.p)-np.log(pred.c)}),how='inner')
            mask=d.caf_rich.to_numpy(bool);y=d.rel.to_numpy();p=d.pred.to_numpy();keys=d.index.to_numpy()
        else:
            s=public[sid];idx=s['idx'].numpy();valid=idx>=0
            mean=lambda k:np.where(valid,L[np.maximum(idx,0),k],0).sum(1)/valid.sum(1)
            p=np.log(mean(0))-np.log(mean(1));y=s['rel'];mask=s['caf'];keys=np.arange(len(p))
        table=pd.DataFrame({'region':keys,'truth':y,'pred':p,'caf_rich':mask})
        selected=table[table.caf_rich]
        assert np.isfinite(selected[['truth','pred']]).all().all(), 'Nonfinite region prediction'
        rho=float(stats.spearmanr(selected.truth,selected.pred).statistic) if len(selected)>10 else np.nan
        return rho,int(mask.sum()),table

    old=pd.read_csv(a8/'pdac.tsv',sep='\t');sanity=[]
    s1={}
    for sid in P.OWN+P.PDAC_HEST:
        r,n,t=eval_one(spatial,sid,mu,sd);s1[sid]=(r,n,t)
        expected=float(old[(old['sample']==sid)&(old.model=='pancancer_zero_shot')].rho.iloc[0])
        sanity.append(dict(arm='S1',sample=sid,rho=r,expected=expected,abs_difference=abs(r-expected),passed=np.isfinite(r) and abs(r-expected)<=.05))
    save_table(pd.DataFrame(sanity),'sanity.tsv')
    if not all(x['passed'] for x in sanity):
        manifest();raise RuntimeError('S1 sanity failed; formal comparison stopped without model adjustment')
    print('S1 original-eight sanity passed',flush=True)

    # Match A5 seeds, labels, optimizer and tile sampling; pauses do not change updates.
    gate();pats,Et,bags=M.load_tcga();idx=np.arange(len(pats))
    sources.extend([A.HE/'tcga_targets.tsv',root/'results/fig2/tcga_expr.tsv.gz'])
    sources.extend([A.CACHE_T/'tcga'/f'{p}.npy' for p in pats])
    y,_=A.make_targets(Et,Et.iloc[:2]);Y=torch.tensor(y[N.HEADS].to_numpy(),device=dev,dtype=torch.float32)
    gate();mu0,sd0=(v.to(dev) for v in N.tile_stats(bags,idx))
    torch.save({'mu':mu0.cpu(),'sd':sd0.cpu()},out/'models/bulk_feature_stats.pt')
    bulk=[]
    for k in [0,1,2]:
        gate();rng=np.random.default_rng(k);torch.manual_seed(k);m=N.I1().to(dev);opt=torch.optim.Adam(m.parameters(),lr=M.LR,weight_decay=M.WD)
        with torch.no_grad():m.b.copy_(Y.mean(0))
        ck=out/'models'/f'bulk_seed{k}_training.pt';done=out/'models'/f'bulk_seed{k}.pt';begin=0
        if done.exists():
            m.load_state_dict(torch.load(done,map_location='cpu',weights_only=True));m.eval();bulk.append(m);continue
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=False);m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            for vals in opt.state.values():
                for key,v in vals.items():
                    if torch.is_tensor(v) and key!='step':vals[key]=v.to(dev)
            torch.set_rng_state(state['torch_rng']);torch.mps.set_rng_state(state['mps_rng']) if dev.type=='mps' else None
            rng.bit_generator.state=state['numpy_rng'];begin=state['epoch']
        for epoch in range(begin,80):
            m.train();losses=[]
            for i in rng.permutation(idx):
                gate();b=bags[i];pick=rng.choice(len(b),min(len(b),N.TILES_PER_STEP),replace=False)
                pred=m((b[pick].to(dev).float()-mu0)/sd0);loss=((pred-Y[i])**2).mean()
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach().cpu()));time.sleep(.01)
            state=dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if dev.type=='mps' else None,numpy_rng=rng.bit_generator.state)
            tmp=ck.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(ck)
            with open(out/'training_log.tsv','a') as f:
                if f.tell()==0:f.write('seed\tepoch\tmse\telapsed_s\tload1\n')
                f.write(f'{k}\t{epoch+1}\t{np.mean(losses)}\t{time.monotonic()-start}\t{os.getloadavg()[0]}\n')
            if (epoch+1)%10==0:print(f'S0 seed {k} epoch {epoch+1}/80 elapsed {time.monotonic()-start:.0f}s',flush=True)
            time.sleep(.5)
        m.eval();torch.save(m.state_dict(),done);bulk.append(m)
    print('S0 all three seeds ready',flush=True)
    s0={}
    old5=pd.read_csv(a5/'spatial.tsv',sep='\t')
    for sid in P.OWN+P.PDAC_HEST:
        r,n,t=eval_one(bulk,sid,mu0,sd0,M.HEAD_PROG,M.HEAD_PAN);s0[sid]=(r,n,t)
        expected=float(old5[(old5['sample']==sid)&(old5.model=='I1_bulk')&(old5.size_um==150)].spearman.iloc[0]) if sid in P.OWN else float(old[(old['sample']==sid)&(old.model=='I1_bulk (TCGA)')].rho.iloc[0])
        sanity.append(dict(arm='S0',sample=sid,rho=r,expected=expected,abs_difference=abs(r-expected),passed=np.isfinite(r) and abs(r-expected)<=.05))
    save_table(pd.DataFrame(sanity),'sanity.tsv')
    if not all(x['passed'] for x in sanity):
        manifest();raise RuntimeError('S0 sanity failed; formal comparison stopped without model adjustment')
    rows=[]
    for _,r in ref.iterrows():
        sid=r['sample']
        if sid not in s0:s0[sid]=eval_one(bulk,sid,mu0,sd0,M.HEAD_PROG,M.HEAD_PAN)
        if sid not in s1:s1[sid]=eval_one(spatial,sid,mu,sd)
        r0,n0,t0=s0[sid];r1,n1,t1=s1[sid]
        assert n0==n1==r.n_regions,(sid,n0,n1,r.n_regions)
        assert np.array_equal(t0.region,t1.region) and np.array_equal(t0.caf_rich,t1.caf_rich) and np.allclose(t0.truth,t1.truth)
        save_table(t0.rename(columns={'pred':'S0'}).assign(S1=t1.pred.to_numpy(),sample=sid),f'regions_{sid}.tsv.gz')
        rows.append(dict(sample=sid,patient=r.patient,source=r.source,tissue=r.tissue,treated=r.treated,n_regions=n0,S0=r0,S1=r1,S2=r.rho,status='ok' if np.isfinite([r0,r1,r.rho]).all() else 'undefined_rho'))
        print('Evaluated',sid,flush=True)
    slide=pd.DataFrame(rows);save_table(slide,'slide_rho.tsv')
    valid=slide.groupby('patient').status.apply(lambda x:(x=='ok').all())
    patient=slide.groupby('patient')[['S0','S1','S2']].mean();patient['n_slides']=slide.groupby('patient').size();patient['complete']=valid
    save_table(patient.reset_index(),'patient_rho.tsv')
    test=paired_stats(patient[patient.complete]);save_table(test,'paired_tests.tsv')
    summaries=[]
    for field in ['source','tissue','treated']:
        for value,d in slide.groupby(field):
            for arm in ['S0','S1','S2']:summaries.append(dict(group_by=field,group=value,arm=arm,n_slides=len(d),median_rho=d[arm].median(),fraction_ge_03=(d[arm]>=.3).mean()))
    save_table(pd.DataFrame(summaries),'group_summaries.tsv');manifest()
    primary=test.iloc[0];verdict=bool(primary.median_paired_delta>0 and primary.p_wilcoxon<.05)
    (out/'README.md').write_text(f'A10 post-hoc paired training-scheme transfer comparison.\nPrimary biological readout: within-slide CAF-state Spearman rho; patient-weighted paired difference.\nPrimary criterion passed: {verdict}. Full model/data/seed provenance in run_config.yaml and input_manifest.tsv.\nS0 differs from S1 in training cohort, scale, heads and target; no isolated causal attribution to teacher type.\nS2 is historical LOPO summary, same region count checked; historical S2 region predictions were not saved, so mask identity is inferred from the unchanged evaluation code and count.\n10x is reference only; original A6 outputs preserved.\n')
    (out/'COMPLETE').write_text('complete\n');print(test.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
