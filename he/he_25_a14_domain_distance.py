"""A14 image-only stage: exact TCGA PCA and sealed slide distances."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import argparse,datetime as dt,hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.linalg import eigh

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.root);out=root/a.out
    assert not (out/'DISTANCE_SEALED.json').exists()
    opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0]);opened.append(p)
            if any(k in p for k in ['slide_rho','lopo_spatial','correlations.tsv','map_rho','patient_rho','panel_e_70','/regions/']):
                raise RuntimeError('Outcome access prohibited in image-only stage: '+p)
    sys.addaudithook(audit)
    inputs=[]
    target=root/'results/he/tcga_targets.tsv';inputs.append(target)
    ids=pd.read_csv(target,sep='\t',usecols=[0]).iloc[:,0]
    paths=[root/'results/he/cache_own/tcga'/f'{s}.npy' for s in ids]
    paths=[p for p in paths if p.exists()];assert len(paths)==146
    scopepath=root/'results/he/A13/2026-10-22_run01/scope.tsv';inputs.append(scopepath)
    scope=pd.read_csv(scopepath,sep='\t');assert len(scope)==70 and scope.patient.nunique()==25
    n=0;total=np.zeros(1536);cross=np.zeros((1536,1536));train=[]
    for i,p in enumerate(paths):
        x=np.load(p,mmap_mode='r');valid=np.isfinite(x).all(1);v=np.asarray(x[valid],dtype=np.float64)
        n+=len(v);total+=v.sum(0);cross+=v.T@v;inputs.append(p)
        train.append(dict(patient=p.stem,n_tiles=len(x),n_valid=len(v)))
        if i%20==0:print(f'TCGA covariance {i+1}/146, tiles={n}',flush=True)
        time.sleep(.01)
    mu=total/n;cov=(cross-n*np.outer(mu,mu))/(n-1)
    vals,vec=eigh(cov,subset_by_index=[1536-50,1535],check_finite=True)
    vals=vals[::-1];vec=vec[:,::-1];assert np.all(vals>0)
    np.savez(out/'tcga_pca50.npz',mean=mu,components=vec.T,eigenvalues=vals,n_tiles=n,ddof=1)
    pd.DataFrame(train).to_csv(out/'tcga_reference.tsv',sep='\t',index=False)
    rows=[]
    for r in scope.itertuples():
        p=root/(f'results/he/hd/{r.sample}.npz' if r.source=='own Visium HD' else f'results/he/A12/2026-10-21_run01/features_128um/{r.sample}.npy')
        if p.suffix=='.npz':
            with np.load(p) as z:x=z['features'].astype(np.float64)
        else:x=np.load(p).astype(np.float64)
        valid=np.isfinite(x).all(1);v=x[valid];assert len(v)>0
        d=np.sqrt((((v-mu)@vec)**2/vals).sum(1));inputs.append(p)
        rows.append(dict(sample=r.sample,patient=r.patient,source=r.source,tissue=r.tissue,
            preservation='FFPE' if r.source in ['own Visium HD','GSE211895'] else 'Fresh frozen',
            distance=float(np.median(d)),q25=float(np.quantile(d,.25)),q75=float(np.quantile(d,.75)),
            n_tiles=len(x),n_valid=int(valid.sum()),n_nonfinite=int((~valid).sum()),feature_path=str(p.relative_to(root))))
        print('Distance complete',r.sample,flush=True)
    pd.DataFrame(rows).to_csv(out/'slide_domain.tsv',sep='\t',index=False,float_format='%.17g')
    manifests=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs]
    pd.DataFrame(manifests).to_csv(out/'distance_input_manifest.tsv',sep='\t',index=False)
    (out/'distance_file_access.json').write_text(json.dumps(sorted(set(opened)),indent=2))
    files=['slide_domain.tsv','tcga_pca50.npz','tcga_reference.tsv','distance_input_manifest.tsv','distance_file_access.json']
    seal=dict(completed_at=dt.datetime.now().astimezone().isoformat(),n_slides=70,n_patients=25,n_tcga_patients=146,n_tcga_tiles=n,
        algorithm='exact covariance PCA50, sample covariance ddof1, median tile Mahalanobis',outcome_files_read=False,
        hashes={s:sha(out/s) for s in files},script_sha256=sha(__file__),python=sys.version,numpy=np.__version__,pandas=pd.__version__)
    (out/'DISTANCE_SEALED.json').write_text(json.dumps(seal,indent=2));print('DISTANCE SEALED',seal['completed_at'],flush=True)

if __name__=='__main__':main()
