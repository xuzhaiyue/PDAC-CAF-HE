"""Read outcomes only after verifying the A14 image-distance seal."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import argparse,datetime as dt,hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    root=Path(a.root);out=root/a.out
    seal=json.loads((out/'DISTANCE_SEALED.json').read_text())
    assert all(sha(out/k)==v for k,v in seal['hashes'].items())
    assert not (out/'outcome_receipt.json').exists()
    receipt=dict(first_outcome_read_at=dt.datetime.now().astimezone().isoformat(),distance_seal_sha256=sha(out/'DISTANCE_SEALED.json'))
    path=root/'results/he/A12/2026-10-21_run01/slide_rho.tsv'
    receipt['outcome_path']=str(path);receipt['outcome_sha256']=sha(path)
    q=pd.read_csv(path,sep='\t');d=pd.read_csv(out/'slide_domain.tsv',sep='\t')
    assert set(q['sample'])==set(d['sample']) and len(q)==70
    s=d.merge(q[['sample','patient','S0_128','S1_128']],on=['sample','patient'],validate='one_to_one');assert len(s)==70
    s.to_csv(out/'slide_domain_outcomes.tsv',sep='\t',index=False)
    p=s.groupby('patient',sort=True).agg(distance=('distance','mean'),S0_128=('S0_128','mean'),S1_128=('S1_128','mean'),
        n_slides=('sample','size'),source=('source',lambda x:';'.join(sorted(set(x)))),preservation=('preservation',lambda x:';'.join(sorted(set(x))))).reset_index()
    assert len(p)==25 and p[['distance','S0_128','S1_128']].notna().all().all()
    p.to_csv(out/'patient_domain.tsv',sep='\t',index=False)
    rows=[]
    for model in ['S0_128','S1_128']:
        r=stats.spearmanr(p.distance,p[model]);rows.append(dict(model=model,n_patients=len(p),spearman=float(r.statistic),p_two_sided=float(r.pvalue)))
    rng=np.random.default_rng(20261021);boot=[]
    x=p[['distance','S0_128','S1_128']].to_numpy()
    for k in range(2000):
        b=x[rng.integers(0,len(x),len(x))];r0=stats.spearmanr(b[:,0],b[:,1]).statistic;r1=stats.spearmanr(b[:,0],b[:,2]).statistic
        boot.append(dict(replicate=k,S0=r0,S1=r1,difference_S1_minus_S0=r1-r0))
    boot=pd.DataFrame(boot);assert np.isfinite(boot.to_numpy()).all()
    boot.to_csv(out/'bootstrap.tsv',sep='\t',index=False)
    pd.DataFrame(rows).to_csv(out/'correlations.tsv',sep='\t',index=False)
    lo,hi=boot.difference_S1_minus_S0.quantile([.025,.975])
    verdict=dict(primary_pass=bool(rows[0]['spearman']<0 and rows[0]['p_two_sided']<.05),
        correlation_difference_S1_minus_S0=rows[1]['spearman']-rows[0]['spearman'],bootstrap_CI95=[lo,hi],
        secondary_difference_CI_excludes_zero=bool(lo>0 or hi<0),bootstrap_n=2000,seed=20261021)
    (out/'verdict.json').write_text(json.dumps(verdict,indent=2))
    s.groupby(['source','preservation']).agg(n_slides=('sample','size'),distance_median=('distance','median'),S0_median=('S0_128','median'),S1_median=('S1_128','median')).to_csv(out/'source_descriptive.tsv',sep='\t')
    font_manager.fontManager.addfont('/System/Library/Fonts/Helvetica.ttc')
    plt.rcParams.update({'font.family':'Helvetica','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.5,2.8));colors=['#677986','#245FA5']
    for model,color,label in zip(['S0_128','S1_128'],colors,['Bulk S0','Spatial S1']):
        for preservation in ['FFPE','Fresh frozen']:
            v=p[p.preservation.eq(preservation)]
            ax.scatter(v.distance,v[model],s=23,facecolors=color if preservation=='FFPE' else 'white',edgecolors=color,lw=.9,zorder=3)
        b=np.polyfit(p.distance,p[model],1);xx=np.array([p.distance.min(),p.distance.max()]);ax.plot(xx,np.polyval(b,xx),color=color,lw=1,label=label)
    ax.axhline(0,color='#BCC6CE',lw=.6,ls='--');ax.set_xlabel('Image feature distance from TCGA');ax.set_ylabel('Local Spearman ρ');ax.legend(frameon=False,fontsize=7)
    fig.text(.15,.015,'Filled: FFPE   Open: fresh frozen   |   25 patients',fontsize=6)
    fig.tight_layout(rect=[0,.05,1,1]);(out/'figures').mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(out/'figures'/f'A14_domain_vs_accuracy.{ext}',dpi=300)
    plt.close(fig)
    receipt['completed_at']=dt.datetime.now().astimezone().isoformat();(out/'outcome_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'correlations':rows,'verdict':verdict},indent=2))

if __name__=='__main__':main()
