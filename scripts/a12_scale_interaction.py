"""Summarize the frozen secondary difference in input-scale responses."""
from pathlib import Path
import argparse
import os
import sys
os.environ.setdefault('OMP_NUM_THREADS','2')
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();os.nice(19)
root=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve();sys.path.insert(0,str(root/'he'))
from he_22_a10_scale_robustness import pair_test
s=pd.read_csv(a.run/'slide_rho.tsv',sep='\t');s['spatial_scale_gain']=s.S1_128-s.S1_112;s['bulk_scale_gain']=s.S0_128-s.S0_112
rows=[]
for scope,d in [('all70_25patients',s),('public66_21patients',s[s.source!='own Visium HD'])]:rows.append(pair_test(d,'spatial_scale_gain','bulk_scale_gain',scope))
dest=a.run/'paired_scale_response.tsv'
if dest.exists():raise FileExistsError(dest)
pd.DataFrame(rows).to_csv(dest,sep='\t',index=False)
