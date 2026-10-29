"""T1: reproduce frozen held-out evaluations and render spatial maps."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
cv2.setNumThreads(2)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import he_14_pancancer_pretrain as P
import he_11_10x_independent as T
import he_16_geo_pdac_features as G

M,A=P.M,P.A
ROOT=Path(os.environ.get('PROJECT_ROOT', Path(__file__).resolve().parents[1]))
DEFAULT=ROOT/'results/he/figures/spatial_maps/2026-10-22_run01'
SELECTED=['GSM6505134','GSM8714177','GSM8452880','GSM9647220']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def gate():
    while os.getloadavg()[0]>8:
        print('load gate: waiting 15 seconds',flush=True);time.sleep(15)


def save_table(d,p):
    temporary=p.with_name(p.name+'.writing')
    d.to_csv(temporary,sep='\t',index=False,compression='gzip' if p.name.endswith('.gz') else None)
    os.replace(temporary,p)


def load_any(path,sid):
    P.FEAT=path;s=P.load_sample(sid)
    with np.load(path/f'{sid}.npz',allow_pickle=True) as z:
        pt=str(z['patient']) if 'patient' in z.files else sid
        s['patient']=re.sub(r'_Pri\d+$','',pt if pt not in ('','nan','None') else sid)
        s['tissue']=str(z['tissue']) if 'tissue' in z.files else 'Primary PDAC'
    return s


def train_same(samples,seed,mu,sd,model,log,held_sample,heartbeat=None):
    # Keep A9 batch size, sampling, optimizer, seeds and update count unchanged.
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(model.parameters(),lr=P.FT_LR,weight_decay=P.WD)
    valid=[np.where(s['valid'])[0] for s in samples]
    w=np.array([len(v) for v in valid],float);w/=w.sum()
    steps=max(1,int(sum(map(len,valid))//P.BATCH))
    for epoch in range(P.FT_EPOCHS):
        model.train();start=time.time()
        for step in range(steps):
            gate()
            k=rng.choice(len(samples),p=w);s=samples[k]
            rows=rng.choice(valid[k],min(P.BATCH,len(valid[k])),replace=False)
            sub=s['idx'][rows];u,inv=torch.unique(sub[sub>=0],return_inverse=True)
            remap=torch.full_like(sub,-1);remap[sub>=0]=inv
            loc=model.local((s['X'][u].to(P.DEV).float()-mu)/sd)
            pred=model.region(loc,remap.to(P.DEV))
            loss=((pred-torch.from_numpy(s['y'][rows]).to(P.DEV))**2).mean()
            opt.zero_grad();loss.backward();opt.step()
            time.sleep(.02)
        time.sleep(.5)
        row={'held_sample':held_sample,'seed':seed,'epoch':epoch+1,'steps':steps,'seconds':time.time()-start,'last_loss':float(loss.detach().cpu())}
        log.append(row);print(row,flush=True)
        if heartbeat is not None:
            heartbeat.write_text(json.dumps({**row,"pid":os.getpid(),"updated":time.strftime("%Y-%m-%d %H:%M:%S")},indent=2))
            save_table(pd.DataFrame(log),heartbeat.parent/"training_log.tsv")
    model.eval();return model


@torch.no_grad()
def local(models,X,mu,sd):
    values=[]
    for model in models:
        v=[]
        for start in range(0,len(X),256):
            gate();v.append(P.local_out(model,X[start:start+256],mu,sd));time.sleep(.02)
        values.append(np.concatenate(v))
    return np.mean(values,axis=0)


def hd_table(s,z,tiles,lab,model):
    step=M.FIELD_UM/float(z['mpp'])
    ix,iy=M.E.region_index(z['x_px']+step/2,z['y_px']+step/2,float(z['mpp']),150)
    keys=ix.astype(np.int64)*1000000+iy
    g=tiles.assign(key=keys).groupby('key')
    pred=np.log(g[f'{model}__p'].mean())-np.log(g[f'{model}__c'].mean())
    d=pd.DataFrame({'pred':pred}).join(lab,how='inner').reset_index()
    d['x_um']=(d.key//1000000+.5)*150;d['y_um']=(d.key%1000000+.5)*150
    d['sample']=s;d['model']=model;d['region_um']=150
    return d


def ranks(d):
    for col in ['rel','pred']:
        d[col+'_percentile']=np.nan
        mask=d.caf_rich & np.isfinite(d[col])
        d.loc[mask,col+'_percentile']=stats.rankdata(d.loc[mask,col],method='average')/mask.sum()*100
    return d


def own_image(s):
    smp=next(v for v in M.C.samples(M.C.load_config()) if v.sample_id==s)
    # Space Ranger PNG placeholders are empty here; read a small Loupe pyramid level.
    path=smp.he_image
    with path.open('rb') as f:
        header=json.loads(f.read(4096).split(b'\x00')[0]);block=header['indexBlock']
        f.seek(block['Start']);index=json.loads(gzip.decompress(f.read(block['End']-block['Start'])))
    tiles=index['SpatialImageTiles'][0];W,H=index['SpatialImages'][0]['Dims']
    levels=sorted({int(k.split('/')[0]) for k in tiles['Tiles']});top=max(levels)
    level=next((v for v in reversed(levels) if max(W,H)/2**(top-v)<=2000),min(levels))
    factor=2**(top-level);w,h=int(np.ceil(W/factor)),int(np.ceil(H/factor))
    img=np.full((h,w,3),255,np.uint8);size,overlap=tiles['TileSize'],tiles['TileOverlap']
    for name,entry in tiles['Tiles'].items():
        if not name.startswith(f'{level}/'):continue
        col,row=map(int,name.split('/')[1][:-4].split('_'))
        with path.open('rb') as f:
            f.seek(entry['Start']);raw=f.read(entry['End']-entry['Start'])
        tile=cv2.cvtColor(cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        x=col*size-(overlap if col else 0);y=row*size-(overlap if row else 0)
        th,tw=min(tile.shape[0],h-y),min(tile.shape[1],w-x);img[y:y+th,x:x+tw]=tile[:th,:tw]
    return Image.fromarray(img),(0,W*smp.mpp,H*smp.mpp,0),[path]


def tenx_image():
    archive=Path('/Volumes/INTENSO')/T.ROOT.name/'Visium_HD_3prime_Human_Pancreatic_Cancer_spatial.tar.gz'
    with tarfile.open(archive) as tar:
        raw=tar.extractfile('spatial/tissue_hires_image.png').read()
    im=Image.open(io.BytesIO(raw)).convert('RGB')
    sfpath=T.BIN/'spatial/scalefactors_json.json';sf=json.loads(sfpath.read_text())
    umpx=float(sf['microns_per_pixel'])/float(sf['tissue_hires_scalef'])
    return im,(0,im.width*umpx,im.height*umpx,0),[archive,sfpath]


def public_image(sid,s,geometry):
    row=geometry.set_index('sample').loc[sid]
    fs=G.files_of(G.ROOT/s['dataset']/'samples'/sid)
    im=Image.open(io.BytesIO(fs['hires'])).convert('RGB');sf=json.loads(fs['scalefactors'])
    z=np.load(A.HE/'A9/geo'/f'{sid}.npz',allow_pickle=True)
    pos=G.read_positions(fs['positions']).loc[z['barcode'].astype(str)]
    xy=pos[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy(float)*float(row.ratio)
    assert np.array_equal(np.round(xy).astype(np.int64),z['coords'])
    umpx=s['px']*float(row.ratio)/float(sf['tissue_hires_scalef'])
    return im,(0,im.width*umpx,im.height*umpx,0),xy*s['px'],[fs['image']]


def draw(d,im,extent,title,subtitle,path,shape,region_size):
    d=ranks(d.copy());fig,axs=plt.subplots(1,3,figsize=(10.5,3.65))
    evaluated=d.caf_rich.to_numpy(bool);rho=stats.spearmanr(d.loc[evaluated,'rel'],d.loc[evaluated,'pred']).statistic
    for i,ax in enumerate(axs):
        if i==0:ax.imshow(im,extent=extent)
        else:
            ax.imshow(im,extent=extent,alpha=.15)
            def patches(mask):
                if shape=='square':return [Rectangle((x-region_size/2,y-region_size/2),region_size,region_size) for x,y in d.loc[mask,['x_um','y_um']].to_numpy()]
                return [Circle((x,y),region_size*.43) for x,y in d.loc[mask,['x_um','y_um']].to_numpy()]
            excluded=~evaluated
            if excluded.any():ax.add_collection(PatchCollection(patches(excluded),facecolor='#d6d6d6',edgecolor='none',alpha=.55))
            pc=PatchCollection(patches(evaluated),cmap='viridis',edgecolor='none',rasterized=True)
            pc.set_array(d.loc[evaluated,'rel_percentile' if i==1 else 'pred_percentile'].to_numpy());pc.set_clim(0,100);ax.add_collection(pc)
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal');ax.axis('off')
        x0=extent[1]*.06;y0=extent[2]*.92
        ax.plot([x0,x0+500],[y0,y0],color='black',lw=2);ax.text(x0+250,y0-extent[2]*.025,'500 µm',ha='center',fontsize=7)
    for ax,label in zip(axs,['H&E','Measured relative CAF state','Predicted relative CAF state']):ax.set_title(label,fontsize=10)
    fig.suptitle(title+f'  |  ρ = {rho:.3f}; n = {int(evaluated.sum()):,}',fontsize=11,y=.99)
    fig.text(.5,.078,subtitle,ha='center',fontsize=8)
    fig.text(.5,.030,'Gray: excluded by QC / CAF-rich gate. Colours: percentiles within evaluated regions.',ha='center',fontsize=7.5)
    fig.subplots_adjust(left=.015,right=.90,top=.86,bottom=.16,wspace=.035)
    cax=fig.add_axes([.925,.25,.013,.45]);cb=fig.colorbar(pc,cax=cax,ticks=[0,50,100]);cb.ax.tick_params(labelsize=8)
    cb.set_label('Within-sample percentile\n(display only)',fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=220,facecolor='white')
    plt.close(fig);return d


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['own','public','plot','all'],default='all');ap.add_argument('--out',type=Path,default=DEFAULT);ap.add_argument('--quick',action='store_true');ap.add_argument('--samples',nargs='+',default=SELECTED);args=ap.parse_args()
    os.nice(19);torch.set_num_threads(2);cv2.setNumThreads(2);out=args.out
    for sub in ['models','tables','figures','source_snapshot']:(out/sub).mkdir(parents=True,exist_ok=True)
    protected=[A.HE/'A9/models/feature_stats.pt']+sorted((A.HE/'A9/models').glob('pretrain_seed*.pt'))
    manifest=out/'protected_before.json'
    if not manifest.exists():manifest.write_text(json.dumps({str(p):sha(p) for p in protected},indent=2))
    snapshot=out/'source_snapshot'/f'he_20_spatial_maps_{args.stage}_{time.strftime("%H%M%S")}.py'
    snapshot.write_text(Path(__file__).read_text())
    cfg={'task':'T1','selected':SELECTED,'requested_samples':args.samples,'epochs':P.FT_EPOCHS,'batch':P.BATCH,'lr':P.FT_LR,'seed0':A.SEED+1200,'threads':2,'nice':19,'pause_per_batch_s':.02,'load_gate':8,'display':'within-sample CAF-rich percentile; raw values retained','selection':'closest to historical stratum median; fixed before rerun','created':time.strftime('%Y-%m-%d %H:%M:%S'),'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'quick':args.quick}
    (out/'run_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    inputs=[];summary=[]
    if args.stage in ['own','all']:
        hd=M.load_hd()
        for sid in P.OWN+['10x']:
            gate();dest=out/'tables'/f'{sid}_I1_I2_mixed.tsv.gz'
            if dest.exists():continue
            if sid=='10x':
                z=dict(np.load(T.NPZ));lab=T.region_labels_10x(150);source=A.HE/'A6/spatial_tiles_10x.tsv.gz';expected=pd.read_csv(A.HE/'A6/spatial_10x.tsv',sep='\t')
            else:
                z=hd[sid];lab=M.region_labels(sid,150);source=A.HE/'A5'/f'spatial_tiles_{sid}.tsv.gz';expected=pd.read_csv(A.HE/'A5/spatial.tsv',sep='\t')
            tp=pd.read_csv(source,sep='\t');inputs+=[source,T.NPZ if sid=='10x' else A.HE/'hd'/f'{sid}.npz']
            for model in ['I1_bulk','I1_I2_mixed']:
                d=hd_table(sid,z,tp,lab,model);mask=d.caf_rich
                rho=stats.spearmanr(d.loc[mask,'rel'],d.loc[mask,'pred']).statistic
                exp=expected[(expected['sample']==sid)&(expected.model==model)&(expected.size_um==150)].iloc[0]
                delta=rho-float(exp.spearman);assert abs(delta)<.001,(sid,model,rho,exp.spearman)
                summary.append(dict(sample=sid,model=model,rho=rho,reference_rho=float(exp.spearman),delta=delta,n_regions=int(mask.sum()),passed=True))
                save_table(ranks(d),out/'tables'/f'{sid}_{model}.tsv.gz')
            print('HD reproduced',sid,flush=True)
        if summary:save_table(pd.DataFrame(summary),out/'own_reproduction.tsv')
    if args.stage in ['public','all']:
        pdac=[]
        hest_root=Path(yaml.safe_load((ROOT/'config/fig1.yaml').read_text())['he_amendments']['A12_a10_patch_scale_robustness']['paths']['hest_root'])
        selection=hest_root/'_manifest/selection.tsv';inputs.append(selection)
        hest_order=pd.read_csv(selection,sep='\t').id.tolist()
        for sid in [sid for sid in hest_order if sid in P.PDAC_HEST]:
            s=load_any(A.HE/'A8/hest',sid);s['patient']=sid;pdac.append(s);inputs.append(A.HE/'A8/hest'/f'{sid}.npz')
        for f in sorted((A.HE/'A9/geo').glob('*.npz')):
            s=load_any(f.parent,f.stem);inputs.append(f)
            if s['gp']>=7 and s['gc']==4 and s['caf'].sum()>=100:pdac.append(s)
        assert len(pdac)==66
        hd=M.load_hd();own=[];inputs.extend(A.HE/'hd'/f'{sid}.npz' for sid in P.OWN)
        for sid in P.OWN:
            z=hd[sid];ok=z['ok'];own.append({'id':sid,'X':z['X'][ok],'idx':torch.arange(int(ok.sum()))[:,None],'valid':np.ones(int(ok.sum()),bool),'y':z['yloc'][ok]})
        st=torch.load(protected[0],map_location='cpu');mu,sd=st['mu'].to(P.DEV),st['sd'].to(P.DEV)
        base=[torch.load(p,map_location='cpu') for p in protected[1:]];assert len(base)==3
        historical=pd.read_csv(A.HE/'A9/lopo_spatial.tsv',sep='\t');log=pd.read_csv(out/'training_log.tsv',sep='\t').to_dict('records') if (out/'training_log.tsv').exists() else []
        summary=pd.read_csv(out/'public_reproduction.tsv',sep='\t').to_dict('records') if (out/'public_reproduction.tsv').exists() else []
        for row in log:
            if 'held_sample' not in row:row['held_sample']=summary[0]['sample'] if len(summary)==1 else 'legacy_unassigned'
        selections=args.samples[:1] if args.quick else args.samples
        for sid in selections:
            if (out/'tables'/f'{sid}_A9_LOPO.tsv.gz').exists():continue
            s=next(s for s in pdac if s['id']==sid);patient=s['patient'];ft=[v for v in pdac if v['patient']!=patient]+own
            save_table(pd.DataFrame([{'sample':v['id'],'patient':v.get('patient',v['id'])} for v in ft]),out/'tables'/f'{sid}_training_members.tsv')
            assert all(v.get('patient',v['id'])!=patient for v in ft)
            models=[]
            for k,state in enumerate(base):
                f=out/'models'/f'{sid}_seed{k}.pt';m=P.Head().to(P.DEV)
                if f.exists():m.load_state_dict(torch.load(f,map_location=P.DEV));m.eval()
                else:
                    m.load_state_dict(state)
                    if args.quick:
                        old=P.FT_EPOCHS;P.FT_EPOCHS=1
                    m=train_same(ft,A.SEED+1200+k,mu,sd,m,log,sid,out/"training_heartbeat.json")
                    if args.quick:P.FT_EPOCHS=old
                    torch.save(m.state_dict(),f);save_table(pd.DataFrame(log),out/'training_log.tsv')
                models.append(m)
            L=local(models,s['X'],mu,sd);idx=s['idx'].numpy();valid=idx>=0
            amount=np.where(valid[:,:,None],L[np.maximum(idx,0)],0).sum(1)/valid.sum(1)[:,None]
            pred=np.log(amount[:,0])-np.log(amount[:,1]);rho=stats.spearmanr(pred[s['caf']],s['rel'][s['caf']]).statistic
            ref=float(historical.set_index('sample').loc[sid,'rho']);passed=abs(rho-ref)<=.03
            summary.append(dict(sample=sid,model='A9_LOPO',rho=rho,reference_rho=ref,delta=rho-ref,n_regions=int(s['caf'].sum()),passed=passed))
            save_table(pd.DataFrame(summary),out/'public_reproduction.tsv')
            print('PUBLIC REPRODUCED',sid,rho,ref,'pass',passed,flush=True)
            z=np.load(A.HE/'A9/geo'/f'{sid}.npz',allow_pickle=True)
            d=pd.DataFrame({'sample':sid,'patient':patient,'model':'A9_LOPO','rel':s['rel'],'pred':pred,'caf_rich':s['caf'],'valid':s['valid'],'barcode':z['barcode'].astype(str),'x_cache_px':z['coords'][:,0],'y_cache_px':z['coords'][:,1]})
            save_table(d,out/'tables'/f'{sid}_A9_LOPO.tsv.gz')
            if not passed and not args.quick:
                raise RuntimeError(f'{sid}: A9 reproduction exceeds 0.03 tolerance; see public_reproduction.tsv')
    if args.stage in ['plot','all']:
        inputs.append(A.HE/'A12/2026-10-21_run01/extraction_qc.tsv')
        geometry=pd.read_csv(A.HE/'A12/2026-10-21_run01/extraction_qc.tsv',sep='\t')
        plotrows=[]
        for sid in P.OWN+['10x']+SELECTED:
            models_to_plot=['I1_bulk','I1_I2_mixed'] if sid in P.OWN+['10x'] else ['A9_LOPO']
            if not any((out/'tables'/f'{sid}_{model}.tsv.gz').exists() for model in models_to_plot):continue
            if sid in P.OWN:im,extent,ip=own_image(sid);shape='square';size=150;subtitle='Held-out patient | A5 mixed / TCGA bulk | 150 µm grid';tissue='own Visium HD'
            elif sid=='10x':im,extent,ip=tenx_image();shape='square';size=150;subtitle='Independent section | A6 mixed / TCGA bulk | 150 µm grid';tissue='10x Visium HD'
            else:
                s=load_any(A.HE/'A9/geo',sid);im,extent,xy,ip=public_image(sid,s,geometry);shape='circle';size=s['nn_um'];subtitle='A9 leave-one-patient-out | spot + 1.1×NN neighbourhood';tissue=s['dataset']+' | '+s['tissue'].replace('tussue','tissue')
            inputs+=ip
            for model in (['I1_bulk','I1_I2_mixed'] if shape=='square' else ['A9_LOPO']):
                file=out/'tables'/f'{sid}_{model}.tsv.gz'
                if not file.exists():continue
                d=pd.read_csv(file,sep='\t')
                if shape=='circle':d['x_um']=xy[:,0];d['y_um']=xy[:,1];d['nn_um']=size
                if not (out/'figures'/f'{sid}_{model}.pdf').exists():
                    display_model={'I1_bulk':'Bulk supervision','I1_I2_mixed':'Bulk + HD supervision','A9_LOPO':'Spatial supervision'}[model]
                    context=('A5 patient-held-out' if sid in P.OWN else 'A6 independent section')+' | 150 µm grid' if shape=='square' else 'A9 patient-held-out | spot + 1.1×NN neighbourhood; markers show centres'
                    d=draw(d,im,extent,sid+' | '+tissue+' | '+display_model,context,out/'figures'/f'{sid}_{model}',shape,size)
                    save_table(d,file)
                plotrows.append({'sample':sid,'model':model,'image_inputs':';'.join(map(str,ip)),'extent_um':str(extent),'shape':shape,'size_um':size})
                print('PLOTTED',sid,model,flush=True)
        save_table(pd.DataFrame(plotrows),out/'figure_manifest.tsv')
    rows=[]
    for p in sorted(set(inputs+protected+[ROOT/'config/fig1.yaml',ROOT/'config/aim1.yaml'])):
        rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p) if p.stat().st_size<50000000 else 'metadata_only_large_input'})
    save_table(pd.DataFrame(rows),out/f'input_manifest_{args.stage}.tsv')
    before=json.loads(manifest.read_text());assert all(sha(p)==v for p,v in before.items())
    (out/'protected_after.json').write_text(json.dumps({str(p):sha(p) for p in protected},indent=2))
    print('DONE',args.stage,flush=True)


if __name__=='__main__':main()
