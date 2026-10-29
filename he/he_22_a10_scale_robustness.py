"""Frozen A12: a read-only 2x2 scale sensitivity with resumable UNI2-h inference."""
from __future__ import annotations
import argparse
import datetime as dt
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(name,'2')
os.environ['HF_HUB_OFFLINE']='1'
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import tifffile
import torch
import yaml

Image.MAX_IMAGE_PIXELS=None


def gate():
    while os.getloadavg()[0]>8:
        print(f'{dt.datetime.now().isoformat()} load {os.getloadavg()[0]:.2f}>8; paused 15s',flush=True)
        time.sleep(15)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
    temp.replace(path)


def pair_test(frame,first,second,scope):
    a=frame.groupby('patient')[[first,second]].mean()
    keep=frame.groupby('patient')[[first,second]].apply(lambda x:np.isfinite(x.to_numpy()).all())
    a=a[keep];d=(a[first]-a[second]).to_numpy()
    if len(d)<2:raise ValueError('Insufficient complete patients')
    boot=np.median(d[np.random.default_rng(20261021).integers(0,len(d),(2000,len(d)))],axis=1)
    p=1. if np.all(d==0) else stats.wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='approx').pvalue
    return dict(scope=scope,contrast=f'{first}-{second}',n_patients=len(d),median_paired_delta=float(np.median(d)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),p_wilcoxon=float(p),median_first=float(a[first].median()),median_second=float(a[second].median()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--stage',choices=['all','preflight','extract','evaluate'],default='all')
    args=ap.parse_args()
    root=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()
    cfgpath=root/'config/fig1.yaml';cfg=yaml.safe_load(cfgpath.read_text());rule=cfg['he_amendments']['A12_a10_patch_scale_robustness']
    assert rule['status'].startswith('frozen_')
    out=args.output.resolve();assert out.is_relative_to(root/rule['paths']['output'])
    out.mkdir(parents=True,exist_ok=args.resume)
    for sub in ['features_128um','regions','geometry','preflight','source_snapshot']:(out/sub).mkdir(exist_ok=args.resume)
    if args.resume:
        assert yaml.safe_load((out/'run_config.yaml').read_text())['rule']==rule,'Frozen rule changed'
    else:
        (out/'run_config.yaml').write_text(yaml.safe_dump(dict(rule=rule,date=dt.datetime.now(dt.timezone.utc).isoformat(),command=sys.argv,seed=20261021,versions=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,h5py=h5py.__version__,tifffile=tifffile.__version__)),sort_keys=False,allow_unicode=True))
        (out/'source_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (out/'source_snapshot/fig1.yaml').write_bytes(cfgpath.read_bytes())
    os.nice(19);torch.set_num_threads(2);torch.manual_seed(20261021)
    start=time.monotonic();paths=rule['paths'];a10=root/paths['a10'];a9=root/paths['a9'];a8=root/paths['a8'];hest=Path(paths['hest_root'])
    sources={};protected={}
    def record(path,fullhash=False):
        path=Path(path).resolve()
        if str(path) not in sources:
            st=path.stat();sources[str(path)]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest(path) if fullhash else '')
        return path
    def table(data,name):pd.DataFrame(data).to_csv(out/name,sep='\t',index=False)
    def write_manifest():table(list(sources.values()),'input_manifest.tsv')
    def protect(path):
        path=record(path,True);protected[str(path)]=digest(path)
    def log(message):print(f'{dt.datetime.now().isoformat()} {message}',flush=True)
    if args.quick:
        d=pd.DataFrame({'patient':['a','a','b','c','d'],'x':[.1,.3,.3,.2,.4],'y':[0,.1,.2,.1,.3]})
        q=pair_test(d,'x','y','synthetic');assert q['n_patients']==4 and q['median_paired_delta']>0
        table([q],'synthetic_paired_tests.tsv');(out/'README.md').write_text('Synthetic pairing check passed; no real outcome or image accessed.\n');return
    import he_16_geo_pdac_features as G
    import he_14_pancancer_pretrain as P
    torch.set_num_threads(2)
    M,N=P.M,P.N;dev=P.DEV
    assert M.A.PROJ.resolve()==root
    G.ROOT=Path(paths['geo_raw'])
    for path in [cfgpath,Path(__file__),root/'he/he_16_geo_pdac_features.py',root/'he/he_05_extract_tcga_uni2h.py',root/'he/he_14_pancancer_pretrain.py',a10/'slide_rho.tsv']:
        record(path,True)
    reference=pd.read_csv(a10/'slide_rho.tsv',sep='\t')
    assert len(reference)==70 and reference.patient.nunique()==25
    public=reference[reference.source!='own Visium HD'].copy()
    assert len(public)==66 and public.patient.nunique()==21
    specs={}
    for _,r in public.iterrows():
        sid=r['sample'];cache=(a8/'hest' if r.source=='HEST' else a9/'geo')/f'{sid}.npz';record(cache,True)
        with np.load(cache,allow_pickle=True) as z:
            specs[sid]=dict(sample=sid,source=r.source,cache=cache,coords=z['coords'].copy(),barcode=z['barcode'].astype(str),pixel_size=float(z['pixel_size']),n=len(z['coords']))
        record(a10/f'regions_{sid}.tsv.gz',True)
    log(f'Inputs: {len(public)} public slides, {sum(s["n"] for s in specs.values())} spots; no downloads')
    for f in list((a9/'models').glob('*.pt'))+list((a10/'models').glob('bulk_seed?.pt'))+[a10/'models/bulk_feature_stats.pt']:
        protect(f)
    # Resolve image geometry against cached barcodes; never change evaluation coordinates.
    def image_data(spec):
        gate();sid=spec['sample'];geom=[]
        if spec['source']=='HEST':
            image=record(Path(paths['hest_wsis'])/f'{sid}.tif')
            metadata=record(hest/'metadata'/f'{sid}.json',True);meta=json.loads(metadata.read_text());mpp=float(meta['pixel_size_um_estimated'])
            assert abs(mpp-spec['pixel_size'])<1e-8
            patch=record(hest/'patches'/f'{sid}.h5');st=record(hest/'st'/f'{sid}.h5ad')
            with h5py.File(patch,'r') as h:
                bc=np.array([x[0].decode() if isinstance(x[0],bytes) else str(x[0]) for x in h['barcode'][:]])
                assert np.array_equal(bc,spec['barcode']) and np.array_equal(h['coords'][:],spec['coords'])
                src=int(h['img'].attrs['patch_size_src'])
            centers=spec['coords'].astype(float)+src/2
            with h5py.File(st,'r') as h:
                obs=h['obs'];ids=obs[obs.attrs['_index']][:];lookup={(v.decode() if isinstance(v,bytes) else str(v)):i for i,v in enumerate(ids)}
                actual=h['obsm/spatial'][:][[lookup[b] for b in spec['barcode']]]
                center_error=float(np.max(np.abs(actual-centers)));assert center_error<=1
            with tifffile.TiffFile(image) as tif:
                img=tif.pages[0].asarray(maxworkers=2)[...,:3]
            assert img.shape[:2]==(meta['fullres_height'],meta['fullres_width'])
            info=dict(sample=sid,image=str(image),mpp=mpp,coordinate_type='patch_top_left_plus_half_source_size',max_center_error_px=center_error,source_patch_pixels=src,width=img.shape[1],height=img.shape[0],ratio=1.)
        else:
            folder=G.ROOT/spec['source']/'samples'/sid
            fs=G.files_of(folder);image=record(fs['image'])
            for f in folder.iterdir():
                if f.is_file() and not f.name.startswith('._'):record(f)
            sf=json.loads(fs['scalefactors']);hires=Image.open(io.BytesIO(fs['hires']));img=G.read_image(image)
            ratio=img.shape[1]/(hires.size[0]/sf['tissue_hires_scalef']);mpp=55./sf['spot_diameter_fullres']/ratio
            pos=G.read_positions(fs['positions']).loc[spec['barcode']]
            assert (pos.in_tissue==1).all()
            centers=np.c_[pos.pxl_col_in_fullres,pos.pxl_row_in_fullres]*ratio
            assert np.array_equal(np.round(centers).astype(np.int64),spec['coords'])
            assert abs(mpp-spec['pixel_size'])<1e-8
            info=dict(sample=sid,image=str(image),mpp=mpp,coordinate_type='original_unrounded_spot_center',max_center_error_px=float(np.max(np.abs(centers-spec['coords']))),source_patch_pixels=np.nan,width=img.shape[1],height=img.shape[0],ratio=ratio)
        Image.fromarray(img[::max(1,img.shape[0]//800),::max(1,img.shape[1]//800)]).save(out/'geometry'/f'{sid}_thumbnail.png') if not (out/'geometry'/f'{sid}_thumbnail.png').exists() else None
        return img,centers,mpp,info
    def crop(img,center,mpp,field,source):
        h,w=img.shape[:2];x,y=center
        if source=='HEST':
            size=int(round(field/mpp));left=int(round(x-size/2));top=int(round(y-size/2));right=left+size;bottom=top+size
        else:
            half=field/mpp/2;left=int(max(0,x-half));top=int(max(0,y-half));right=int(min(w,x+half));bottom=int(min(h,y+half))
        left=max(0,left);top=max(0,top);right=min(w,right);bottom=min(h,bottom)
        if right<=left or bottom<=top:raise ValueError('Empty crop')
        return Image.fromarray(img[top:bottom,left:right]).resize((224,224)),right-left,bottom-top
    @torch.inference_mode()
    def encode(model,prep,ims):
        gate();result=model(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy();assert np.isfinite(result).all();time.sleep(.05);return result.astype(np.float16)
    marker=out/'preflight/PASSED'
    if args.stage in ['all','preflight','extract']:
        gate();from he_05_extract_tcga_uni2h import load_model
        cache=Path(paths['uni_cache']).expanduser()
        for f in cache.glob('snapshots/*/config.json'):record(f,True)
        for f in cache.glob('snapshots/*/pytorch_model.bin'):record(f,True)
        for f in cache.glob('snapshots/*/model.safetensors'):record(f,True)
        encoder,prep=load_model(str(dev));torch.set_num_threads(2)
        if not marker.exists():
            checks=[];geo_checks=[]
            selected=list(public[public.source=='HEST']['sample'])+list(public[public.source!='HEST'].groupby('source',sort=True).first()['sample'])
            for sid in selected:
                spec=specs[sid];img,centers,mpp,info=image_data(spec);geo_checks.append(info)
                indices=np.linspace(0,spec['n']-1,8,dtype=int)
                ims=[crop(img,centers[i],mpp,112.,spec['source'])[0] for i in indices]
                fresh=encode(encoder,prep,ims).astype(float)
                with np.load(spec['cache'],allow_pickle=True) as z:old=z['features'][indices].astype(float)
                cosine=np.sum(fresh*old,axis=1)/(np.linalg.norm(fresh,axis=1)*np.linalg.norm(old,axis=1))
                for i,c in zip(indices,cosine):checks.append(dict(sample=sid,source=spec['source'],spot=int(i),cosine=float(c)))
                preview=Image.new('RGB',(224*4,224*2),(255,255,255))
                for j,im in enumerate(ims):preview.paste(im,(j%4*224,j//4*224))
                preview.save(out/'preflight'/f'{sid}_reconstructed112.png')
                log(f'preflight {sid} cosine median={np.median(cosine):.5f}, min={cosine.min():.5f}')
                del img;gc.collect()
            table(checks,'preflight/feature_cosines.tsv');table(geo_checks,'preflight/geometry.tsv');write_manifest()
            summary=pd.DataFrame(checks).groupby('sample').cosine.agg(['median','min'])
            if not ((summary['median']>=.98)&(summary['min']>=.90)).all():
                raise RuntimeError('112um reconstruction failed frozen feature sanity; no formal A12 comparison produced')
            marker.write_text('112um feature reconstruction passed\n')
        if args.stage=='preflight':log('Preflight complete');return
        if args.stage in ['all','extract']:
            qc=[]
            for number,(_,r) in enumerate(public.iterrows(),1):
                sid=r['sample'];spec=specs[sid];dest=out/'features_128um'/f'{sid}.npy';meta=dest.with_suffix('.json');partial=dest.with_suffix('.partial.npy');progress=dest.with_suffix('.progress.json')
                if dest.exists():
                    assert meta.exists();m=json.loads(meta.read_text());assert m['cache_sha256']==sources[str(spec['cache'].resolve())]['sha256'] and m['field_um']==128
                    qc.append(m);log(f'[{number}/66] reuse completed {sid}');continue
                img,centers,mpp,info=image_data(spec);Nspot=spec['n'];ts=time.monotonic();done=0
                if partial.exists():
                    assert progress.exists();old=json.loads(progress.read_text());assert old['n_spots']==Nspot and old['cache_sha256']==sources[str(spec['cache'].resolve())]['sha256'];done=old['completed']
                    arr=np.lib.format.open_memmap(partial,mode='r+')
                else:arr=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(Nspot,1536))
                widths=[];heights=[]
                for i in range(done,Nspot,32):
                    gate();ims=[]
                    for center in centers[i:i+32]:
                        im,w,h=crop(img,center,mpp,128.,spec['source']);ims.append(im);widths.append(w);heights.append(h)
                    try:arr[i:i+len(ims)]=encode(encoder,prep,ims)
                    except RuntimeError as e:
                        if 'out of memory' not in str(e).lower():raise
                        log(f'OOM {sid} batch {i}; retry in batches of 8')
                        torch.mps.empty_cache()
                        arr[i:i+len(ims)]=np.concatenate([encode(encoder,prep,ims[j:j+8]) for j in range(0,len(ims),8)])
                    arr.flush();atomic_json(progress,dict(n_spots=Nspot,completed=min(i+32,Nspot),cache_sha256=sources[str(spec['cache'].resolve())]['sha256']))
                    if (i//32)%20==0:log(f'[{number}/66] {sid}: {min(i+32,Nspot)}/{Nspot} spots elapsed={time.monotonic()-start:.0f}s')
                assert np.isfinite(arr).all();del arr;partial.replace(dest)
                info.update(n_spots=Nspot,field_um=128,feature_shape=[Nspot,1536],cache_sha256=sources[str(spec['cache'].resolve())]['sha256'],feature_sha256=digest(dest),seconds_this_session=time.monotonic()-ts,edge_crops_this_session=int(sum(w<round(128/mpp)-2 or h<round(128/mpp)-2 for w,h in zip(widths,heights))),resumed_at_spot=done)
                atomic_json(meta,info);qc.append(info);table(qc,'extraction_qc.tsv');write_manifest()
                log(f'[{number}/66] completed {sid}, {Nspot} spots, {info["seconds_this_session"]:.1f}s')
                del img;gc.collect();torch.mps.empty_cache() if dev.type=='mps' else None
            (out/'EXTRACTION_COMPLETE').write_text('66 public slides extracted\n')
        del encoder;gc.collect();torch.mps.empty_cache() if dev.type=='mps' else None
        if args.stage=='extract':log('Feature extraction complete');return
    assert marker.exists() and (out/'EXTRACTION_COMPLETE').exists(),'Complete preflight and extraction first'
    @torch.inference_mode()
    def local_ensemble(models,X,mu,sd,columns):
        values=[]
        for m in models:
            rows=[]
            for sub in torch.from_numpy(np.asarray(X)).split(256):
                gate();loc=torch.nn.functional.softplus(m.tile((sub.to(dev).float()-mu)/sd))+1e-4
                rows.append(loc[:,columns].cpu().numpy());time.sleep(.01)
            values.append(np.concatenate(rows))
        return np.mean(values,axis=0)
    norm0=torch.load(a10/'models/bulk_feature_stats.pt',map_location='cpu',weights_only=True);norm1=torch.load(a9/'models/feature_stats.pt',map_location='cpu',weights_only=True)
    norms=[tuple(d[k].to(dev) for k in ['mu','sd']) for d in [norm0,norm1]];models=[[],[]]
    for k in range(3):
        m=N.I1().to(dev);m.load_state_dict(torch.load(a10/'models'/f'bulk_seed{k}.pt',map_location='cpu',weights_only=True));m.eval();models[0].append(m)
        m=P.Head().to(dev);m.load_state_dict(torch.load(a9/'models'/f'pretrain_seed{k}.pt',map_location='cpu',weights_only=True));m.eval();models[1].append(m)
    rows=[];sanity=[]
    for _,r in reference.iterrows():
        sid=r['sample'];base=dict(sample=sid,patient=r.patient,source=r.source,tissue=r.tissue,treated=r.treated,n_regions=r.n_regions)
        if r.source=='own Visium HD':
            base.update(S0_112=r.S0,S1_112=r.S1,S0_128=r.S0,S1_128=r.S1,S2_historical=r.S2,low_scale_actual_um=128.,status='ok_fixed_HD_anchor');rows.append(base);continue
        spec=specs[sid];P.FEAT=spec['cache'].parent;s=P.load_sample(sid)
        original=pd.read_csv(a10/f'regions_{sid}.tsv.gz',sep='\t',float_precision='round_trip')
        assert np.array_equal(original.region,np.arange(len(s['rel']))), 'Region index mismatch'
        assert np.array_equal(original.caf_rich,s['caf']), 'CAF-rich mask mismatch'
        # CSV serialization must be round-tripped to the original floating dtype.
        assert np.array_equal(original.truth.to_numpy(dtype=s['rel'].dtype),s['rel']), 'Label value mismatch after original-dtype restoration'
        assert int(s['caf'].sum())==r.n_regions
        idx=s['idx'].numpy();valid=idx>=0
        with np.load(spec['cache'],allow_pickle=True) as z:Xold=z['features']
        Xnew=np.load(out/'features_128um'/f'{sid}.npy',mmap_mode='r')
        region=original[['region','truth','caf_rich']].copy()
        for scale,X in [(112,Xold),(128,Xnew)]:
            assert X.shape==(spec['n'],1536)
            for arm,(ms,(mu,sd),cols) in enumerate(zip(models,norms,[[M.HEAD_PROG,M.HEAD_PAN],[0,1]])):
                L=local_ensemble(ms,X,mu,sd,cols)
                means=[np.where(valid,L[np.maximum(idx,0),k],0).sum(1)/valid.sum(1) for k in [0,1]]
                pred=np.log(means[0])-np.log(means[1]);rho=float(stats.spearmanr(pred[s['caf']],s['rel'][s['caf']]).statistic)
                key=f'S{arm}_{scale}';base[key]=rho;region[key]=pred
                if scale==112:
                    expected=float(r[f'S{arm}']);sanity.append(dict(sample=sid,arm=f'S{arm}',rho=rho,expected=expected,abs_difference=abs(rho-expected),passed=np.isfinite(rho) and abs(rho-expected)<=.001))
        base.update(S2_historical=r.S2,low_scale_actual_um=112.,status='ok' if np.isfinite([base[k] for k in ['S0_112','S1_112','S0_128','S1_128']]).all() else 'undefined_rho')
        rows.append(base);region.to_csv(out/'regions'/f'{sid}.tsv.gz',sep='\t',index=False);table(sanity,'legacy_sanity.tsv');table(rows,'slide_rho.tsv')
        log(f'evaluated 2x2 {sid}; exact A10 region/mask agreement')
    slide=pd.DataFrame(rows);table(sanity,'legacy_sanity.tsv');table(slide,'slide_rho.tsv')
    if not pd.DataFrame(sanity).passed.all():raise RuntimeError('Legacy A10 reproduction failed; paired statistical summary stopped')
    contrasts=[('S1_128','S0_128'),('S1_112','S0_112'),('S1_112','S0_128'),('S0_128','S0_112'),('S1_128','S1_112')]
    tests=[]
    for scope,frame in [('all70_25patients',slide),('public66_21patients',slide[slide.source!='own Visium HD'])]:
        for a,b in contrasts:tests.append(pair_test(frame,a,b,scope))
    table(tests,'paired_tests.tsv')
    cols=['S0_112','S1_112','S0_128','S1_128'];patient=slide.groupby('patient')[cols].mean();patient['n_slides']=slide.groupby('patient').size();patient['complete']=slide.groupby('patient').apply(lambda d:np.isfinite(d[cols].to_numpy()).all(),include_groups=False)
    table(patient.reset_index(),'patient_rho.tsv')
    groups=[]
    for field in ['source','tissue']:
        for value,d in slide.groupby(field):
            for col in cols:groups.append(dict(group_by=field,group=value,model_scale=col,n_slides=len(d),median_rho=d[col].median(),fraction_ge_03=float((d[col]>=.3).mean())))
    table(groups,'group_summaries.tsv')
    # The between-model difference in scale response is descriptive only.
    public_patient=slide[slide.source!='own Visium HD'].groupby('patient')[cols].mean()
    interaction=(public_patient.S1_128-public_patient.S1_112)-(public_patient.S0_128-public_patient.S0_112)
    table(pd.DataFrame({'patient':interaction.index,'scale_response_difference':interaction.to_numpy()}),'descriptive_scale_response.tsv')
    for f,h in protected.items():assert digest(f)==h,f'Protected weight changed: {f}'
    atomic_json(out/'protected_hashes.json',protected);write_manifest()
    primary=tests[0];passed=primary['median_paired_delta']>0 and primary['p_wilcoxon']<.05
    atomic_json(out/'verdict.json',dict(primary_passed=passed,primary=primary,elapsed_seconds=time.monotonic()-start,n_public_slides=66,n_total_slides=70,n_patients=25))
    (out/'README.md').write_text('A12 post-hoc 2x2 input-scale sensitivity. No training, no downloading.\nPrimary: S1_128 minus S0_128, identical input scale on all70 slides, patient-weighted. Public66 have true 2x2; own HD anchors only have128um and are held fixed in both named columns.\nLegacy112 predictions reproduced against A10; evaluation coordinates, labels and masks unchanged. HEST image centers restored from patch-origin metadata, not from evaluation-coordinate reinterpretation.\nPassing supports robustness to this tested scale difference, not isolation of a causal supervision effect. Prior A9 success threshold unchanged.\nSee run_config.yaml, input_manifest.tsv, extraction_qc.tsv, geometry/, preflight/, source_snapshot/, regions/, and verdict.json.\n')
    (out/'COMPLETE').write_text('complete\n');log(f'A12 COMPLETE primary passed={passed}');print(pd.DataFrame(tests).to_string(index=False),flush=True)


if __name__=='__main__':main()
