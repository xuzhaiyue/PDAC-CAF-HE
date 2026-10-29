"""Frozen A15: sealed TCGA stain reference, normalized features and fixed-head inference."""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
import argparse,datetime as dt,gc,gzip,hashlib,io,json,sys,time,traceback
import numpy as np
import pandas as pd
from PIL import Image
import cv2,openslide,tifffile,torch,yaml
from scipy import stats
Image.MAX_IMAGE_PIXELS=None
cv2.setNumThreads(2)
torch.set_num_threads(2)

def now():return dt.datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,default=str));t.replace(p)
def log(s):print(now(),s,flush=True)
def gate():
    while os.getloadavg()[0]>8:log('Load >8; pausing 15s');time.sleep(15)
def od(rgb):return -np.log((np.asarray(rgb,dtype=np.float64)+1)/256.)
def tissue(o):return np.all(o>.15,axis=1)
def fit_stains(pixels):
    o=od(pixels.reshape(-1,3));o=o[tissue(o)]
    if len(o)<100:raise ValueError('Fewer than 100 stain-fit tissue pixels')
    _,v=np.linalg.eigh(np.cov(o,rowvar=False));v=v[:,-2:]
    for k in range(2):
        if v[:,k].sum()<0:v[:,k]*=-1
    # Keep angular quantiles away from the atan2 branch cut.
    xy=o@v;phi=np.arctan2(xy[:,1],xy[:,0]);centre=np.angle(np.exp(1j*phi).mean());phi=centre+np.angle(np.exp(1j*(phi-centre)))
    a,b=np.percentile(phi,[1,99]);m=np.column_stack([v@np.array([np.cos(a),np.sin(a)]),v@np.array([np.cos(b),np.sin(b)])])
    for k in range(2):
        if m[:,k].sum()<0:m[:,k]*=-1
    if m[0,0]<m[0,1]:m=m[:,::-1]
    m/=np.linalg.norm(m,axis=0)
    if np.linalg.matrix_rank(m)<2:raise ValueError('Degenerate stain matrix')
    c=np.maximum(o@np.linalg.pinv(m).T,0);q=np.percentile(c,99,axis=0)
    if np.any(q<=1e-8):raise ValueError('Degenerate stain concentration')
    return m,q,len(o)
def normalize(im,m,q,refm,refq):
    rgb=np.asarray(im,dtype=np.uint8);o=od(rgb.reshape(-1,3));c=np.maximum(o@np.linalg.pinv(m).T,0)*(refq/q)
    arr=np.clip(256*np.exp(-(c@refm.T))-1,0,255).astype(np.uint8).reshape(rgb.shape)
    # Leave near-white/background pixels unchanged; the tissue mask is fixed by RGB.
    bg=np.max(o,axis=1)<=.15;arr.reshape(-1,3)[bg]=rgb.reshape(-1,3)[bg]
    return Image.fromarray(arr)
def pixel_sample(ims,rng,n=4096):
    return np.concatenate([np.asarray(im).reshape(-1,3)[rng.choice(im.width*im.height,min(n,im.width*im.height),replace=False)] for im in ims])
def quality(ims,mpp):
    density_sum=0.;density_n=0;sharp=[]
    for im in ims:
        o=od(np.asarray(im).reshape(-1,3));mask=tissue(o)
        if mask.any():density_sum+=float(o[mask].mean(1).sum());density_n+=int(mask.sum())
        size=(max(3,round(im.width*mpp/.5)),max(3,round(im.height*mpp/.5)))
        a=np.asarray(im.resize(size,Image.Resampling.BILINEAR));g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);mask=tissue(od(a.reshape(-1,3))).reshape(g.shape)
        lap=cv2.Laplacian(g,cv2.CV_64F)
        if mask.sum()>=100:sharp.append(float(lap[mask].var()))
    return density_sum/density_n if density_n else np.nan,float(np.median(sharp)) if sharp else np.nan

def loupe_reader(path):
    with open(path,'rb') as f:
        header=json.loads(f.read(4096).split(b'\x00')[0]);b=header['indexBlock'];f.seek(b['Start']);index=json.loads(gzip.decompress(f.read(b['End']-b['Start'])))
    tiles=index['SpatialImageTiles'][0];W,H=index['SpatialImages'][0]['Dims'];top=max(int(k.split('/')[0]) for k in tiles['Tiles']);size=tiles['TileSize'];overlap=tiles['TileOverlap']
    def read(x,y,w,h):
        canvas=Image.new('RGB',(w,h),'white')
        for row in range(max(0,y//size),min((H-1)//size,(y+h-1)//size)+1):
            for col in range(max(0,x//size),min((W-1)//size,(x+w-1)//size)+1):
                prefix=f'{top}/{col}_{row}';key=next((prefix+ext for ext in ['.png','.jpg','.jpeg'] if prefix+ext in tiles['Tiles']),None)
                if key is None:raise KeyError('Missing native Loupe tile '+prefix)
                ent=tiles['Tiles'][key]
                with open(path,'rb') as f:f.seek(ent['Start']);raw=f.read(ent['End']-ent['Start'])
                tile=Image.open(io.BytesIO(raw)).convert('RGB');ox=col*size-(overlap if col else 0);oy=row*size-(overlap if row else 0);canvas.paste(tile,(ox-x,oy-y))
        return canvas
    return read

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--stage',choices=['all','reference'],default='all');ap.add_argument('--quick',action='store_true');a=ap.parse_args()
    root=Path(os.environ['PROJECT_ROOT']).resolve();rc=yaml.safe_load(Path(a.config).read_text());out=root/rc['output'];out.mkdir(exist_ok=True,parents=True)
    cfg=yaml.safe_load((root/'config/fig1.yaml').read_text());rule=cfg['he_amendments']['A15_stain_normalisation_intervention'];assert rule['status'].startswith('frozen_')
    for d in ['reference','features_norm','regions','preflight','source_snapshot','figures']:(out/d).mkdir(exist_ok=True)
    status={'pid':os.getpid(),'started_at':now(),'stage':'initialization','completed_slides':0,'total_slides':66}
    def state(**kw):status.update(kw,updated_at=now());save(out/'status.json',status)
    inputs={}
    def record(p,hash_file=False):
        p=Path(p);s=p.stat();inputs[str(p)]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p) if hash_file else '');pd.DataFrame(inputs.values()).to_csv(out/'input_manifest.tsv',sep='\t',index=False);return p
    try:
        state();record(root/'config/fig1.yaml',True);record(Path(__file__),True);(out/'source_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        save(out/'run_config.json',dict(config=rc,rule=rule,python=sys.version,numpy=np.__version__,torch=str(torch.__version__),started_at=now(),command=sys.argv))
        if a.quick:
            rng=np.random.default_rng(1);m=np.array([[.65,.07],[.70,.99],[.29,.11]]);m/=np.linalg.norm(m,axis=0);c=rng.uniform(.1,2,(30000,2));rgb=np.clip(256*np.exp(-c@m.T)-1,0,255).astype(np.uint8)
            h,q,n=fit_stains(rgb);im=Image.fromarray(rgb[:224*100].reshape(100,224,3));res=normalize(im,h,q,h,q);assert np.isfinite(np.asarray(res)).all();assert res.size==im.size
            save(out/'synthetic_checks.json',dict(passed=True,n_pixels=n,stain_matrix=h.tolist(),concentration=q.tolist()));state(stage='synthetic_complete');return
        scope=pd.read_csv(root/rc['scope'],sep='\t');public=scope[scope.source!='own Visium HD'].copy();assert len(public)==66 and public.patient.nunique()==21
        # Reference construction does not import evaluation modules or read rho files.
        refseal=out/'reference/SEALED.json'
        if not refseal.exists():
            state(stage='TCGA_reference');rng=np.random.default_rng(20261027)
            ids=pd.read_csv(root/rc['tcga_ids'],sep='\t',usecols=[0]).iloc[:,0];ids=sorted(s for s in ids if (root/rc['tcga_features']/f'{s}.npy').exists());assert len(ids)==146
            chosen=rng.choice(ids,20,replace=False);st=pd.read_csv(root/rc['tcga_status'],sep='\t');st=st[st.status.eq('ok')];ims=[];sampled=[]
            for patient in chosen:
                gate();rows=st[st.patient.eq(patient)].sort_values('file_id');cache=[]
                for r in rows.itertuples():
                    p=record(root/rc['tcga_slide_cache']/f'{r.file_id}.npz');z=np.load(p);cache.extend((r,int(i),int(x),int(y),int(z['tile_w']),int(z['tile_h'])) for i,(x,y) in enumerate(zip(z['x'],z['y'])))
                ix=rng.choice(len(cache),10,replace=False)
                for k in ix:
                    r,i,x,y,w,h=cache[k];p=record(Path(rc['tcga_wsis'])/r.file_name)
                    with openslide.OpenSlide(str(p)) as sl:im=sl.read_region((x,y),0,(w,h)).convert('RGB')
                    ims.append(im);sampled.append(dict(patient=patient,file_id=r.file_id,path=str(p),tile_index=i,x=x,y=y,w=w,h=h))
                log(f'Reference {patient}: 10 tiles')
            pix=pixel_sample(ims,rng);m,q,n=fit_stains(pix);np.savez(out/'reference/tcga_macenko.npz',matrix=m,concentration99=q,pixels=pix)
            pd.DataFrame(sampled).to_csv(out/'reference/selected_200_tiles.tsv',sep='\t',index=False)
            preview=Image.new('RGB',(10*128,4*128),'white')
            for i,im in enumerate(ims[:40]):preview.paste(im.resize((128,128)),(i%10*128,i//10*128))
            preview.save(out/'reference/TCGA_tiles_preview.png')
            save(refseal,dict(sealed_at=now(),n_patients=20,n_tiles=200,seed=20261027,n_fit_pixels=n,reference_sha256=sha(out/'reference/tcga_macenko.npz'),selection_sha256=sha(out/'reference/selected_200_tiles.tsv'),outcome_files_read=False))
            del ims,pix;gc.collect();log('TCGA REFERENCE SEALED')
        seal=json.loads(refseal.read_text());assert sha(out/'reference/tcga_macenko.npz')==seal['reference_sha256'];z=np.load(out/'reference/tcga_macenko.npz');refm,refq=z['matrix'],z['concentration99']
        if a.stage=='reference':state(stage='reference_complete');return
        # Complete the unchanged HD anchors before any outcome access.
        hdmetrics=out/'color_HD_descriptive.tsv'
        if not hdmetrics.exists():
            state(stage='HD_RGB_metrics');aim=yaml.safe_load((root/'config/aim1.yaml').read_text());hdrows=[]
            for sample in aim['samples']:
                sid=sample['sample_id'];gate();path=record(sample['he_image']);reader=loupe_reader(path)
                with np.load(root/'results/he/hd'/f'{sid}.npz') as zz:x=zz['x_px'];y=zz['y_px'];mpp=float(zz['mpp'])
                rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('A15:'+sid).encode()).digest()[:8],'little'));indices=np.sort(rng.choice(len(x),min(64,len(x)),replace=False));w=round(128/mpp);ims=[reader(int(x[i]),int(y[i]),w,w) for i in indices]
                matrix,q,nfit=fit_stains(pixel_sample(ims,rng));angles=np.degrees(np.arccos(np.clip(np.sum(matrix*refm,axis=0),-1,1)));density,sharpness=quality(ims,mpp)
                hdrows.append(dict(sample=sid,patient=sid,source='own Visium HD',preservation='FFPE',n_sampled_tiles=len(indices),sampled_indices=indices.tolist(),OD_orig=density,OD_norm=density,laplacian_orig=sharpness,laplacian_norm=sharpness,H_angle_orig=float(angles[0]),E_angle_orig=float(angles[1]),H_angle_norm=float(angles[0]),E_angle_norm=float(angles[1]),stain_fit_tissue_pixels=nfit,intervention_applied=False))
                log('HD descriptive metrics '+sid)
            pd.DataFrame(hdrows).to_csv(hdmetrics,sep='\t',index=False)
        import he_16_geo_pdac_features as G
        from he_05_extract_tcga_uni2h import load_model
        dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');gate();encoder,prep=load_model(str(dev));torch.set_num_threads(2)
        def enc(ims):
            gate()
            with torch.inference_mode():v=encoder(torch.stack([prep(im) for im in ims]).to(dev)).float().cpu().numpy()
            assert np.isfinite(v).all();time.sleep(.05);return v.astype(np.float16)
        colors=[]
        for num,r in enumerate(public.itertuples(),1):
            sid=r.sample;dest=out/'features_norm'/f'{sid}.npy';meta=dest.with_suffix('.json');partial=dest.with_suffix('.partial.npy');progress=dest.with_suffix('.progress.json')
            if dest.exists():
                info=json.loads(meta.read_text());assert sha(dest)==info['feature_sha256'];colors.append(info['color']);state(stage='extracting',completed_slides=num,current_sample=sid);continue
            state(stage='reading_image',current_sample=sid,completed_slides=num-1);gate()
            geompath=root/rc['a12']/f'features_128um/{sid}.json';info=json.loads(record(geompath,True).read_text());imagepath=record(info['image']);mpp=info['mpp']
            if sid.startswith('NCBI'):
                with tifffile.TiffFile(imagepath) as tif:img=tif.pages[0].asarray(maxworkers=2)[...,:3]
            else:
                with G.opener(imagepath) as f:raw=f.read()
                if imagepath.name.lower().endswith(('.tif.gz','.tiff.gz')):
                    with tifffile.TiffFile(io.BytesIO(raw)) as tif:img=tif.pages[0].asarray(maxworkers=2)
                    if img.ndim==3 and img.shape[0] in (3,4) and img.shape[-1] not in (3,4):img=np.moveaxis(img,0,-1)
                    img=np.ascontiguousarray(img[...,:3]).astype(np.uint8)
                else:img=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
                del raw
            assert img.shape[:2]==(info['height'],info['width'])
            geometry=pd.read_csv(record(root/rc['a12']/f'crop_geometry/{sid}.tsv.gz',True),sep='\t');n=len(geometry);assert n==info['n_spots']
            def crop(i):
                b=geometry.iloc[i];return Image.fromarray(img[int(b.top_128):int(b.bottom_128),int(b.left_128):int(b.right_128)])
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('A15:'+sid).encode()).digest()[:8],'little'));indices=np.sort(rng.choice(n,min(64,n),replace=False));samples=[crop(i) for i in indices]
            pixels=pixel_sample(samples,rng);m,q,nfit=fit_stains(pixels);normsamples=[normalize(im,m,q,refm,refq) for im in samples]
            od0,lap0=quality(samples,mpp);od1,lap1=quality(normsamples,mpp)
            fitted,_,_=fit_stains(pixel_sample(normsamples,rng));angle=lambda a,b:np.degrees(np.arccos(np.clip(np.sum(a*b,axis=0),-1,1))).tolist()
            col=dict(sample=sid,patient=r.patient,source=r.source,preservation='FFPE' if r.source=='GSE211895' else 'Fresh frozen',n_sampled_tiles=len(indices),sampled_indices=indices.tolist(),OD_orig=od0,OD_norm=od1,laplacian_orig=lap0,laplacian_norm=lap1,H_angle_orig=angle(m,refm)[0],E_angle_orig=angle(m,refm)[1],H_angle_norm=angle(fitted,refm)[0],E_angle_norm=angle(fitted,refm)[1],stain_fit_tissue_pixels=nfit)
            save(out/'preflight'/f'{sid}_stain.json',dict(color=col,matrix=m.tolist(),concentration99=q.tolist()))
            # Replay eight unmodified 128um patches before accepting normalized inference.
            check=indices[:8];original=enc([crop(i).resize((224,224)) for i in check]).astype(float);cached=np.load(root/rc['a12']/f'features_128um/{sid}.npy',mmap_mode='r')[check].astype(float)
            cosine=np.sum(original*cached,1)/(np.linalg.norm(original,axis=1)*np.linalg.norm(cached,axis=1));assert np.median(cosine)>=.98 and np.min(cosine)>=.90,(sid,cosine)
            save(out/'preflight'/f'{sid}_feature_replay.json',dict(indices=check.tolist(),cosine=cosine.tolist(),passed=True))
            prev=Image.new('RGB',(8*224,2*224),'white')
            for j in range(8):prev.paste(samples[j].resize((224,224)),(j*224,0));prev.paste(normsamples[j].resize((224,224)),(j*224,224))
            prev.save(out/'preflight'/f'{sid}_before_after.png')
            done=0
            if partial.exists():
                pr=json.loads(progress.read_text());assert pr['reference_sha256']==seal['reference_sha256'];done=pr['completed'];arr=np.lib.format.open_memmap(partial,mode='r+')
            else:arr=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(n,1536))
            for i in range(done,n,32):
                gate();state(stage='extracting',current_sample=sid,current_tile=i,total_tiles_current=n,completed_slides=num-1)
                batch=[normalize(crop(k),m,q,refm,refq).resize((224,224)) for k in range(i,min(i+32,n))];arr[i:i+len(batch)]=enc(batch);arr.flush()
                save(progress,dict(completed=min(i+32,n),reference_sha256=seal['reference_sha256']))
                if i%640==0:log(f'[{num}/66] {sid} {min(i+32,n)}/{n}')
            assert np.isfinite(arr).all();del arr;partial.replace(dest)
            save(meta,dict(sample=sid,n_spots=n,feature_shape=[n,1536],feature_sha256=sha(dest),reference_sha256=seal['reference_sha256'],color=col,completed_at=now()));colors.append(col);pd.DataFrame(colors).to_csv(out/'color_descriptive.tsv',sep='\t',index=False)
            del img,samples,normsamples,pixels;gc.collect();torch.mps.empty_cache() if dev.type=='mps' else None;state(completed_slides=num);log(f'Completed {num}/66 {sid}')
        del encoder;gc.collect();torch.mps.empty_cache() if dev.type=='mps' else None
        metrics=pd.concat([pd.read_csv(hdmetrics,sep='\t'),pd.DataFrame(colors)],ignore_index=True);assert len(metrics)==70 and metrics['sample'].nunique()==70
        metrics['stain_difference']=metrics.H_angle_orig+metrics.E_angle_orig
        assert np.isfinite(metrics[['stain_difference','laplacian_orig']].to_numpy()).all()
        metrics.to_csv(out/'image_metrics_70.tsv',sep='\t',index=False)
        save(out/'IMAGE_METRICS_SEALED.json',dict(sealed_at=now(),n_slides=70,n_patients=25,metrics_sha256=sha(out/'image_metrics_70.tsv'),reference_sha256=seal['reference_sha256'],outcomes_read=False))
        (out/'EXTRACTION_COMPLETE').write_text(now());state(stage='fixed_head_inference')
        # Only now read stored outcomes, with the reference already sealed.
        save(out/'first_outcome_read.json',dict(time=now(),reference_seal_sha256=sha(refseal)))
        import he_14_pancancer_pretrain as P
        torch.set_num_threads(2)
        M,N=P.M,P.N;a10=root/rc['a10'];a9=root/rc['a9'];norms=[];models=[[],[]]
        for path in [a10/'models/bulk_feature_stats.pt',a9/'models/feature_stats.pt']:
            s=torch.load(record(path,True),map_location='cpu',weights_only=True);norms.append(tuple(s[k].to(dev) for k in ['mu','sd']))
        for k in range(3):
            m0=N.I1().to(dev);m0.load_state_dict(torch.load(record(a10/'models'/f'bulk_seed{k}.pt',True),map_location='cpu',weights_only=True));models[0].append(m0.eval())
            m1=P.Head().to(dev);m1.load_state_dict(torch.load(record(a9/'models'/f'pretrain_seed{k}.pt',True),map_location='cpu',weights_only=True));models[1].append(m1.eval())
        original=pd.read_csv(record(root/rc['a12']/'slide_rho.tsv',True),sep='\t').set_index('sample');rows=[]
        metric_joint=metrics.merge(original.reset_index()[['sample','patient','S0_128','S1_128']],on=['sample','patient'],validate='one_to_one');assert len(metric_joint)==70
        metric_patient=metric_joint.groupby('patient')[['stain_difference','laplacian_orig','S0_128','S1_128']].mean();assert len(metric_patient)==25;metric_patient.to_csv(out/'image_metrics_patient25.tsv',sep='\t');correlations=[]
        for metric in ['stain_difference','laplacian_orig']:
            for model in ['S0_128','S1_128']:
                corr=stats.spearmanr(metric_patient[metric],metric_patient[model]);correlations.append(dict(metric=metric,model=model,n_patients=25,spearman=float(corr.statistic),p_two_sided=float(corr.pvalue)))
        pd.DataFrame(correlations).to_csv(out/'image_quality_correlations.tsv',sep='\t',index=False)
        for r in public.itertuples():
            sid=r.sample;cache=root/(rc['hest_cache'] if sid.startswith('NCBI') else rc['geo_cache']);P.FEAT=cache;s=P.load_sample(sid);idx=s['idx'].numpy();valid=idx>=0
            baseline=pd.read_csv(record(root/rc['a12']/f'regions/{sid}.tsv.gz',True),sep='\t',float_precision='round_trip');assert np.array_equal(baseline.region,np.arange(len(s['rel'])));assert np.array_equal(baseline.caf_rich,s['caf']);assert np.array_equal(baseline.truth.to_numpy(dtype=s['rel'].dtype),s['rel'])
            region=baseline[['region','truth','caf_rich']].copy();row=dict(sample=sid,patient=r.patient,source=r.source,preservation='FFPE' if r.source=='GSE211895' else 'Fresh frozen',n_regions=int(s['caf'].sum()),S0_orig=float(original.loc[sid,'S0_128']),S1_orig=float(original.loc[sid,'S1_128']))
            X=np.load(out/'features_norm'/f'{sid}.npy',mmap_mode='r')
            for arm,(ms,(mu,sd),cols) in enumerate(zip(models,norms,[[M.HEAD_PROG,M.HEAD_PAN],[0,1]])):
                vals=[]
                for model in ms:
                    chunks=[]
                    for i in range(0,len(X),256):
                        gate()
                        with torch.inference_mode():v=torch.nn.functional.softplus(model.tile((torch.tensor(X[i:i+256],dtype=torch.float32,device=dev)-mu)/sd))+1e-4
                        chunks.append(v[:,cols].cpu().numpy());time.sleep(.02)
                    vals.append(np.concatenate(chunks))
                L=np.mean(vals,axis=0);means=[np.where(valid,L[np.maximum(idx,0),k],0).sum(1)/valid.sum(1) for k in [0,1]];pred=np.log(means[0])-np.log(means[1]);region[f'S{arm}_norm']=pred;row[f'S{arm}_norm']=float(stats.spearmanr(pred[s['caf']],s['rel'][s['caf']]).statistic)
            rows.append(row);region.to_csv(out/'regions'/f'{sid}.tsv.gz',sep='\t',index=False);pd.DataFrame(rows).to_csv(out/'slide_rho.tsv',sep='\t',index=False);log(f'Inferred {sid}')
        frame=pd.DataFrame(rows);p=frame.groupby('patient')[[f'S{k}_{v}' for k in [0,1] for v in ['orig','norm']]].mean();assert len(p)==21 and np.isfinite(frame[[f'S{k}_{v}' for k in [0,1] for v in ['orig','norm']]].to_numpy()).all()
        p.to_csv(out/'patient_rho.tsv',sep='\t');tests=[]
        for name,first,second in [('S0_intervention','S0_norm','S0_orig'),('S1_intervention','S1_norm','S1_orig'),('norm_S1_minus_S0','S1_norm','S0_norm')]:
            d=(p[first]-p[second]).to_numpy();boot=np.median(d[np.random.default_rng(20261021).integers(0,len(d),(2000,len(d)))],axis=1);pv=1. if np.all(d==0) else float(stats.wilcoxon(d,method='approx',alternative='two-sided',zero_method='wilcox').pvalue)
            tests.append(dict(contrast=name,n_patients=len(p),median_delta=float(np.median(d)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),p_wilcoxon=pv,n_improved=int((d>0).sum())))
        pd.DataFrame(tests).to_csv(out/'paired_tests.tsv',sep='\t',index=False);verdict=dict(primary_pass=bool(tests[0]['median_delta']>0 and tests[0]['p_wilcoxon']<.05),primary=tests[0],completed_at=now());save(out/'verdict.json',verdict)
        save(out/'protected_model_hashes.json',{v['path']:v['sha256'] for v in inputs.values() if '/models/' in v['path']});assert all(sha(k)==v for k,v in json.loads((out/'protected_model_hashes.json').read_text()).items())
        (out/'RESULTS.md').write_text('# A15 results\n\n```text\n'+pd.DataFrame(tests).to_string(index=False)+'\n```\n\nPrimary criterion passed: '+str(verdict['primary_pass'])+'. See the fixed configuration, sealed TCGA reference, input manifest, per-tile replay checks and the original region-mask checks.\n')
        state(stage='complete',completed_slides=66);(out/'COMPLETE').write_text(now());log('A15 COMPLETE')
    except Exception as exc:
        state(stage='error',error=str(exc),traceback=traceback.format_exc());log(traceback.format_exc());raise

if __name__=='__main__':main()
