# Post-hoc A11: conditional CAF-state prognosis with abundance adjustment.
suppressMessages({library(survival); library(metafor); library(estimate); library(yaml)})
options(warn=1)
args <- commandArgs(trailingOnly=TRUE)
quick <- '--quick' %in% args
value <- function(key, default) { i <- match(key,args); if(is.na(i)) default else args[i+1] }
root <- normalizePath(Sys.getenv('PROJECT_ROOT', unset=getwd()))
cfgpath <- file.path(root,'config/fig1.yaml'); cfg <- read_yaml(cfgpath)
rule <- cfg$he_amendments$A11_state_within_abundance_strata
stopifnot(startsWith(rule$status,'frozen_'))
out <- value('--output',file.path(root,rule$paths$output,format(Sys.time(),'%Y-%m-%d_%H%M%S')))
if(dir.exists(out)) stop('Output exists; use a new run directory')
dir.create(out,recursive=TRUE); dir.create(file.path(out,'estimate')); dir.create(file.path(out,'figures'))
file.copy(cfgpath,file.path(out,'run_config.yaml'))
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
set.seed(20261021); start <- Sys.time(); sources <- character(); warnings_log <- list(); fits <- list(); phrows <- list(); masks <- list(); metrics <- list(); all_data <- list()
write_tsv <- function(d,name) write.table(d,file.path(out,name),sep='\t',quote=FALSE,row.names=FALSE,na='NA')
gate <- function() {
 repeat {
  line <- system('sysctl -n vm.loadavg',intern=TRUE)
  load <- as.numeric(strsplit(gsub('[{}]','',line),' +')[[1]][2])
  if(!is.finite(load)) stop('Cannot read load average')
  if(load<=8) break
  cat(format(Sys.time()),'load',load,'>8; paused 15s\n'); flush.console(); Sys.sleep(15)
 }
}
z <- function(x) { s <- sd(x,na.rm=TRUE); if(!is.finite(s)||s==0) rep(NA_real_,length(x)) else (x-mean(x,na.rm=TRUE))/s }
read_source <- function(path, row_names=1) { sources <<- unique(c(sources,path)); read.delim(path,row.names=row_names,check.names=FALSE) }
source_root <- file.path(root,rule$paths$source)
cohorts <- c('tcga','cptac','GSE28735','GSE62452','GSE57495','GSE71729','GSE85916','GSE21501')
scores <- c('program_rel','Moffitt_activated_rel','Dominguez_LRRC15_rel')
platforms <- c(GSE28735='affymetrix',GSE62452='affymetrix',GSE57495='affymetrix',GSE85916='affymetrix',GSE71729='agilent',GSE21501='agilent')
get_estimate <- function(co,ids) {
 path <- file.path(source_root,'geo',paste0(co,'_expr.tsv.gz')); sources <<- unique(c(sources,path)); gate()
 e <- read.delim(gzfile(path),check.names=FALSE); colnames(e)[1] <- 'GeneSymbol'
 tin <- file.path(out,'estimate',paste0(co,'_input.tsv')); tg <- file.path(out,'estimate',paste0(co,'_common.gct')); ts <- file.path(out,'estimate',paste0(co,'_scores.gct'))
 write.table(e,tin,sep='\t',quote=FALSE,row.names=FALSE)
 invisible(capture.output(filterCommonGenes(input.f=tin,output.f=tg,id='GeneSymbol')))
 gate(); invisible(capture.output(estimateScore(input.ds=tg,output.ds=ts,platform=platforms[[co]])))
 s <- read.delim(ts,skip=2,check.names=FALSE)
 v <- as.numeric(s[s$NAME=='StromalScore',-(1:2)]); names(v) <- colnames(e)[-1]
 write_tsv(data.frame(sample=names(v),StromalScore=v),paste0('estimate/',co,'_estimate.tsv'))
 # Temporary expression copies are derived intermediates inside the new run only.
 unlink(tin)
 v[ids]
}
record_fit <- function(d,co,score,model,formula,terms) {
 gate(); vars <- all.vars(formula); keep <- complete.cases(d[,vars,drop=FALSE])
 masks[[length(masks)+1]] <<- data.frame(cohort=co,sample=d$sample,score=score,model=model,keep=keep,reason=ifelse(keep,'','missing_model_variable'))
 x <- d[keep,,drop=FALSE]; warn <- character(); error <- ''
 fit <- tryCatch(withCallingHandlers(coxph(formula,data=x,x=TRUE,y=TRUE,control=coxph.control(iter.max=50)),warning=function(w){warn <<- c(warn,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e){error <<- conditionMessage(e);NULL})
 bad <- is.null(fit)||nrow(x)==0||sum(x$OS)==0||any(grepl('infinite|converg|singular',warn,ignore.case=TRUE))
 if(!is.null(fit)) bad <- bad||any(!is.finite(coef(fit)))||any(!is.finite(diag(vcov(fit))))||any(diag(vcov(fit))<=0)
 reason <- paste(c(error,warn,if(nrow(x)>0&&sum(x$OS)==0)'no_events'),collapse=';')
 if(bad&&reason=='') reason <- 'nonfinite_or_singular_model'
 warnings_log[[length(warnings_log)+1]] <<- data.frame(cohort=co,score=score,model=model,status=ifelse(bad,'excluded','ok'),message=reason)
 if(!bad) {
  ph <- tryCatch(cox.zph(fit),error=function(e)NULL)
  if(!is.null(ph)) phrows[[length(phrows)+1]] <<- data.frame(cohort=co,score=score,model=model,term=rownames(ph$table),ph$table,row.names=NULL)
 }
 for(term in terms) {
  b <- if(!bad&&term %in% names(coef(fit))) unname(coef(fit)[term]) else NA_real_
  se <- if(is.finite(b)) sqrt(vcov(fit)[term,term]) else NA_real_
  fits[[length(fits)+1]] <<- data.frame(cohort=co,score=score,model=model,term=term,n=nrow(x),events=sum(x$OS),logHR=b,se=se,HR=exp(b),lo=exp(b-1.96*se),hi=exp(b+1.96*se),p=2*pnorm(-abs(b/se)),low_events=sum(x$OS)<10*max(1,length(if(is.null(fit)) terms else coef(fit))),status=ifelse(is.finite(b),'ok','excluded'))
 }
 invisible(fit)
}
if(quick) {
 d <- data.frame(sample=paste0('synthetic',1:120),OS_time=rexp(120),OS=rbinom(120,1,.7),state=rnorm(120),abundance=rnorm(120))
 record_fit(d,'synthetic','program_rel','joint',Surv(OS_time,OS)~state+abundance,c('state','abundance'))
 write_tsv(do.call(rbind,fits),'synthetic_checks.tsv');writeLines('Synthetic Cox smoke check passed; no real survival data used.',file.path(out,'README.md'));quit(status=0)
}
for(co in cohorts) {
 gate(); base <- if(co %in% c('tcga','cptac')) source_root else file.path(source_root,'geo')
 s <- read_source(file.path(base,paste0(co,'_scores.tsv'))); cl <- read_source(file.path(base,paste0(co,'_clinical.tsv')))
 if(anyDuplicated(rownames(s))||anyDuplicated(rownames(cl))) stop('Duplicate patient IDs')
 d <- cbind(s,cl[rownames(s),c('OS','OS_time')]);d$sample <- rownames(s); d$cohort <- co
 valid <- !is.na(d$OS_time)&d$OS_time>0&!is.na(d$OS)&d$OS %in% c(0,1)
 masks[[length(masks)+1]] <- data.frame(cohort=co,sample=d$sample,score='all',model='OS_eligibility',keep=valid,reason=ifelse(valid,'','missing_invalid_OS_or_time'))
 d <- d[valid,,drop=FALSE]; if(co %in% c('tcga','cptac')) d$OS_time <- d$OS_time/30.44
 d$abundance <- z(d$pan_CAF);d$bc_z <- z(d$Moffitt_basal_minus_classical)
 if(co %in% c('tcga','cptac')) { es <- read_source(file.path(source_root,paste0(co,'_estimate.tsv'))); d$stromal <- es[d$sample,'StromalScore'] } else d$stromal <- get_estimate(co,d$sample)
 d$stroma_z <- z(d$stromal); cut <- median(d$pan_CAF,na.rm=TRUE);d$abundance_group <- ifelse(d$pan_CAF>cut,'high','low')
 metrics[[length(metrics)+1]] <- data.frame(cohort=co,n=nrow(d),events=sum(d$OS),panCAF_median=cut,state_abundance_rho=cor(d$program_rel,d$pan_CAF,method='spearman',use='complete.obs'))
 for(sc in scores) {
  d$state <- z(d[[sc]]);raw <- sub('_rel$','',sc);d$raw_z <- z(d[[raw]])
  record_fit(d,co,sc,'joint',Surv(OS_time,OS)~state+abundance,c('state','abundance'))
  record_fit(d,co,sc,'state_only',Surv(OS_time,OS)~state,'state')
  record_fit(d,co,sc,'raw_only',Surv(OS_time,OS)~raw_z,'raw_z')
  record_fit(d,co,sc,'raw_joint',Surv(OS_time,OS)~raw_z+abundance,c('raw_z','abundance'))
  record_fit(d,co,sc,'interaction',Surv(OS_time,OS)~state*abundance,'state:abundance')
  record_fit(d,co,sc,'secondary',Surv(OS_time,OS)~state+abundance+stroma_z+bc_z,c('state','abundance'))
  for(group in c('low','high')) record_fit(d[d$abundance_group==group,],co,sc,paste0('stratum_',group),Surv(OS_time,OS)~state,'state')
  state_cut <- median(d[[sc]],na.rm=TRUE)
  d$four_group <- factor(paste0('state_',ifelse(d[[sc]]>state_cut,'high','low'),'_CAF_',d$abundance_group),levels=c('state_low_CAF_low','state_high_CAF_low','state_low_CAF_high','state_high_CAF_high'))
  km <- survfit(Surv(OS_time,OS)~four_group,data=d)
  draw_km <- function() {
   par(mar=c(4,4,3,1));plot(km,col=c('#768692','#C1666B','#457B9D','#8D2D3E'),lwd=1.5,mark.time=TRUE,xlab='Overall survival (months)',ylab='Survival probability',main=paste(co,sc),conf.int=FALSE)
   legend('topright',legend=levels(d$four_group),col=c('#768692','#C1666B','#457B9D','#8D2D3E'),lty=1,lwd=1.5,cex=.62,bty='n')
  }
  stem <- file.path(out,'figures',paste0('km_',co,'_',sc))
  pdf(paste0(stem,'.pdf'),width=6,height=4.5);draw_km();dev.off()
  png(paste0(stem,'.png'),width=1500,height=1125,res=250);draw_km();dev.off()
  svg(paste0(stem,'.svg'),width=6,height=4.5);draw_km();dev.off()
  sk <- summary(km);if(length(sk$time)) write_tsv(data.frame(time=sk$time,n_risk=sk$n.risk,n_event=sk$n.event,survival=sk$surv,stratum=sk$strata),paste0('km_',co,'_',sc,'.tsv'))
  all_data[[length(all_data)+1]] <- data.frame(cohort=co,sample=d$sample,score=sc,OS=d$OS,OS_time=d$OS_time,state=d$state,abundance=d$abundance,abundance_group=d$abundance_group,four_group=d$four_group,stroma_z=d$stroma_z,bc_z=d$bc_z)
 }
 cat('Completed',co,'n',nrow(d),'events',sum(d$OS),'\n');flush.console();Sys.sleep(.5)
}
per <- do.call(rbind,fits); pooled <- list()
for(sc in scores) for(mn in unique(per$model)) for(term in unique(per$term[per$model==mn])) for(ex in c('none','GSE71729')) {
 d <- per[per$score==sc&per$model==mn&per$term==term&per$cohort!=ex&per$status=='ok',]
 if(nrow(d)<2) next
 re <- rma(yi=d$logHR,sei=d$se,method='REML');fe <- rma(yi=d$logHR,sei=d$se,method='FE')
 pooled[[length(pooled)+1]] <- data.frame(score=sc,model=mn,term=term,excluded=ex,k=nrow(d),n=sum(d$n),events=sum(d$events),HR_RE=exp(re$b[1]),lo_RE=exp(re$ci.lb),hi_RE=exp(re$ci.ub),p_RE=re$pval,HR_FE=exp(fe$b[1]),lo_FE=exp(fe$ci.lb),hi_FE=exp(fe$ci.ub),p_FE=fe$pval,I2=re$I2)
}
pool <- do.call(rbind,pooled); all <- do.call(rbind,all_data)
for(sc in scores) {
 d <- all[all$score==sc,];d$four_group <- factor(d$four_group,levels=c('state_low_CAF_low','state_high_CAF_low','state_low_CAF_high','state_high_CAF_high'))
 record_fit(d,'pooled_stratified_baseline',sc,'four_groups',Surv(OS_time,OS)~four_group+strata(cohort),paste0('four_group',levels(d$four_group)[-1]))
}
write_tsv(do.call(rbind,fits),'per_cohort.tsv');write_tsv(pool,'pooled.tsv');write_tsv(do.call(rbind,masks),'inclusion_masks.tsv');write_tsv(do.call(rbind,warnings_log),'model_status.tsv');write_tsv(do.call(rbind,phrows),'ph_tests.tsv');write_tsv(do.call(rbind,metrics),'cohort_qc.tsv');write_tsv(all,'analysis_cells.tsv')
v <- pool[pool$score=='program_rel'&pool$model=='joint'&pool$term=='state'&pool$excluded=='none',]
strata <- pool[pool$score=='program_rel'&pool$model %in% c('stratum_low','stratum_high')&pool$excluded=='none',]
verdict <- data.frame(criterion=c('primary_conditional_state','secondary_both_strata_direction'),passed=c(nrow(v)==1&&v$HR_RE>1&&v$p_RE<.05,nrow(strata)==2&&all(strata$HR_RE>1)))
write_tsv(verdict,'verdict.tsv')
# Compare the existing single-score meta results to confirm cohort and scale identity.
old <- read_source(file.path(source_root,'meta_per_cohort.tsv'),row_names=NULL);ours <- per[per$model=='state_only'&per$term=='state',]
chk <- merge(ours,old[old$model=='univariable',],by=c('cohort','score'),suffixes=c('_new','_old'))
chk$delta_logHR <- chk$logHR_new-chk$logHR_old;write_tsv(chk,'legacy_reproduction.tsv')
if(nrow(chk)!=24||any(abs(chk$delta_logHR)>1e-7)) stop('Legacy state-only reproduction mismatch; review outputs')
write_tsv(data.frame(path=sources,md5=unname(tools::md5sum(sources))),'input_manifest.tsv')
writeLines(c('A11 post-hoc conditional CAF-state prognosis. See run_config.yaml for frozen rules.','Primary analysis is association conditional on a molecular pan-CAF proxy, not a causal masking test.','Z-scales are defined before strata; low/high strata need not individually be significant.','All prior analysis files are read-only. Figures are per-cohort descriptive KM; pooled group HR uses cohort-stratified Cox.',paste('Elapsed seconds:',as.numeric(difftime(Sys.time(),start,units='secs')))),file.path(out,'README.md'))
writeLines(capture.output(print(v)),file.path(out,'primary_summary.txt'));writeLines('complete',file.path(out,'COMPLETE'))
cat('A11 complete:',out,'\n');print(v);print(strata);print(verdict)
