# Multi-cohort survival meta-analysis (rules in the fig2_meta section of config/fig1.yaml, fixed before computation).
# 1) ESTIMATE stromal scores for GEO cohorts; 2) per-cohort Cox (univariable primary; secondary model + ESTIMATE stroma + basal−classical);
# 3) random-effects (REML) and fixed-effect pooling, I². TCGA / CPTAC are recomputed with the same models and pooled together.
suppressMessages({library(survival); library(metafor); library(estimate)})
out <- "results/fig2"; geo <- file.path(out, "geo")
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
platform <- c(GSE28735 = "affymetrix", GSE62452 = "affymetrix", GSE57495 = "affymetrix", GSE85916 = "affymetrix",
              GSE71729 = "agilent", GSE21501 = "agilent")
scores_all <- c("program_rel", "program", "CTHRC1_alone", "Moffitt_activated", "Moffitt_activated_rel",
                "Dominguez_LRRC15", "Dominguez_LRRC15_rel", "pan_CAF", "Moffitt_basal_minus_classical")

run_estimate <- function(expr_file, plat) {
  e <- read.delim(gzfile(expr_file), check.names = FALSE); colnames(e)[1] <- "GeneSymbol"
  tin <- tempfile(); tg <- tempfile(); ts <- tempfile()
  write.table(e, tin, sep = "\t", quote = FALSE, row.names = FALSE)
  invisible(capture.output(filterCommonGenes(input.f = tin, output.f = tg, id = "GeneSymbol")))
  invisible(capture.output(estimateScore(input.ds = tg, output.ds = ts, platform = plat)))
  s <- read.delim(ts, skip = 2, check.names = FALSE)
  v <- as.numeric(s[s$NAME == "StromalScore", -(1:2)]); names(v) <- colnames(e)[-1]; v
}

load <- function(co) {
  if (co %in% c("tcga", "cptac")) {
    s <- read.delim(file.path(out, paste0(co, "_scores.tsv")), row.names = 1, check.names = FALSE)
    cl <- read.delim(file.path(out, paste0(co, "_clinical.tsv")), row.names = 1, check.names = FALSE)
    es <- read.delim(file.path(out, paste0(co, "_estimate.tsv")), row.names = 1, check.names = FALSE)
    d <- cbind(s, cl[rownames(s), c("OS", "OS_time")], StromalScore = es[rownames(s), "StromalScore"])
    d$OS_time <- d$OS_time / 30.44
  } else {
    s <- read.delim(file.path(geo, paste0(co, "_scores.tsv")), row.names = 1, check.names = FALSE)
    cl <- read.delim(file.path(geo, paste0(co, "_clinical.tsv")), row.names = 1, check.names = FALSE)
    st <- run_estimate(file.path(geo, paste0(co, "_expr.tsv.gz")), platform[[co]])
    d <- cbind(s, cl[rownames(s), c("OS", "OS_time")], StromalScore = st[rownames(s)])
  }
  d <- d[!is.na(d$OS_time) & d$OS_time > 0 & !is.na(d$OS), ]
  d$cohort <- co; d
}

cohorts <- c("tcga", "cptac", "GSE28735", "GSE62452", "GSE57495", "GSE71729", "GSE85916", "GSE21501")
rows <- list()
for (co in cohorts) {
  d <- load(co)
  d$stroma_z <- z(d$StromalScore); d$bc_z <- z(d$Moffitt_basal_minus_classical)
  for (sc in scores_all) {
    d$x <- z(d[[sc]])
    for (mn in c("univariable", "secondary")) {
      f <- if (mn == "univariable") Surv(OS_time, OS) ~ x else
        if (sc == "Moffitt_basal_minus_classical") Surv(OS_time, OS) ~ x + stroma_z else Surv(OS_time, OS) ~ x + stroma_z + bc_z
      fit <- coxph(f, data = d)
      rows[[length(rows) + 1]] <- data.frame(cohort = co, score = sc, model = mn, n = fit$n, events = fit$nevent,
                                             logHR = coef(fit)[["x"]], se = sqrt(vcov(fit)["x", "x"]))
    }
  }
}
per <- do.call(rbind, rows)
per$HR <- exp(per$logHR); per$p <- 2 * pnorm(-abs(per$logHR / per$se))
write.table(per, file.path(out, "meta_per_cohort.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

meta <- list()
for (sc in scores_all) for (mn in c("univariable", "secondary")) for (excl in c("none", "GSE71729")) {
  p <- per[per$score == sc & per$model == mn & per$cohort != excl, ]
  re <- rma(yi = p$logHR, sei = p$se, method = "REML")
  fe <- rma(yi = p$logHR, sei = p$se, method = "FE")
  meta[[length(meta) + 1]] <- data.frame(score = sc, model = mn, excluded = excl, k = nrow(p), n = sum(p$n), events = sum(p$events),
                                         HR_RE = exp(re$b[1]), lo_RE = exp(re$ci.lb), hi_RE = exp(re$ci.ub), p_RE = re$pval,
                                         HR_FE = exp(fe$b[1]), p_FE = fe$pval, I2 = re$I2)
}
meta <- do.call(rbind, meta)
write.table(meta, file.path(out, "meta_pooled.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
options(width = 220)
cat("\n== per cohort, program_rel / program (univariable)\n")
print(per[per$score %in% c("program_rel", "program") & per$model == "univariable", c("cohort", "score", "n", "events", "HR", "p")], digits = 3, row.names = FALSE)
cat("\n== pooled (all cohorts)\n")
print(meta[meta$excluded == "none", ], digits = 3, row.names = FALSE)
cat("\n== pooled excluding GSE71729 (Moffitt discovery cohort)\n")
print(meta[meta$excluded == "GSE71729" & meta$model == "univariable", ], digits = 3, row.names = FALSE)
v <- meta[meta$score == "program_rel" & meta$excluded == "none", ]
cat(sprintf("\nVERDICT program_rel: univariable RE HR %.2f (%.2f-%.2f) p=%.3g -> %s; secondary RE HR %.2f p=%.3g\n",
            v$HR_RE[v$model == "univariable"], v$lo_RE[v$model == "univariable"], v$hi_RE[v$model == "univariable"],
            v$p_RE[v$model == "univariable"], ifelse(v$p_RE[v$model == "univariable"] < 0.05, "PASS", "FAIL"),
            v$HR_RE[v$model == "secondary"], v$p_RE[v$model == "secondary"]))
