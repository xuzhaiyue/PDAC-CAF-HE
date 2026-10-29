# 修订 A7：H&E 预测的 CAF 状态与总生存（规则见 config/fig1.yaml 的 A7，计算前已冻结）。
# 预测来源：A2（I1、B2）与 A1_ownTCGA（岭回归）；临床：results/fig2/{tcga,cptac}_clinical.tsv
# 输出：results/he/A7/{per_cohort.tsv, pooled.tsv}
suppressMessages({library(survival); library(metafor)})
he <- "results/he"; out <- file.path(he, "A7"); dir.create(out, showWarnings = FALSE)
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
preds <- c("program_resid", "program_sub", "pan_CAF", "program", "Dominguez_LRRC15_sub", "Moffitt_activated_sub")

# ---- TCGA out-of-fold predictions (patient x target), per model
a2 <- read.delim(file.path(he, "A2/oof_tcga.tsv"))
a1 <- read.delim(file.path(he, "A1_ownTCGA/oof_tcga.tsv"))
wide <- function(d) { w <- reshape(d[, c("patient", "target", "y_pred")], idvar = "patient", timevar = "target", direction = "wide")
  names(w) <- sub("^y_pred\\.", "", names(w)); rownames(w) <- w$patient; w }
a1m <- aggregate(y_pred ~ patient + target, a1, mean)
tcga <- list(I1 = wide(a2[a2$model == "I1", ]), B2 = wide(a2[a2$model == "B2", ]), Ridge = wide(a1m))
truth_t <- wide(transform(a2[a2$model == "I1", ], y_pred = y_true))           # bulk-measured (fold-standardised) labels
# ---- CPTAC external predictions
pc <- read.delim(file.path(he, "A2/pred_cptac.tsv"), check.names = FALSE); names(pc)[1] <- "patient"
cpr <- read.delim(file.path(he, "A1_ownTCGA/pred_cptac.tsv"), check.names = FALSE); names(cpr)[1] <- "patient"
lab_c <- read.delim(file.path(he, "A1_ownTCGA/labels_cptac.tsv"), check.names = FALSE); names(lab_c)[1] <- "patient"
cptac <- list(I1 = pc[pc$model == "I1", ], B2 = pc[pc$model == "B2", ], Ridge = cpr)
for (k in names(cptac)) rownames(cptac[[k]]) <- cptac[[k]]$patient
rownames(lab_c) <- lab_c$patient

clin <- list(tcga = read.delim("results/fig2/tcga_clinical.tsv", row.names = 1),
             cptac = read.delim("results/fig2/cptac_clinical.tsv", row.names = 1))

fit <- function(cl, x, multi) {
  d <- data.frame(t = cl$OS_time, e = cl$OS, x = z(x), age = z(cl$age), stage = cl$stage_late)
  d <- d[!is.na(d$t) & d$t > 0 & !is.na(d$e) & !is.na(d$x), ]
  f <- if (multi) coxph(Surv(t, e) ~ x + age + stage, data = d) else coxph(Surv(t, e) ~ x, data = d)
  c(logHR = unname(coef(f)["x"]), se = sqrt(vcov(f)["x", "x"]), n = f$n, events = f$nevent)
}
rows <- list()
add <- function(cohort, source, P, cl) for (v in preds) if (v %in% names(P)) for (mv in c(FALSE, TRUE)) {
  pts <- intersect(rownames(P), rownames(cl)); r <- fit(cl[pts, ], P[pts, v], mv)
  rows[[length(rows) + 1]] <<- data.frame(cohort, source, predictor = v, model = ifelse(mv, "multivariable", "univariable"), t(r))
}
for (k in names(tcga)) add("TCGA", paste0("H&E ", k), tcga[[k]], clin$tcga)
for (k in names(cptac)) add("CPTAC", paste0("H&E ", k), cptac[[k]], clin$cptac)
add("TCGA", "bulk measured", truth_t, clin$tcga)
add("CPTAC", "bulk measured", lab_c, clin$cptac)
per <- do.call(rbind, rows)
per$HR <- exp(per$logHR); per$lo <- exp(per$logHR - 1.96 * per$se); per$hi <- exp(per$logHR + 1.96 * per$se)
per$p <- 2 * pnorm(-abs(per$logHR / per$se))
write.table(per, file.path(out, "per_cohort.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

pool <- list()
for (s in unique(per$source)) for (v in preds) for (m in c("univariable", "multivariable")) {
  d <- per[per$source == s & per$predictor == v & per$model == m, ]
  if (nrow(d) != 2) next
  fe <- rma(yi = d$logHR, sei = d$se, method = "FE"); re <- rma(yi = d$logHR, sei = d$se, method = "REML")
  pool[[length(pool) + 1]] <- data.frame(source = s, predictor = v, model = m, n = sum(d$n), events = sum(d$events),
    HR_FE = exp(fe$b[1]), lo = exp(fe$ci.lb), hi = exp(fe$ci.ub), p_FE = fe$pval, HR_RE = exp(re$b[1]), p_RE = re$pval, I2 = re$I2)
}
pool <- do.call(rbind, pool)
write.table(pool, file.path(out, "pooled.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
options(width = 220)
cat("== per cohort (primary predictor program_resid)\n")
print(per[per$predictor == "program_resid", c("cohort", "source", "model", "n", "events", "HR", "lo", "hi", "p")], digits = 3, row.names = FALSE)
cat("\n== pooled TCGA + CPTAC (fixed effect)\n")
print(pool[, c("source", "predictor", "model", "n", "events", "HR_FE", "lo", "hi", "p_FE", "p_RE", "I2")], digits = 3, row.names = FALSE)
v <- pool[pool$source == "H&E I1" & pool$predictor == "program_resid", ]
cat(sprintf("\nVERDICT (H&E I1, program_resid): univariable HR %.2f (%.2f-%.2f) p=%.3g -> %s; multivariable HR %.2f p=%.3g\n",
  v$HR_FE[v$model == "univariable"], v$lo[v$model == "univariable"], v$hi[v$model == "univariable"], v$p_FE[v$model == "univariable"],
  ifelse(v$HR_FE[v$model == "univariable"] > 1 & v$p_FE[v$model == "univariable"] < 0.05, "PASS", "FAIL"),
  v$HR_FE[v$model == "multivariable"], v$p_FE[v$model == "multivariable"]))
