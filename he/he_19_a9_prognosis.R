# 修订 A9（次要终点）：A9 胰腺癌空间微调模型预测的 CAF 状态与总生存，规则同 A7（单因素、年龄 + 分期多因素、TCGA + CPTAC 固定效应合并）。
# 预测来源：results/he/A9/patient_preds_{TCGA,CPTAC}_{all70,own4_only,primary_only}.tsv（模型未见过 TCGA / CPTAC，两队列均为外部预测）
#           对照：A8 finetuned_all8（results/he/A8/patient_preds_finetuned_{cohort}.tsv）
# 输出：results/he/A9/prognosis_{per_cohort,pooled}.tsv
suppressMessages({library(survival); library(metafor)})
he <- "results/he"; out <- file.path(he, "A9")
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
clin <- list(TCGA = read.delim("results/fig2/tcga_clinical.tsv", row.names = 1),
             CPTAC = read.delim("results/fig2/cptac_clinical.tsv", row.names = 1))
src <- list(A9_all70 = "A9/patient_preds_%s_all70.tsv", A9_own4_only = "A9/patient_preds_%s_own4_only.tsv",
            A9_primary_only = "A9/patient_preds_%s_primary_only.tsv", A8_finetuned_all8 = "A8/patient_preds_finetuned_%s.tsv")

fit <- function(cl, x, multi) {
  d <- data.frame(t = cl$OS_time, e = cl$OS, x = z(x), age = z(cl$age), stage = cl$stage_late)
  d <- d[!is.na(d$t) & d$t > 0 & !is.na(d$e) & !is.na(d$x), ]
  f <- if (multi) coxph(Surv(t, e) ~ x + age + stage, data = d) else coxph(Surv(t, e) ~ x, data = d)
  c(logHR = unname(coef(f)["x"]), se = sqrt(vcov(f)["x", "x"]), n = f$n, events = f$nevent)
}
rows <- list()
for (s in names(src)) for (co in names(clin)) {
  P <- read.delim(file.path(he, sprintf(src[[s]], co))); rownames(P) <- P$patient
  pts <- intersect(rownames(P), rownames(clin[[co]]))
  for (mv in c(FALSE, TRUE)) rows[[length(rows) + 1]] <- data.frame(cohort = co, source = s,
    model = ifelse(mv, "multivariable", "univariable"), t(fit(clin[[co]][pts, ], P[pts, "pred_state"], mv)))
}
per <- do.call(rbind, rows)
per$HR <- exp(per$logHR); per$lo <- exp(per$logHR - 1.96 * per$se); per$hi <- exp(per$logHR + 1.96 * per$se)
per$p <- 2 * pnorm(-abs(per$logHR / per$se))
write.table(per, file.path(out, "prognosis_per_cohort.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
pool <- list()
for (s in names(src)) for (m in c("univariable", "multivariable")) {
  d <- per[per$source == s & per$model == m, ]
  fe <- rma(yi = d$logHR, sei = d$se, method = "FE"); re <- rma(yi = d$logHR, sei = d$se, method = "REML")
  pool[[length(pool) + 1]] <- data.frame(source = s, model = m, n = sum(d$n), events = sum(d$events),
    HR_FE = exp(fe$b[1]), lo = exp(fe$ci.lb), hi = exp(fe$ci.ub), p_FE = fe$pval, p_RE = re$pval, I2 = re$I2)
}
pool <- do.call(rbind, pool)
write.table(pool, file.path(out, "prognosis_pooled.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
options(width = 220)
print(per[, c("cohort", "source", "model", "n", "events", "HR", "lo", "hi", "p")], digits = 3, row.names = FALSE)
cat("\n== pooled TCGA + CPTAC (fixed effect)\n"); print(pool, digits = 3, row.names = FALSE)
v <- pool[pool$source == "A9_all70" & pool$model == "univariable", ]
cat(sprintf("\nVERDICT (A9 all70): univariable HR %.2f (%.2f-%.2f) p=%.3g -> %s\n", v$HR_FE, v$lo, v$hi, v$p_FE,
            ifelse(v$HR_FE > 1 & v$p_FE < 0.05, "PASS", "FAIL")))
