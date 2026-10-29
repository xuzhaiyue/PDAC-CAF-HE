# Figure 2 预设分析（规则见 config/fig1.yaml 的 fig2，读取生存数据前已冻结）。
# A 相对状态与基质多少的关系；B 总生存 Cox（主要）；C 各比较得分的同模型比较；D 肿瘤内在特征关联（探索性）
# 输出：results/fig2/{A_decoupling,B_cox,C_head_to_head,D_biology,verdict}.tsv
suppressMessages(library(survival))
out <- "results/fig2"
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
score_cols <- c("program_rel", "program", "CTHRC1_alone", "Moffitt_activated", "Moffitt_activated_rel",
                "Dominguez_LRRC15", "Dominguez_LRRC15_rel", "pan_CAF")

load_cohort <- function(cohort) {
  s <- read.delim(file.path(out, paste0(cohort, "_scores.tsv")), check.names = FALSE, row.names = 1)
  cl <- read.delim(file.path(out, paste0(cohort, "_clinical.tsv")), check.names = FALSE, row.names = 1)
  es <- read.delim(file.path(out, paste0(cohort, "_estimate.tsv")), check.names = FALSE, row.names = 1)
  d <- cbind(s, cl[rownames(s), ], es[rownames(s), c("StromalScore", "ImmuneScore")])
  d$cohort <- cohort
  d
}

A <- list(); B <- list(); C <- list(); D <- list(); V <- list()
for (cohort in c("tcga", "cptac")) {
  d <- load_cohort(cohort)
  # ---------------- A. decoupling
  against <- c("StromalScore", "pan_CAF", "Stromal_fraction", "Neoplastic_cellularity")
  against <- against[against %in% colnames(d)]
  for (sc in score_cols) for (ag in against) {
    ok <- complete.cases(d[, c(sc, ag)])
    ct <- cor.test(d[ok, sc], d[ok, ag], method = "spearman", exact = FALSE)
    A[[length(A) + 1]] <- data.frame(cohort, score = sc, against = ag, n = sum(ok), rho = unname(ct$estimate), p = ct$p.value)
  }
  # ---------------- B / C. Cox
  d$age_z <- z(d$age); d$stroma_z <- z(d$StromalScore); d$bc_z <- z(d$Moffitt_basal_minus_classical)
  if ("Stromal_fraction" %in% colnames(d)) d$pathstroma_z <- z(d$Stromal_fraction)
  endpoints <- if (cohort == "tcga") list(OS = c("OS_time", "OS"), PFI = c("PFI_time", "PFI")) else list(OS = c("OS_time", "OS"))
  for (ep in names(endpoints)) {
    tt <- endpoints[[ep]][1]; ev <- endpoints[[ep]][2]
    for (sc in score_cols) {
      d$x <- z(d[[sc]])
      models <- list(univariable = "x", multivariable = "x + age_z + stage_late + stroma_z + bc_z")
      if (cohort == "cptac") models$multivariable_pathstroma <- "x + age_z + stage_late + stroma_z + bc_z + pathstroma_z"
      for (mn in names(models)) {
        f <- as.formula(paste0("Surv(", tt, ", ", ev, ") ~ ", models[[mn]]))
        dd <- d[d[[tt]] > 0 & !is.na(d[[tt]]), ]
        fit <- coxph(f, data = dd)
        sm <- summary(fit)
        row <- data.frame(cohort, endpoint = ep, score = sc, model = mn, n = fit$n, events = fit$nevent,
                          HR = sm$conf.int["x", "exp(coef)"], lo = sm$conf.int["x", "lower .95"],
                          hi = sm$conf.int["x", "upper .95"], p = sm$coefficients["x", "Pr(>|z|)"],
                          c_index = unname(sm$concordance[1]))
        if (sc == "program_rel") B[[length(B) + 1]] <- row
        C[[length(C) + 1]] <- row
      }
    }
  }
  # ---------------- D. biology (exploratory)
  for (it in c("Moffitt_basal_minus_classical", "MPAS", "StromalScore", "ImmuneScore")) {
    ct <- cor.test(d$program_rel, d[[it]], method = "spearman", exact = FALSE)
    D[[length(D) + 1]] <- data.frame(cohort, item = it, test = "spearman", n = sum(complete.cases(d[, c("program_rel", it)])),
                                     stat = unname(ct$estimate), p = ct$p.value)
  }
  k <- d[!is.na(d$KRAS_allele), ]
  k$mut <- k$KRAS_allele != "WT"
  if (length(unique(k$mut)) == 2) {
    w <- wilcox.test(program_rel ~ mut, data = k)
    D[[length(D) + 1]] <- data.frame(cohort, item = "KRAS mutant vs WT", test = "wilcoxon", n = nrow(k),
                                     stat = median(k$program_rel[k$mut]) - median(k$program_rel[!k$mut]), p = w$p.value)
  }
  k$allele3 <- ifelse(k$KRAS_allele %in% c("G12D", "G12V", "G12R"), k$KRAS_allele, NA)
  kk <- k[!is.na(k$allele3), ]
  kw <- kruskal.test(program_rel ~ allele3, data = kk)
  D[[length(D) + 1]] <- data.frame(cohort, item = "KRAS G12D/G12V/G12R", test = "kruskal", n = nrow(kk),
                                   stat = unname(kw$statistic), p = kw$p.value)
}
A <- do.call(rbind, A); B <- do.call(rbind, B); C <- do.call(rbind, C); D <- do.call(rbind, D)
write.table(A, file.path(out, "A_decoupling.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(B, file.path(out, "B_cox_program_rel.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(C, file.path(out, "C_head_to_head.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(D, file.path(out, "D_biology.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# ---------------- pre-specified verdicts
a <- function(co, sc, ag) A$rho[A$cohort == co & A$score == sc & A$against == ag]
for (co in c("tcga", "cptac")) for (ag in c("StromalScore", if (co == "cptac") "Stromal_fraction")) {
  V[[length(V) + 1]] <- data.frame(check = paste0("A ", co, " vs ", ag),
                                   value = sprintf("rel %.2f | raw %.2f", a(co, "program_rel", ag), a(co, "program", ag)),
                                   pass = abs(a(co, "program_rel", ag)) <= 0.3 & abs(a(co, "program_rel", ag)) < abs(a(co, "program", ag)))
}
b <- function(co) B[B$cohort == co & B$endpoint == "OS" & B$model == "multivariable", ]
for (co in c("tcga", "cptac")) {
  r <- b(co)
  V[[length(V) + 1]] <- data.frame(check = paste0("B ", co, " OS multivariable"),
                                   value = sprintf("HR %.2f (%.2f-%.2f), p=%.3g", r$HR, r$lo, r$hi, r$p),
                                   pass = r$HR > 1 & r$p < 0.05)
}
V <- do.call(rbind, V)
write.table(V, file.path(out, "verdict.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
options(width = 200)
cat("\n== A decoupling (program_rel / program)\n"); print(A[A$score %in% c("program_rel", "program"), ], digits = 2, row.names = FALSE)
cat("\n== B Cox program_rel\n"); print(B, digits = 3, row.names = FALSE)
cat("\n== C head-to-head (OS, multivariable)\n"); print(C[C$endpoint == "OS" & C$model == "multivariable", ], digits = 3, row.names = FALSE)
cat("\n== D biology\n"); print(D, digits = 3, row.names = FALSE)
cat("\n== VERDICT\n"); print(V, row.names = FALSE)
