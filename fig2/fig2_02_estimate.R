# ESTIMATE stromal / immune scores (R estimate package). For RNA-seq only StromalScore / ImmuneScore are used;
# the ESTIMATE purity formula was fitted for Affymetrix arrays and does not apply to RNA-seq, so purity is not computed (recorded deviation from plan).
suppressMessages(library(estimate))
out <- "results/fig2"
for (cohort in c("tcga", "cptac")) {
  e <- read.delim(gzfile(file.path(out, paste0(cohort, "_expr.tsv.gz"))), check.names = FALSE)
  colnames(e)[1] <- "GeneSymbol"
  tmp_in <- tempfile(fileext = ".txt"); tmp_gct <- tempfile(fileext = ".gct"); tmp_sc <- tempfile(fileext = ".gct")
  write.table(e, tmp_in, sep = "\t", quote = FALSE, row.names = FALSE)
  filterCommonGenes(input.f = tmp_in, output.f = tmp_gct, id = "GeneSymbol")
  estimateScore(input.ds = tmp_gct, output.ds = tmp_sc, platform = "illumina")
  s <- read.delim(tmp_sc, skip = 2, check.names = FALSE)
  m <- t(s[, -(1:2)]); colnames(m) <- s$NAME
  m <- data.frame(sample = colnames(e)[-1], m, check.names = FALSE)
  write.table(m, file.path(out, paste0(cohort, "_estimate.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
  cat(cohort, nrow(m), "samples\n")
}
