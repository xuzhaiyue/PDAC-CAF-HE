#!/usr/bin/env bash
# 清理后的间质层 → CAF 层：每轮整合后按 config.cluster_exclusion 规则标记，排除后重跑，直至无亚群被标记。
# 每一轮的 cluster_flags.tsv 都保留，用于补充材料。
set -euo pipefail
cd "$(dirname "$0")/.."
SCVI=~/miniforge3/envs/pdac_scvi/bin/python
HEM=~/miniforge3/envs/hemorph/bin/python
R=results/fig1

flag() {  # $1 = integrated dir; prints the exclude list (empty if none)
  $HEM fig1/fig1_08_clean.py flag "$R/$1/fibro_integrated.h5ad" 2>/dev/null | grep '^--exclude' | awk '{print $2}' | sed 's/(none)//'
}

echo "== round 0: mesenchymal (clean cells)"
nice -n 10 $SCVI fig1/fig1_02_integrate_cluster.py --input mesenchymal_clean_raw.h5ad --mode caf \
  --out integrate_mes_clean > $R/integrate_mes_clean_run.log 2>&1
prev=integrate_mes_clean
ex=$(flag $prev); echo "   flagged: ${ex:-none}"

i=1
while [[ -n "$ex" && $i -le 4 ]]; do
  out=integrate_caf_clean_r$i
  echo "== round $i: exclude [$ex] from $prev -> $out"
  nice -n 10 $SCVI fig1/fig1_02_integrate_cluster.py --input mesenchymal_clean_raw.h5ad --mode caf \
    --out $out --exclude-from $R/$prev/fibro_integrated.h5ad --exclude "$ex" > $R/${out}_run.log 2>&1
  prev=$out
  ex=$(flag $prev); echo "   flagged: ${ex:-none}"
  i=$((i + 1))
done
echo "final CAF atlas: $R/$prev"
ln -sfn "$prev" $R/integrate_caf_final
