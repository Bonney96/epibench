#!/bin/bash
set -euo pipefail

# --- User-editable paths ---
CPG_ISLANDS_BED="/storage1/fs1/dspencer/Active/spencerlab/abonney/CpG_islands/cpgislands.bed"
DATA_DIR="/storage2/fs1/dspencer/Active/spencerlab/data/wgbs"
OUT_DIR="/storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/methylation_extracted"

declare -A SAMPLE_PATHS=(
  [RO04050]="RO04050-CD34-wgbs"
  [RO04046]="RO04046-CD34-wgbs"
  [RO04068]="RO04068-CD34-wgbs"
  [263578]="263578-dx-wgbs"
  [463352]="463352-dx-wgbs"
  [847670]="847670-dx-wgbs"
)

# Methfast expects: -f 4 (island name), -c 5 (coverage), -m 6 (meth), -u 7 (unmeth)
for SAMPLE in "${!SAMPLE_PATHS[@]}"; do
  DIR="${SAMPLE_PATHS[$SAMPLE]}"
  SAMPLE_BED="$DATA_DIR/$DIR/$DIR.meth.bed.gz"
  SAMPLE_OUTDIR="$OUT_DIR/$SAMPLE"
  OUTFILE="$SAMPLE_OUTDIR/${SAMPLE}_cpgislands_methylation.tsv"
  mkdir -p "$SAMPLE_OUTDIR"

  # Transform: add methylated and unmethylated counts as columns 6 and 7
  zcat "$SAMPLE_BED" | \
    awk 'BEGIN{OFS="\t"} {cov=$5; frac=$4; m=int(frac*cov+0.5); u=cov-m; print $1,$2,$3,$4,$5,m,u}' | \
    /storage2/fs1/dspencer/Active/spencerlab/abonney/methfast/methfast - "$CPG_ISLANDS_BED" -f 4 -c 5 -m 6 -u 7 > "$OUTFILE"
done