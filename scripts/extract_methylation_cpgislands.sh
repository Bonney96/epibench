#!/bin/bash
set -euo pipefail

# --- User-editable paths ---
CPG_ISLANDS_BED="/storage1/fs1/dspencer/Active/spencerlab/abonney/CpG_islands/hg38_cpgislands_filtered.bed"
DATA_DIR="/storage2/fs1/dspencer/Active/spencerlab/data/wgbs"
OUT_DIR="/storage2/fs1/dspencer/Active/spencerlab/abonney/epibench/methylation_extracted"
LOG_FILE="methfast_extraction.log"

# --- Sample list ---
declare -A SAMPLE_PATHS=(
  [RO04050]="RO04050-CD34-wgbs"
  [RO04046]="RO04046-CD34-wgbs"
  [RO04068]="RO04068-CD34-wgbs"
  [263578]="263578-dx-wgbs"
  [463352]="463352-dx-wgbs"
  [847670]="847670-dx-wgbs"
)

# --- Methfast parameters ---
METHFAST_PARAMS="-f 4 -c 5 -m 6 -u 7"

# --- Create output and log directories ---
mkdir -p "$OUT_DIR"
echo "Methfast extraction started at $(date)" > "$LOG_FILE"

# --- Main loop ---
for SAMPLE in "${!SAMPLE_PATHS[@]}"; do
  DIR="${SAMPLE_PATHS[$SAMPLE]}"
  SAMPLE_BED="$DATA_DIR/$DIR/$DIR.meth.bed.gz"
  SAMPLE_OUTDIR="$OUT_DIR/$SAMPLE"
  mkdir -p "$SAMPLE_OUTDIR"

  echo "[$(date)] Processing $SAMPLE..." | tee -a "$LOG_FILE"

  if [[ ! -f "$SAMPLE_BED" ]]; then
    echo "[$(date)] ERROR: Input file not found: $SAMPLE_BED" | tee -a "$LOG_FILE"
    continue
  fi

  OUTFILE="$SAMPLE_OUTDIR/${SAMPLE}_cpgislands_methylation.tsv"
  
  # Log the command structure as it will be effectively run
  LOGGED_CMD_STR="/storage2/fs1/dspencer/Active/spencerlab/abonney/methfast/methfast \\"$SAMPLE_BED\\" \\"$CPG_ISLANDS_BED\\" $METHFAST_PARAMS > \\"$OUTFILE\\""
  echo "[$(date)] Running: $LOGGED_CMD_STR" | tee -a "$LOG_FILE"

  # Execute the command:
  # methfast's standard output goes to $OUTFILE
  # methfast's standard error is appended to $LOG_FILE
  if /storage2/fs1/dspencer/Active/spencerlab/abonney/methfast/methfast "$SAMPLE_BED" "$CPG_ISLANDS_BED" $METHFAST_PARAMS > "$OUTFILE" 2>> "$LOG_FILE"; then
    echo "[$(date)] SUCCESS: $SAMPLE extraction complete." | tee -a "$LOG_FILE"
  else
    # The specific error from methfast should now be in $LOG_FILE
    echo "[$(date)] ERROR: Methfast failed for $SAMPLE. See $LOG_FILE for details from methfast itself." | tee -a "$LOG_FILE"
  fi
done

echo "Methfast extraction finished at $(date)" >> "$LOG_FILE"