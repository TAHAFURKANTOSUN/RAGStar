#!/usr/bin/env bash
# VectorDBBench -> lokal OpenSearch benchmark koşusu (Linux / macOS)
# Ön koşullar: build_dataset.py -> patch_vdbb.py -> install_dataset.py
#
# Kullanım:
#   ./benchmark/run_opensearch.sh
#   DRY_RUN=1 ./benchmark/run_opensearch.sh
#   SWEEP=1 ./benchmark/run_opensearch.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
META="$HERE/dataset/dataset_meta.json"

[[ -f "$META" ]] || { echo "HATA: $META yok. Once: python benchmark/build_dataset.py"; exit 1; }

# shellcheck disable=SC1091
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a

OS_HOST="${OS_HOST:-localhost}"
OS_PORT="${OS_PORT:-9200}"
OPENSEARCH_USER="${OPENSEARCH_USER:-admin}"
: "${OPENSEARCH_PASSWORD:?OPENSEARCH_PASSWORD tanimli degil}"

# OpenSearch dev kurulumu 9200'de HTTPS + self-signed konusur.
export OSS_OPENSEARCH_USE_SSL="${OSS_OPENSEARCH_USE_SSL:-true}"
export OSS_OPENSEARCH_VERIFY_CERTS="${OSS_OPENSEARCH_VERIFY_CERTS:-false}"

jqv() {
  python3 -c "import json,sys
m=json.load(open('$META'))
if '$1' not in m:
    sys.stderr.write(\"HATA: dataset_meta.json'da '$1' yok. Once: python benchmark/install_dataset.py\n\"); sys.exit(1)
print(m['$1'])"
}
SIZE=$(jqv size); DIM=$(jqv dim); METRIC=$(jqv metric_type); GT_K=$(jqv gt_k)
METRIC_LOWER=$(echo "$METRIC" | tr '[:upper:]' '[:lower:]')

DATASET_NAME="${DATASET_NAME:-$(jqv dataset_name)}"
DATASET_DIR="${DATASET_DIR:-$(jqv dataset_dir)}"
K="${K:-10}"
M_DEFAULT="${M:-16}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-256}"
EF_SEARCH_DEFAULT="${EF_SEARCH:-100}"
ENGINE="${ENGINE:-faiss}"
NUM_CONCURRENCY="${NUM_CONCURRENCY:-1,4,8}"
CONCURRENCY_DURATION="${CONCURRENCY_DURATION:-30}"

if (( K > GT_K )); then
  echo "HATA: K=$K ama ground truth $GT_K komsu iceriyor. K'yi dusur veya --gt-k $K ile yeniden uret."
  exit 1
fi

echo
echo "=== Benchmark yapilandirmasi ==="
echo "  hedef     : $OS_HOST:$OS_PORT (ssl=$OSS_OPENSEARCH_USE_SSL)"
echo "  dataset   : $SIZE vektor x $DIM boyut, metric=$METRIC"
echo "  arama     : k=$K M=$M_DEFAULT ef_construction=$EF_CONSTRUCTION engine=$ENGINE"
echo

run_bench() {
  local m="$1" efc="$2" efs="$3" label="$4"
  local extra=()
  [[ "${DRY_RUN:-0}" == "1" ]] && extra+=("--dry-run")

  echo ">>> $label"
  vectordbbench ossopensearch \
    --host "$OS_HOST" --port "$OS_PORT" \
    --user "$OPENSEARCH_USER" --password "$OPENSEARCH_PASSWORD" \
    --db-label "$label" \
    --case-type PerformanceCustomDataset \
    --custom-case-name "$DATASET_NAME-custom" \
    --custom-case-description "DeneyselRAG kendi PDF korpusu" \
    --custom-dataset-name "$DATASET_NAME" \
    --custom-dataset-dir "$DATASET_DIR" \
    --custom-dataset-size "$SIZE" \
    --custom-dataset-dim "$DIM" \
    --custom-dataset-metric-type "$METRIC" \
    --custom-dataset-file-count 1 \
    --custom-dataset-with-gt \
    --skip-custom-dataset-use-shuffled \
    --metric-type "$METRIC_LOWER" \
    --engine "$ENGINE" \
    --m "$m" --ef-construction "$efc" --ef-search "$efs" \
    --k "$K" \
    --num-concurrency "$NUM_CONCURRENCY" \
    --concurrency-duration "$CONCURRENCY_DURATION" \
    --number-of-shards 1 --number-of-replicas 0 \
    --drop-old \
    "${extra[@]}" || echo "UYARI: kosu basarisiz -> $label"
  echo
}

if [[ "${SWEEP:-0}" == "1" ]]; then
  for m in 8 16 32; do
    for efs in 32 64 128 256; do
      run_bench "$m" "$EF_CONSTRUCTION" "$efs" "os-m${m}-efs${efs}"
    done
  done
else
  run_bench "$M_DEFAULT" "$EF_CONSTRUCTION" "$EF_SEARCH_DEFAULT" "os-m${M_DEFAULT}-efs${EF_SEARCH_DEFAULT}"
fi

echo "Sonuclar: vectordb_bench/results/ altinda JSON."
echo "Streamlit arayuzu:  init_bench"
