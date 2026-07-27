<#
    VectorDBBench -> lokal OpenSearch benchmark koşusu (Windows / PowerShell)
    ========================================================================

    Ön koşullar:
      1. python benchmark/build_dataset.py    (RAG venv'inde)
      2. python benchmark/patch_vdbb.py       (bench venv'inde)
      3. python benchmark/install_dataset.py  (bench venv'inde)
      4. OpenSearch ayakta ve OPENSEARCH_USER / OPENSEARCH_PASSWORD tanımlı

    Kullanım:
      .\benchmark\run_opensearch.ps1
      .\benchmark\run_opensearch.ps1 -DryRun
      .\benchmark\run_opensearch.ps1 -Sweep          # M / ef_search taraması
      .\benchmark\run_opensearch.ps1 -K 10 -EfSearch 64
#>

param(
    [string] $OpenSearchHost = "localhost",
    [int]    $Port           = 9200,
    [switch] $NoSsl,                       # OpenSearch güvenlik eklentisi kapalıysa
    [string] $DatasetName    = "deneysel",
    [string] $DatasetDir     = "deneysel_bge_m3",
    [int]    $K              = 10,
    [int]    $M              = 16,
    [int]    $EfConstruction = 256,
    [int]    $EfSearch       = 100,
    [ValidateSet("faiss", "lucene")]
    [string] $Engine         = "faiss",
    [string] $NumConcurrency = "1,4,8",
    [int]    $ConcurrencyDuration = 30,
    [switch] $DryRun,
    [switch] $Sweep
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$metaPath = Join-Path $PSScriptRoot "dataset\dataset_meta.json"

if (-not (Test-Path $metaPath)) {
    throw "dataset_meta.json yok. Once: python benchmark/build_dataset.py"
}
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json

# --- .env'den OpenSearch kimlik bilgileri ---
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"').Trim("'"))
        }
    }
}
$osUser = $env:OPENSEARCH_USER
$osPass = $env:OPENSEARCH_PASSWORD
if (-not $osPass) { throw "OPENSEARCH_PASSWORD tanimli degil (.env veya ortam degiskeni)." }
if (-not $osUser) { $osUser = "admin" }

# --- SSL: OpenSearch dev kurulumu 9200'de HTTPS + self-signed konusur ---
if ($NoSsl) {
    $env:OSS_OPENSEARCH_USE_SSL = "false"
} else {
    $env:OSS_OPENSEARCH_USE_SSL      = "true"
    $env:OSS_OPENSEARCH_VERIFY_CERTS = "false"
}

if ($K -gt $meta.gt_k) {
    throw "K=$K ama ground truth sadece $($meta.gt_k) komsu iceriyor. K'yi dusur ya da build_dataset.py'yi --gt-k $K ile calistir."
}

$metricLower = $meta.metric_type.ToLower()

Write-Host ""
Write-Host "=== Benchmark yapilandirmasi ===" -ForegroundColor Cyan
Write-Host ("  hedef      : {0}:{1} (ssl={2})" -f $OpenSearchHost, $Port, (-not $NoSsl))
Write-Host ("  dataset    : {0} vektor x {1} boyut, metric={2}" -f $meta.size, $meta.dim, $meta.metric_type)
Write-Host ("  embedding  : {0} ({1})" -f $meta.embed_model, $meta.embed_backend)
Write-Host ("  chunk      : {0}/{1}" -f $meta.chunk_size, $meta.chunk_overlap)
Write-Host ("  arama      : k={0} ef_search={1} M={2} ef_construction={3} engine={4}" -f $K, $EfSearch, $M, $EfConstruction, $Engine)
Write-Host ""

function Invoke-Bench {
    param([int]$m, [int]$efc, [int]$efs, [string]$label)

    $bench = @(
        "ossopensearch"
        "--host", $OpenSearchHost
        "--port", $Port
        "--user", $osUser
        "--password", $osPass
        "--db-label", $label
        "--case-type", "PerformanceCustomDataset"
        "--custom-case-name", "$DatasetName-custom"
        "--custom-case-description", "DeneyselRAG kendi PDF korpusu"
        "--custom-dataset-name", $DatasetName
        "--custom-dataset-dir", $DatasetDir
        "--custom-dataset-size", $meta.size
        "--custom-dataset-dim", $meta.dim
        "--custom-dataset-metric-type", $meta.metric_type
        "--custom-dataset-file-count", "1"
        "--custom-dataset-with-gt"
        "--skip-custom-dataset-use-shuffled"
        "--metric-type", $metricLower
        "--engine", $Engine
        "--m", $m
        "--ef-construction", $efc
        "--ef-search", $efs
        "--k", $K
        "--num-concurrency", $NumConcurrency
        "--concurrency-duration", $ConcurrencyDuration
        "--number-of-shards", "1"
        "--number-of-replicas", "0"
        "--drop-old"
    )
    if ($DryRun) { $bench += "--dry-run" }

    Write-Host ">>> $label" -ForegroundColor Yellow
    & vectordbbench @bench
    if ($LASTEXITCODE -ne 0) { Write-Warning "Kosu basarisiz: $label (exit $LASTEXITCODE)" }
    Write-Host ""
}

if ($Sweep) {
    # HNSW parametre taramasi: recall/QPS egrisini cikarir.
    foreach ($m in @(8, 16, 32)) {
        foreach ($efs in @(32, 64, 128, 256)) {
            Invoke-Bench -m $m -efc $EfConstruction -efs $efs -label "os-m$m-efs$efs"
        }
    }
} else {
    Invoke-Bench -m $M -efc $EfConstruction -efs $EfSearch -label "os-m$M-efs$EfSearch"
}

Write-Host "Sonuclar: vectordb_bench/results/ altinda JSON olarak duruyor." -ForegroundColor Green
Write-Host "Streamlit arayuzunde gormek icin:  init_bench" -ForegroundColor Green
