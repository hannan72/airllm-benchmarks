<#
.SYNOPSIS
  AirLLM benchmarks on a Windows PC with an 8 GB NVIDIA GPU (written for a GTX 1070, Pascal sm_61).

.DESCRIPTION
  1. builds .venv with uv: Python 3.12, PyTorch 2.14.1 for CUDA 12.6 (the newest build that still
     ships Pascal kernels; CUDA 12.8+ builds dropped sm_50-sm_61), the repo's requirements and
     transformers 5.23.0 (same as the RTX 2070 runs)
  2. preflight: GPU, fp16 / bf16 matmul and bitsandbytes NF4 on this GPU; falls back to fp16 if bf16 fails
  3. bench/run_matrix.py with the "tuned8gb" plan (10 jobs, every model), resumable: rerun to continue
  4. prints a summary; with -Push, commits results\<tag> and pushes it

  Needs: NVIDIA driver, git, uv (https://docs.astral.sh/uv/), about 100 GB free where the HF cache
  lives (checkpoints are deleted after each model), and many hours for ~220 GB of downloads.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\run_pc.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\run_pc.ps1 -HfHome D:\hf -WorkDir D:\airllm-work -Push
#>
param(
    [string]$Tag = "",            # results\<tag>; default local-<gpu>, e.g. local-gtx1070
    [string]$Plan = "tuned8gb",
    [string]$HfHome = "",         # where checkpoints are downloaded (default %USERPROFILE%\.cache\huggingface)
    [string]$WorkDir = "",        # where AirLLM writes split layers (default %USERPROFILE%\airllm-work)
    [string]$Only = "",           # space-separated subset of models, e.g. "gemma-4-E4B-it Qwen3.5-9B"
    [switch]$SetupOnly,
    [switch]$Push
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$env:PYTHONUTF8 = "1"                  # model output is UTF-8; Windows' default code page would crash on it
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($HfHome) { $env:HF_HOME = $HfHome }
if (-not $WorkDir) { $WorkDir = Join-Path $env:USERPROFILE "airllm-work" }
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Check-Exit($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

Step "GPU"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw "nvidia-smi not found: install the NVIDIA driver first" }
$gpuName = (& nvidia-smi "--query-gpu=name" "--format=csv,noheader" | Select-Object -First 1).Trim()
& nvidia-smi "--query-gpu=name,memory.total,memory.used,driver_version" "--format=csv"
if (-not $Tag) {
    $short = ($gpuName -replace "NVIDIA", "" -replace "GeForce", "" -replace "[^A-Za-z0-9]", "").ToLower()
    $Tag = "local-$short"
}
Write-Host "results go to results\$Tag"

Step "Python environment (.venv)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Install it (see https://docs.astral.sh/uv/getting-started/installation/), e.g.:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "then open a new PowerShell window and run this script again."
    exit 1
}
if (-not (Test-Path $Py)) { uv venv .venv --python 3.12; Check-Exit "uv venv" }
# one resolve, so nothing swaps the CUDA 12.6 torch for PyPI's (CUDA 13, no Pascal kernels)
uv pip install -p $Py "torch==2.14.1+cu126" -r requirements.txt "transformers==5.23.0" `
    --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
Check-Exit "package install"
& $Py -c "import torch, transformers, airllm, bitsandbytes; print('torch', torch.__version__, '| transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__)"
Check-Exit "import check"

Step "Preflight"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$pre = "logs\preflight_$Tag.json"
& $Py scripts\windows\pc_bench.py preflight --out $pre
Check-Exit "preflight (see $pre)"
$dtype = (Get-Content $pre -Raw | ConvertFrom-Json).dtype
if ($dtype -ne "auto") { Write-Host "bf16 kernels failed on this GPU: running everything in $dtype" -ForegroundColor Yellow }
if ($SetupOnly) { Write-Host "setup done (-SetupOnly)"; exit 0 }

Step "Benchmark matrix: plan $Plan (resumable; Ctrl+C and rerun to continue)"
$env:BENCH_HOST_TAG = $Tag
$matrixArgs = @("bench\run_matrix.py", "--plan", $Plan, "--out-dir", "results\$Tag", "--work-dir", $WorkDir,
                "--cleanup", "--no-wait-gpu", "--dtype", $dtype)
if ($Only) { $matrixArgs += @("--only") + ($Only -split "\s+") }
& $Py @matrixArgs
Check-Exit "run_matrix"

Step "Summary"
& $Py scripts\windows\pc_bench.py summary "results\$Tag"
Copy-Item $pre "results\$Tag\_preflight.json.txt" -Force   # not *.json, so the report ignores it

if ($Push) {
    Step "git push results\$Tag"
    git add "results\$Tag"; Check-Exit "git add"
    git commit -m "Results: $gpuName (AirLLM tuned, Windows, plan $Plan)"; Check-Exit "git commit"
    git pull --rebase; Check-Exit "git pull"
    git push; Check-Exit "git push"
} else {
    Write-Host "`nTo share the results: git add results\$Tag; git commit -m 'Results: $gpuName'; git push"
}
