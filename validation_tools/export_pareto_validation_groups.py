from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(r"F:\CODE1")
SRC_ROOT = ROOT / "opt_runs" / "pareto_validation_16_current"
OUT_ROOT = ROOT / "opt_runs_pareto_validation_export"
SIP_SRC = ROOT / "cmg" / "2026_0421.sip"

GROUPS = [
    ("machine_1_pareto_validation_0001_0010", list(range(1, 11)), r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe", 5),
    ("machine_4_pareto_validation_0011_0013", list(range(11, 14)), r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe", 3),
    ("machine_2_pareto_validation_0014_0015", list(range(14, 16)), r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe", 2),
    ("machine_5_pareto_validation_0016_0016", [16], r"E:\projects\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe", 1),
]


RUN_GROUP_PS1 = r'''param(
    [string]$Root = $PSScriptRoot,
    [string]$CmgExe = "__CMG_EXE__",
    [int]$Parasol = 8,
    [int]$MaxParallel = __MAX_PARALLEL__,
    [switch]$UseJacpar,
    [int]$PollIntervalSec = 30,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GroupName = Split-Path -Leaf $ScriptDir
$ManifestPath = Join-Path $ScriptDir "manifest.json"
$StatePath = Join-Path $ScriptDir "run_state.json"

if (-not (Test-Path $ManifestPath)) {
    throw "Manifest not found: $ManifestPath"
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

if ($LogFile) {
    $LogPath = $LogFile
} else {
    $LogPath = Join-Path $ScriptDir "run_log.txt"
}

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Line = "$Timestamp [$GroupName] $Message"
    Write-Host $Line
    Add-Content -Path $LogPath -Value $Line
}

function Update-DatSipPath {
    param([string]$DatPath)
    $SipPath = Join-Path $Root "cmg\2026_0421.sip"
    if (-not (Test-Path $SipPath)) {
        throw "SIP file not found: $SipPath"
    }
    $Text = Get-Content -LiteralPath $DatPath -Raw
    $SipForDat = $SipPath.Replace("\", "/")
    $Text = [regex]::Replace(
        $Text,
        "FILENAMES\s+SIPDATA-IN\s+'[^']*'",
        "FILENAMES SIPDATA-IN '$SipForDat'"
    )
    Set-Content -LiteralPath $DatPath -Value $Text -Encoding UTF8
}

function Test-CaseCompleted {
    param($Case)
    $DatPath = Join-Path $Root $Case.relative_dat_path
    $WorkDir = Split-Path $DatPath -Parent
    $Sr3Path = Join-Path $WorkDir "opt_001.sr3"
    $LockPath = Join-Path $WorkDir "opt_001.sr3.nw.lck"
    $OutPath = Join-Path $WorkDir "opt_001.out"
    if ((Test-Path $Sr3Path) -and (-not (Test-Path $LockPath))) {
        $Sr3 = Get-Item $Sr3Path -ErrorAction SilentlyContinue
        if ($Sr3 -and $Sr3.Length -gt 0) {
            if (Test-Path $OutPath) {
                $StopLine = Select-String -Path $OutPath -Pattern "STOP GEM" -ErrorAction SilentlyContinue | Select-Object -Last 1
                if ($StopLine) {
                    if ($StopLine.Line -match "Data error|Abnormal|Fatal|Time step too small") {
                        return $false
                    }
                    return $true
                }
            }
        }
    }
    return $false
}

function Save-State {
    param([System.Collections.ArrayList]$CompletedIndices)
    $Payload = [PSCustomObject]@{
        group_name = $GroupName
        completed_indices = @($CompletedIndices | Sort-Object -Unique)
        updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    }
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8
}

$CompletedIndices = [System.Collections.ArrayList]::new()
if (Test-Path $StatePath) {
    $State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    foreach ($idx in $State.completed_indices) {
        [void]$CompletedIndices.Add([int]$idx)
    }
}

$Jobs = [System.Collections.ArrayList]::new()
foreach ($Case in $Manifest.cases) {
    if ($CompletedIndices -contains [int]$Case.sample_index) {
        continue
    }
    if (Test-CaseCompleted -Case $Case) {
        [void]$CompletedIndices.Add([int]$Case.sample_index)
        continue
    }
    $DatPath = Join-Path $Root $Case.relative_dat_path
    $WorkDir = Split-Path $DatPath -Parent
    Update-DatSipPath -DatPath $DatPath
    $Args = @(
        "-f", ('"{0}"' -f $DatPath),
        "-wd", ('"{0}"' -f $WorkDir),
        "-parasol", "$Parasol"
    )
    if ($UseJacpar) {
        $Args += "-jacpar"
    } else {
        $Args += "-doms"
    }
    $ArgLine = $Args -join " "
    [void]$Jobs.Add(@{
        Case = $Case
        DatPath = $DatPath
        WorkDir = $WorkDir
        ArgLine = $ArgLine
    })
}

$Total = $Manifest.cases.Count
$PendingTotal = $Jobs.Count
$AlreadyDone = @($CompletedIndices | Sort-Object -Unique).Count
Save-State -CompletedIndices $CompletedIndices

$Running = @{}
$Completed = $AlreadyDone
$NextIdx = 0

Write-Log "START: total=$Total pending=$PendingTotal completed=$AlreadyDone MaxParallel=$MaxParallel Parasol=$Parasol CmgExe=$CmgExe"

while ($Completed -lt $Total) {
    $Finished = [System.Collections.ArrayList]::new()
    foreach ($Entry in $Running.GetEnumerator()) {
        $Proc = $Entry.Key
        if ($Proc.HasExited) {
            $Info = $Entry.Value
            [void]$Finished.Add($Proc)
            if ($Proc.ExitCode -eq 0) {
                $Completed++
                [void]$CompletedIndices.Add([int]$Info.Case.sample_index)
                Save-State -CompletedIndices $CompletedIndices
                Write-Log "[$Completed/$Total] OK  si=$($Info.Case.sample_index) dat=$($Info.DatPath)"
            } else {
                Write-Log "[$Completed/$Total] FAIL (exit=$($Proc.ExitCode)) si=$($Info.Case.sample_index) dat=$($Info.DatPath)"
            }
        }
    }
    foreach ($Proc in $Finished) {
        $Running.Remove($Proc)
        $Proc.Dispose()
    }

    while ($Running.Count -lt $MaxParallel -and $NextIdx -lt $PendingTotal) {
        $Job = $Jobs[$NextIdx]
        $NextIdx++
        $StdOut = Join-Path $Job.WorkDir "cmg_stdout.txt"
        $StdErr = Join-Path $Job.WorkDir "cmg_stderr.txt"
        Write-Log "START si=$($Job.Case.sample_index) dat=$($Job.DatPath)"
        $Proc = Start-Process -FilePath $CmgExe -ArgumentList $Job.ArgLine `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $StdOut `
            -RedirectStandardError $StdErr
        $Running[$Proc] = $Job
    }

    if ($Completed -lt $Total) {
        Start-Sleep -Seconds $PollIntervalSec
    }
}

Write-Log "DONE: $Total/$Total cases completed."
'''


def load_current_manifest() -> dict[int, dict]:
    path = SRC_ROOT / "manifest.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["case_id"]): item for item in items}


def patch_dat_sip(dat_path: Path) -> None:
    text = dat_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(
        r"FILENAMES\s+SIPDATA-IN\s+'[^']*'",
        "FILENAMES SIPDATA-IN 'cmg/2026_0421.sip'",
        text,
    )
    dat_path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(SRC_ROOT)
    if not SIP_SRC.exists():
        raise FileNotFoundError(SIP_SRC)

    case_meta = load_current_manifest()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary = []
    for group_name, case_ids, cmg_exe, max_parallel in GROUPS:
        group_dir = OUT_ROOT / group_name
        if group_dir.exists():
            shutil.rmtree(group_dir)
        (group_dir / "lhs_doe").mkdir(parents=True)
        (group_dir / "cmg").mkdir(parents=True)
        shutil.copy2(SIP_SRC, group_dir / "cmg" / "2026_0421.sip")

        cases = []
        for case_id in case_ids:
            src_case = SRC_ROOT / f"case_{case_id:04d}_rank{case_id:02d}"
            src_dat = src_case / "opt_001.dat"
            if not src_dat.exists():
                raise FileNotFoundError(src_dat)

            dst_case = group_dir / "lhs_doe" / f"case_{case_id:04d}" / "run_001"
            dst_case.mkdir(parents=True)
            dst_dat = dst_case / "opt_001.dat"
            shutil.copy2(src_dat, dst_dat)
            patch_dat_sip(dst_dat)

            rel_dat = dst_dat.relative_to(group_dir).as_posix()
            meta = case_meta[case_id]
            cases.append(
                {
                    "sample_index": case_id,
                    "case_id": f"case_{case_id:04d}",
                    "relative_dat_path": rel_dat,
                    "source_rank": meta["source_rank"],
                    "pred_pressure_rise_pct": meta["pred_pressure_rise_pct"],
                    "pred_dissolution_pct": meta["pred_dissolution_pct"],
                    "pred_brine_m3": meta["pred_brine_m3"],
                    "inj_wells": meta["inj_wells"],
                    "prod_wells": meta["prod_wells"],
                }
            )

        manifest = {
            "group_name": group_name,
            "source": str(SRC_ROOT),
            "cases": cases,
        }
        (group_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ps1 = RUN_GROUP_PS1.replace("__CMG_EXE__", cmg_exe).replace(
            "__MAX_PARALLEL__", str(max_parallel)
        )
        (group_dir / "run_group.ps1").write_text(ps1, encoding="utf-8")
        summary.append(
            {
                "group": group_name,
                "n_cases": len(cases),
                "case_ids": case_ids,
                "path": str(group_dir),
            }
        )

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
