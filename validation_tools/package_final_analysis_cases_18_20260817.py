from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(
    r"F:\CODE1\_organized_active_work_20260817"
    r"\05_nsga3_current_0_683_20260817\final_analysis_cases_18"
)

ASSIGNMENTS = {
    "machine_1_final_analysis_09": {
        "case_ids": [1, 2, 3, 4, 5, 6, 11, 12, 15],
        "cmg_exe": r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe",
        "max_parallel": 5,
    },
    "machine_4_final_analysis_02": {
        "case_ids": [17, 18],
        "cmg_exe": r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe",
        "max_parallel": 1,
    },
    "machine_5_final_analysis_07": {
        "case_ids": [7, 8, 9, 10, 13, 14, 16],
        "cmg_exe": r"E:\projects\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe",
        "max_parallel": 4,
    },
}


RUN_GROUP_PS1 = r'''param(
    [string]$Root = $PSScriptRoot,
    [string]$CmgExe = "__CMG_EXE__",
    [int]$Parasol = 8,
    [int]$MaxParallel = __MAX_PARALLEL__,
    [int]$PollIntervalSec = 30,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$GroupName = Split-Path -Leaf $ScriptDir
$ManifestPath = Join-Path $ScriptDir "manifest.json"
$StatePath = Join-Path $ScriptDir "run_state.json"

if (-not (Test-Path $ManifestPath)) { throw "Manifest not found: $ManifestPath" }
if (-not (Test-Path $CmgExe)) { throw "CMG executable not found: $CmgExe" }
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$LogPath = if ($LogFile) { $LogFile } else { Join-Path $ScriptDir "run_log.txt" }

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Line = "$Timestamp [$GroupName] $Message"
    Write-Host $Line
    Add-Content -LiteralPath $LogPath -Value $Line
}

function Update-DatSipPath {
    param([string]$DatPath)
    $SipPath = Join-Path $Root "cmg\2026_0421.sip"
    if (-not (Test-Path $SipPath)) { throw "SIP file not found: $SipPath" }
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
        if ($Sr3 -and $Sr3.Length -gt 0 -and (Test-Path $OutPath)) {
            $StopLine = Select-String -LiteralPath $OutPath -Pattern "STOP GEM" -ErrorAction SilentlyContinue | Select-Object -Last 1
            if ($StopLine -and ($StopLine.Line -notmatch "Data error|Abnormal|Fatal|Time step too small")) { return $true }
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
    $Payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

$CompletedIndices = [System.Collections.ArrayList]::new()
if (Test-Path $StatePath) {
    $State = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    foreach ($idx in $State.completed_indices) { [void]$CompletedIndices.Add([int]$idx) }
}

$Jobs = [System.Collections.ArrayList]::new()
foreach ($Case in $Manifest.cases) {
    if ($CompletedIndices -contains [int]$Case.sample_index) { continue }
    if (Test-CaseCompleted -Case $Case) {
        [void]$CompletedIndices.Add([int]$Case.sample_index)
        continue
    }
    $DatPath = Join-Path $Root $Case.relative_dat_path
    if (-not (Test-Path $DatPath)) { throw "DAT file not found: $DatPath" }
    $WorkDir = Split-Path $DatPath -Parent
    Update-DatSipPath -DatPath $DatPath
    $ArgLine = "-f `"$DatPath`" -wd `"$WorkDir`" -parasol $Parasol -doms"
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
            $Proc.WaitForExit()
            $Proc.Refresh()
            $ExitCode = $Proc.ExitCode
            if (Test-CaseCompleted -Case $Info.Case) {
                $Completed++
                [void]$CompletedIndices.Add([int]$Info.Case.sample_index)
                Save-State -CompletedIndices $CompletedIndices
                Write-Log "[$Completed/$Total] OK case=$($Info.Case.sample_index) exit=$ExitCode"
            } else {
                Write-Log "FAIL exit=$ExitCode case=$($Info.Case.sample_index)"
                throw "CMG case failed: $($Info.Case.sample_index)"
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
        Write-Log "START case=$($Job.Case.sample_index) dat=$($Job.DatPath)"
        $Proc = Start-Process -FilePath $CmgExe -ArgumentList $Job.ArgLine `
            -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $StdOut -RedirectStandardError $StdErr
        $Running[$Proc] = $Job
    }
    if ($Completed -lt $Total) { Start-Sleep -Seconds $PollIntervalSec }
}

Write-Log "DONE: $Total/$Total cases completed."
'''


def load_manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    manifest = load_manifest()
    cases_by_id = {int(case["case_id"]): case for case in manifest["cases"]}
    assigned = [case_id for item in ASSIGNMENTS.values() for case_id in item["case_ids"]]
    if sorted(assigned) != list(range(1, 19)):
        raise ValueError("Machine assignments must cover cases 1-18 exactly once")

    original_cmg = ROOT / "cmg" / "2026_0421.sip"
    if not original_cmg.exists():
        raise FileNotFoundError(original_cmg)

    machine_by_case = {}
    for machine_name, settings in ASSIGNMENTS.items():
        machine_dir = ROOT / machine_name
        if machine_dir.exists():
            raise FileExistsError(machine_dir)
        lhs_dir = machine_dir / "lhs_doe"
        cmg_dir = machine_dir / "cmg"
        lhs_dir.mkdir(parents=True)
        cmg_dir.mkdir(parents=True)
        shutil.copy2(original_cmg, cmg_dir / original_cmg.name)

        group_cases = []
        for case_id in settings["case_ids"]:
            case = dict(cases_by_id[case_id])
            old_relative = Path(case["relative_dat_path"])
            source_case_dir = ROOT / old_relative.parts[0]
            destination_case_dir = lhs_dir / source_case_dir.name
            if not source_case_dir.exists():
                raise FileNotFoundError(source_case_dir)
            shutil.move(str(source_case_dir), str(destination_case_dir))
            relative_dat = Path("lhs_doe") / source_case_dir.name / "run_001" / "opt_001.dat"
            case["sample_index"] = case_id
            case["relative_dat_path"] = relative_dat.as_posix()
            case["machine"] = machine_name
            group_cases.append(case)
            machine_by_case[case_id] = (machine_name, relative_dat)

        group_manifest = {
            "group_name": machine_name,
            "n_cases": len(group_cases),
            "sample_indices": settings["case_ids"],
            "cmg_exe_default": settings["cmg_exe"],
            "max_parallel_default": settings["max_parallel"],
            "cases": group_cases,
        }
        (machine_dir / "manifest.json").write_text(
            json.dumps(group_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ps1 = RUN_GROUP_PS1.replace("__CMG_EXE__", settings["cmg_exe"]).replace(
            "__MAX_PARALLEL__", str(settings["max_parallel"])
        )
        (machine_dir / "run_group.ps1").write_text(ps1, encoding="utf-8")

    for case in manifest["cases"]:
        case_id = int(case["case_id"])
        machine_name, relative_dat = machine_by_case[case_id]
        case["machine"] = machine_name
        case["relative_dat_path"] = (Path(machine_name) / relative_dat).as_posix()
    manifest["machine_packages"] = {
        name: {
            "n_cases": len(settings["case_ids"]),
            "case_ids": settings["case_ids"],
            "max_parallel": settings["max_parallel"],
        }
        for name, settings in ASSIGNMENTS.items()
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    csv_path = ROOT / "analysis_cases.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    fields = list(rows[0])
    if "machine" not in fields:
        fields.insert(4, "machine")
    for row in rows:
        case_id = int(row["case_id"])
        machine_name, relative_dat = machine_by_case[case_id]
        row["machine"] = machine_name
        row["relative_dat_path"] = (Path(machine_name) / relative_dat).as_posix()
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    shutil.rmtree(ROOT / "cmg")
    print("Packaged 18 cases for machine 1 (9), machine 4 (2), and machine 5 (7).")


if __name__ == "__main__":
    main()
