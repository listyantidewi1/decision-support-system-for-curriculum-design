<#
run_annotation_stack.ps1 — supervise the gold-standard annotation stack and
survive a reboot.

Usage:
  .\run_annotation_stack.ps1 status
  .\run_annotation_stack.ps1 start     -Mode dryrun
  .\run_annotation_stack.ps1 start     -Mode pilot -Init     # first run only
  .\run_annotation_stack.ps1 stop
  .\run_annotation_stack.ps1 restart   -Mode pilot
  .\run_annotation_stack.ps1 install   -Mode pilot           # auto-start at logon
  .\run_annotation_stack.ps1 uninstall
  .\run_annotation_stack.ps1 backup    -Mode pilot

The stack is two processes that need DIFFERENT venvs (CLAUDE.md section 2):

  collector  .venv      fastapi + uvicorn   127.0.0.1:8000   owns the SQLite file
  annotator  .venv-ml   streamlit            0.0.0.0:8501    talks to the collector

Why this script exists rather than two shell commands: every interesting failure
in this stack is SILENT. The services answer 200 while collecting nothing. The
preflight below turns each of those into a refusal to start.

  * `gold_standard/app.py:97` reads GS_API_URL with a "" default and `:104`
    gates the collector on `bool(API_URL -and token)`. With GS_API_URL unset the
    UI renders perfectly, autosaves "successfully" to local JSON, and the
    collector database stays empty. You find out at IAA time.
  * `annotation_server/api.py:50` defaults GS_DB_PATH to the RELATIVE
    "data/annotations.sqlite", and `store.py:51` mkdir -p's and schema-creates
    whatever path it lands on. A typo, or a task whose working directory is
    C:\Windows\System32, produces a valid EMPTY database with no warning.
  * `annotation_server/auth.py:39-42` returns an empty token table when
    GS_TOKENS_FILE is missing, so every annotator gets 401 while /health still
    returns {"ok":true} — it never consults auth.
  * `gold_standard/app.py:101` falls back to the GS_API_TOKEN environment
    variable. Set that on the service and all three annotators authenticate as
    one identity, UPSERT over each other, and the three independent judgements
    the whole IAA design rests on are gone — with no error anywhere.

Mode is the contamination boundary, and it is physical, not a convention:

  pilot    REAL human work   DATA\gold_standard\collector\pilot_100.sqlite
  dryrun   SYNTHETIC         <ops>\dryrun\annotations_dryrun.sqlite

The dry-run database lives OUTSIDE the repository on purpose, so synthetic rows
cannot end up in the tree that feeds the dissertation's reliability evidence.
annotation_server/README.md:73 requires exactly that ("somewhere disposable,
NOT in DATA/"); this makes it structural instead of remembered.

Logs also live outside the tree: .gitignore covers logs/*.log but NOT
logs\*.log.1 or logs\*.out, so the first rotation inside the repo would create a
tracked file on a PUBLIC repo (CLAUDE.md section 6). Note that --log-level
warning on uvicorn is load-bearing, not cosmetic: at info level the access log
would write every annotator's ?token=... into that file.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'start', 'stop', 'restart', 'install', 'uninstall',
                 'backup', 'run-collector', 'run-ui')]
    [string]$Action = 'status',

    [ValidateSet('pilot', 'dryrun')]
    [string]$Mode = 'pilot',

    # Permit creating the database. Without it, a missing GS_DB_PATH is FATAL
    # rather than silently bootstrapped — see the api.py:50 note above.
    [switch]$Init
)

$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------- layout
$Repo    = $PSScriptRoot
$OpsHome = 'D:\Projects\_skill-extraction-ops'
$LogDir  = Join-Path $OpsHome 'logs'
$BakDir  = Join-Path $OpsHome 'backups'
$RunDir  = Join-Path $OpsHome 'run'

$PyWeb = Join-Path $Repo '.venv\Scripts\python.exe'
$PyMl  = Join-Path $Repo '.venv-ml\Scripts\python.exe'
$AppPy = Join-Path $Repo 'gold_standard\app.py'

$CollectorPort = 8000
$UiPort        = 8501

$TaskFolder    = '\GoldStandard\'
$TaskCollector = 'collector-8000'
$TaskUi        = 'annotator-ui-8501'

function Get-ModeConfig {
    param([string]$M)
    if ($M -eq 'pilot') {
        return [pscustomobject]@{
            Mode       = 'pilot'
            Split      = 'pilot_100'
            Db         = Join-Path $Repo 'DATA\gold_standard\collector\pilot_100.sqlite'
            TokensFile = Join-Path $Repo 'api_keys\annotation_tokens.json'
            Synthetic  = $false
        }
    }
    return [pscustomobject]@{
        Mode       = 'dryrun'
        Split      = 'practice_10'
        Db         = Join-Path $OpsHome 'dryrun\annotations_dryrun.sqlite'
        TokensFile = Join-Path $Repo 'api_keys\dryrun_tokens.json'
        Synthetic  = $true
    }
}

# -------------------------------------------------------------------- helpers
function New-DirIfMissing {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Write-StackLog {
    param([string]$File, [string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz'
    try { Add-Content -LiteralPath $File -Value ("$stamp $Message") -Encoding utf8 }
    catch { }
}

function Get-PortOwnerPid {
    param([int]$Port)
    try {
        $conn = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop)
        if ($conn.Count -gt 0) { return [int]$conn[0].OwningProcess }
    } catch { }
    return $null
}

function Get-ProcInfo {
    param([int]$ProcId)
    try { return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcId" -ErrorAction Stop }
    catch { return $null }
}

# The venv python.exe is a 254 KB LAUNCHER STUB whose CHILD (C:\Python313\
# python.exe) actually owns the socket. Killing one PID leaves the other holding
# the port, after which a restart cannot bind and the OLD process — with the OLD
# configuration — keeps serving. Walk up to the outermost process that still
# belongs to this repo and kill the whole tree from there.
function Get-StackTreeRootPid {
    param([int]$ProcId)
    $cur = $ProcId
    for ($i = 0; $i -lt 8; $i++) {
        $p = Get-ProcInfo $cur
        if (-not $p) { break }
        $parent = Get-ProcInfo ([int]$p.ParentProcessId)
        if (-not $parent) { break }
        $cmd = [string]$parent.CommandLine
        $looksOurs = $cmd -and (($cmd -like "*$Repo*") -or ($cmd -like "*run_annotation_stack*"))
        if (-not $looksOurs) { break }
        $cur = [int]$parent.ProcessId
    }
    return $cur
}

function Stop-Tree {
    param([int]$ProcId)
    # taskkill /T terminates descendants too; Stop-Process does not.
    #
    # 'Continue' is load-bearing, not tidiness: with 2>&1 every taskkill stderr
    # line becomes a NativeCommandError, which under this script's 'Stop' is
    # TERMINATING. A stale PID ("The process ... not found.") would then abort
    # Invoke-Stop before it reached the second port, the still-bound warnings
    # and the runtime-state cleanup. 2>$null does not help: the ErrorRecord is
    # still constructed. The preference is function-scoped, so the rest of the
    # script keeps 'Stop'.
    $ErrorActionPreference = 'Continue'
    $out = & "$env:WINDIR\System32\taskkill.exe" /PID $ProcId /T /F 2>&1
    # 0 = killed; 128 = no such process, which is the benign already-exited race
    # that this guard exists for. The port check in Invoke-Stop is authoritative.
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 128) {
        Write-Host ("  taskkill {0} -> code {1}: {2}" -f $ProcId, $LASTEXITCODE, ($out -join '; ')) -ForegroundColor Yellow
    }
}

function Get-AdminToken {
    param([string]$TokensFile)
    if (-not (Test-Path -LiteralPath $TokensFile)) { return $null }
    try {
        $raw = Get-Content -LiteralPath $TokensFile -Raw -Encoding utf8 | ConvertFrom-Json
    } catch { return $null }
    foreach ($p in $raw.PSObject.Properties) {
        if ($p.Value.admin -eq $true) { return $p.Name }
    }
    return $null
}

function Invoke-Probe {
    param([string]$Url, [string]$Token)
    $headers = @{}
    if ($Token) { $headers['Authorization'] = "Bearer $Token" }
    try {
        $r = Invoke-WebRequest -Uri $Url -Headers $headers -UseBasicParsing -TimeoutSec 10
        return [pscustomobject]@{ Ok = $true; Code = [int]$r.StatusCode; Body = $r.Content }
    } catch {
        $code = 0
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        return [pscustomobject]@{ Ok = $false; Code = $code; Body = $_.Exception.Message }
    }
}

function Get-DbCounts {
    param([string]$DbPath)
    if (-not (Test-Path -LiteralPath $DbPath)) {
        return [pscustomobject]@{ State = 'nofile' }
    }
    # mode=ro, NOT immutable=1. immutable promises the file never changes, so
    # SQLite skips the -wal entirely and would under-report rows committed since
    # the last checkpoint — a status read that silently lags the truth. mode=ro
    # takes real shared locks, hence the busy_timeout.
    $code = @'
import json, sqlite3, sys
try:
    c = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
    c.execute("PRAGMA busy_timeout=15000")
    out = {t: c.execute("select count(*) from " + t).fetchone()[0]
           for t in ("annotations", "events")}
    out["splits"] = [r[0] for r in c.execute(
        "select distinct split from annotations order by split")]
    out["annotators"] = [r[0] for r in c.execute(
        "select distinct annotator from annotations order by annotator")]
except Exception as e:
    out = {"error": repr(e)}
print(json.dumps(out))
'@
    $tmp = Join-Path $RunDir '_counts.py'
    New-DirIfMissing $RunDir
    Set-Content -LiteralPath $tmp -Value $code -Encoding utf8
    # Discriminate the three outcomes. Returning a bare payload-or-$null made a
    # failed probe render as a plausible zero-row database, and a missing file
    # indistinguishable from an unreadable one.
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $raw = & $PyWeb -I $tmp $DbPath 2>&1
        $ErrorActionPreference = $prev
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{ State = 'probefail'; Detail = ($raw -join ' ') }
        }
        $j = $raw | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{ State = 'probefail'; Detail = $_.Exception.Message }
    }
    if ($null -ne $j.error) {
        return [pscustomobject]@{ State = 'dberror'; Detail = [string]$j.error }
    }
    return [pscustomobject]@{ State = 'ok'; Counts = $j }
}

# Work written by the UI's local-file fallback. app.py:104 needs BOTH GS_API_URL
# and a ?token=; a visit without the token renders a fully working UI and writes
# here instead of to the collector. gold_standard/iaa.py reads this exact
# directory, so a stray file later blends with real pilot work — which has
# already happened once (annotation_server/auth.py:7-8). Non-recursive on
# purpose: the known _trial/ archive lives in a subdirectory.
function Get-StrayLocalAnnotations {
    $dir = Join-Path $Repo 'DATA\gold_standard\annotations'
    return @(Get-ChildItem -LiteralPath $dir -Filter '*.json' -File -ErrorAction SilentlyContinue)
}

# ------------------------------------------------------------------ preflight
# Every check here REFUSES TO START. That is the whole point: this stack's
# failure mode is looking healthy while losing data, so a loud stop is strictly
# better than a quiet start.
function Test-Preflight {
    param([pscustomobject]$Cfg, [string]$Half, [switch]$AllowInit, [string]$LogFile)

    $problems = New-Object System.Collections.ArrayList

    if (-not (Test-Path -LiteralPath $PyWeb)) {
        [void]$problems.Add("missing web venv: $PyWeb  (see PYTHON_ENV.md; the collector needs fastapi, which .venv-ml does not have)")
    }
    if (-not (Test-Path -LiteralPath $PyMl)) {
        [void]$problems.Add("missing ML venv: $PyMl  (see PYTHON_ENV.md; the UI needs streamlit, which .venv does not have)")
    }
    if (-not (Test-Path -LiteralPath $AppPy)) {
        [void]$problems.Add("missing annotator UI: $AppPy")
    }

    if (-not (Test-Path -LiteralPath $Cfg.TokensFile)) {
        [void]$problems.Add("missing tokens file: $($Cfg.TokensFile)  (auth.py:39-42 would serve an EMPTY token table: every annotator 401 while /health still returns ok. Mint with: python -m annotation_server.make_tokens)")
    }

    # Refuse to run the REAL pilot against dry-run tokens. The dry-run links
    # resolve to valid (annotator, split) pairs, so nothing would complain.
    if (-not $Cfg.Synthetic) {
        if ($Cfg.TokensFile -match 'dryrun') {
            [void]$problems.Add("pilot mode is pointed at a dry-run tokens file: $($Cfg.TokensFile). Delete the dry-run tokens and mint live ones (annotation_server/README.md:107).")
        }
    }

    $samples = Join-Path $Repo 'DATA\gold_standard'
    $sampleCsv = Join-Path $samples ("{0}.csv" -f $Cfg.Split)
    if (-not (Test-Path -LiteralPath $sampleCsv)) {
        [void]$problems.Add("missing sample: $sampleCsv  (draw it with: python -m gold_standard.sample)")
    }

    # Refuse to restart over work that bypassed the collector. These files are
    # real judgements sitting where IAA will read them, so they must be imported
    # or archived deliberately — never silently started over.
    $strays = Get-StrayLocalAnnotations
    if ($strays.Count -gt 0) {
        $names = ($strays | ForEach-Object { $_.Name }) -join ', '
        [void]$problems.Add("DATA\gold_standard\annotations\ already holds $($strays.Count) annotation file(s): $names. Those were written by the UI's local-file fallback (a visit with no ?token=), not by the collector, and gold_standard/iaa.py reads that directory. Move them to DATA\gold_standard\annotations\_trial\ or import them before starting.")
    }

    # A missing database is FATAL unless -Init. api.py:50 + store.py:51 would
    # otherwise create a valid empty one and the annotators would simply see no
    # saved work — indistinguishable from a fresh start. Only the collector
    # touches the database; the UI never does, so do not block it on this.
    if ($Half -eq 'collector' -or $Half -eq 'all') {
        if (-not (Test-Path -LiteralPath $Cfg.Db)) {
            if ($AllowInit) {
                Write-Host "  -Init: will create $($Cfg.Db)" -ForegroundColor Yellow
            } else {
                [void]$problems.Add("database does not exist: $($Cfg.Db). Refusing to bootstrap an empty one silently. Pass -Init if this really is the first run.")
            }
        }
    }

    # GS_API_TOKEN anywhere in the inherited environment collapses all three
    # annotators onto one identity (app.py:101). Check the persisted scopes too
    # — a scheduled task inherits User-scope variables.
    foreach ($scope in @('Process', 'User', 'Machine')) {
        $v = [Environment]::GetEnvironmentVariable('GS_API_TOKEN', $scope)
        if ($v) {
            [void]$problems.Add("GS_API_TOKEN is set in the $scope environment. The UI falls back to it (app.py:101), so every annotator would authenticate as that one identity and UPSERT over each other. Remove it.")
        }
    }

    if ($Half -eq 'collector' -or $Half -eq 'all') {
        $owner = Get-PortOwnerPid $CollectorPort
        if ($owner) {
            [void]$problems.Add("port $CollectorPort is already bound by PID $owner. Refusing to start a second collector: the old one keeps its own GS_DB_PATH and whichever process a link reaches decides where the work lands. Run '.\run_annotation_stack.ps1 stop' first.")
        }
    }
    if ($Half -eq 'ui' -or $Half -eq 'all') {
        $owner = Get-PortOwnerPid $UiPort
        if ($owner) {
            [void]$problems.Add("port $UiPort is already bound by PID $owner. Run '.\run_annotation_stack.ps1 stop' first.")
        }
    }

    if ($problems.Count -gt 0) {
        Write-Host ""
        Write-Host "PREFLIGHT FAILED ($($problems.Count)):" -ForegroundColor Red
        foreach ($p in $problems) { Write-Host "  - $p" -ForegroundColor Red }
        Write-Host ""
        # Under a scheduled task the console above goes to a hidden window and is
        # discarded, so the operator would see only "nothing is listening" with no
        # reason anywhere. Write the same diagnosis to the durable log.
        if ($LogFile) {
            Write-StackLog $LogFile "PREFLIGHT FAILED ($($problems.Count)):"
            foreach ($p in $problems) { Write-StackLog $LogFile "  - $p" }
        }
        return $false
    }
    return $true
}

# ------------------------------------------------------- the supervised halves
# These two run under Task Scheduler (or detached from `start`). Each sets its
# own absolute environment, refuses to proceed on a failed preflight, and
# restarts its child until the child crash-loops.
function Invoke-Runner {
    param([string]$Half, [pscustomobject]$Cfg)

    # Native tools write to stderr as a matter of course; under 'Stop' every
    # such line would abort the supervisor.
    $ErrorActionPreference = 'Continue'

    New-DirIfMissing $LogDir
    New-DirIfMissing $RunDir
    New-DirIfMissing (Split-Path -Parent $Cfg.Db)

    $log = Join-Path $LogDir ("{0}.log" -f $Half)
    Write-StackLog $log "=== supervisor starting: half=$Half mode=$($Cfg.Mode) pid=$PID"

    Set-Location -LiteralPath $Repo

    if (-not (Test-Preflight -Cfg $Cfg -Half $Half -AllowInit:$Init -LogFile $log)) {
        Write-StackLog $log "FATAL preflight failed; not starting $Half"
        exit 2
    }

    # Absolute paths for everything. A scheduled task's working directory is not
    # guaranteed, and both halves carry CWD-relative defaults.
    $env:GS_DB_PATH     = $Cfg.Db
    $env:GS_TOKENS_FILE = $Cfg.TokensFile
    $env:GS_SAMPLES_DIR = Join-Path $Repo 'DATA\gold_standard'
    $env:GS_API_URL     = "http://127.0.0.1:$CollectorPort"
    # GS_SPLIT is read only by the UI's local-file fallback (app.py:260), where
    # it DEFAULTS TO "pilot_100". Without it, a token-less visit to a dry-run
    # stack would serve the real pilot sentences and write to the real
    # pilot_100_*.json — crossing the mode boundary this script is built on.
    $env:GS_SPLIT       = $Cfg.Split
    Remove-Item Env:GS_API_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:GS_TOKENS    -ErrorAction SilentlyContinue   # would shadow the file

    # Record what is ACTUALLY in effect. A running process's environment cannot
    # be read from outside, so without this `status` could only guess which
    # database the live collector is writing to.
    $state = [pscustomobject]@{
        half        = $Half
        mode        = $Cfg.Mode
        pid         = $PID
        started_utc = (Get-Date).ToUniversalTime().ToString('s') + 'Z'
        db          = $Cfg.Db
        tokens_file = $Cfg.TokensFile
        samples_dir = $env:GS_SAMPLES_DIR
        api_url     = $env:GS_API_URL
        split       = $Cfg.Split
        repo        = $Repo
    }
    $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir "$Half.json") -Encoding utf8

    $fails = 0
    while ($true) {
        $t0 = Get-Date
        if ($Half -eq 'collector') {
            Write-StackLog $log "starting uvicorn on 127.0.0.1:$CollectorPort db=$($Cfg.Db)"
            # --log-level warning is load-bearing: the access log would otherwise
            # write every annotator's ?token=... into this file.
            & $PyWeb -m uvicorn annotation_server.api:app `
                --host 127.0.0.1 --port $CollectorPort --log-level warning 2>&1 |
                ForEach-Object { Write-StackLog $log "| $_" }
        } else {
            Write-StackLog $log "starting streamlit on :$UiPort api=$($env:GS_API_URL)"
            # hideWelcomeMessage is not cosmetic: with server.headless set,
            # bootstrap.py:_print_url calls net_util.get_external_ip(), which
            # makes a BLOCKING http GET to an AWS endpoint (5 s timeout, then a
            # second https attempt). At boot that stalls the UI for up to 10 s
            # before it serves, and it contacts an external service on every
            # start for nothing. The early return at bootstrap.py:204 skips it.
            # fileWatcherType none stops an accidental edit to app.py from
            # reloading the script under three live annotators.
            & $PyMl -m streamlit run $AppPy `
                --server.port $UiPort `
                --server.headless true `
                --server.fileWatcherType none `
                --browser.gatherUsageStats false `
                --logger.hideWelcomeMessage true `
                --logger.level warning 2>&1 |
                ForEach-Object { Write-StackLog $log "| $_" }
        }
        $code = $LASTEXITCODE
        $secs = [int]((Get-Date) - $t0).TotalSeconds

        if ($secs -lt 15) { $fails++ } else { $fails = 0 }
        Write-StackLog $log "$Half exited code=$code after ${secs}s (consecutive fast exits: $fails)"

        if ($fails -ge 5) {
            Write-StackLog $log "FATAL $Half crash-looped 5 times; giving up so the failure is visible instead of flapping"
            exit 3
        }
        Start-Sleep -Seconds 5
    }
}

# ------------------------------------------------------------------- commands
function Invoke-Start {
    param([pscustomobject]$Cfg)

    New-DirIfMissing $LogDir
    New-DirIfMissing $RunDir
    New-DirIfMissing (Split-Path -Parent $Cfg.Db)

    if (-not (Test-Preflight -Cfg $Cfg -Half 'all' -AllowInit:$Init)) { exit 2 }

    # Create the database HERE rather than letting the collector do it lazily on
    # its first request. Otherwise the UI supervisor's own preflight could race
    # it, and -Init would have to be passed down to both children, which would
    # weaken their refuse-on-missing-DB guard for every later restart.
    if ($Init -and -not (Test-Path -LiteralPath $Cfg.Db)) {
        $code = @'
import sys
sys.path.insert(0, sys.argv[1])
from annotation_server.store import Store
s = Store(sys.argv[2])
print("initialised", s.path)
'@
        $tmp = Join-Path $RunDir '_init_db.py'
        Set-Content -LiteralPath $tmp -Value $code -Encoding utf8
        & $PyWeb -I $tmp $Repo $Cfg.Db
        if ($LASTEXITCODE -ne 0) { Write-Error "could not create $($Cfg.Db)"; exit 1 }
        $sha = (Get-FileHash -LiteralPath $Cfg.Db -Algorithm SHA256).Hash
        Write-Host "  created empty DB, sha256=$sha" -ForegroundColor Yellow
        Write-Host "  (record that hash: it is the proof this pilot started from an empty database)"
    }

    Write-Host "starting the annotation stack (mode=$($Cfg.Mode))" -ForegroundColor Cyan
    Write-Host "  db     $($Cfg.Db)"
    Write-Host "  tokens $($Cfg.TokensFile)"
    Write-Host "  logs   $LogDir"

    $self = Join-Path $Repo 'run_annotation_stack.ps1'
    foreach ($half in @('collector', 'ui')) {
        # No -Init here on purpose: the DB exists by now, so each supervisor
        # keeps its strict refuse-on-missing-DB guard for every later restart.
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-WindowStyle', 'Hidden',
                     '-File', "`"$self`"", "run-$half", '-Mode', $Cfg.Mode)
        Start-Process -FilePath (Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe') `
                      -ArgumentList $argList -WorkingDirectory $Repo -WindowStyle Hidden | Out-Null
        Write-Host "  launched $half supervisor"
        if ($half -eq 'collector') { Start-Sleep -Seconds 6 }   # collector first
    }

    Write-Host "  waiting for both ports..." -NoNewline
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Seconds 2
        if ((Get-PortOwnerPid $CollectorPort) -and (Get-PortOwnerPid $UiPort)) { break }
        Write-Host "." -NoNewline
    }
    Write-Host ""
    Invoke-Status -Cfg $Cfg
}

function Invoke-Stop {
    Write-Host "stopping the annotation stack" -ForegroundColor Cyan

    # Task-managed first: Stop-ScheduledTask terminates the task's whole tree.
    foreach ($t in @($TaskCollector, $TaskUi)) {
        $task = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $t -ErrorAction SilentlyContinue
        if ($task -and $task.State -eq 'Running') {
            Stop-ScheduledTask -TaskPath $TaskFolder -TaskName $t
            Write-Host "  stopped scheduled task $t"
        }
    }

    foreach ($port in @($UiPort, $CollectorPort)) {
        $owner = Get-PortOwnerPid $port
        if (-not $owner) { Write-Host "  :$port already free"; continue }
        $root = Get-StackTreeRootPid $owner
        Stop-Tree $root
        Write-Host "  :$port killed tree from PID $root (socket owner was $owner)"
    }

    for ($i = 0; $i -lt 10; $i++) {
        if (-not (Get-PortOwnerPid $CollectorPort) -and -not (Get-PortOwnerPid $UiPort)) { break }
        Start-Sleep -Seconds 1
    }
    foreach ($port in @($CollectorPort, $UiPort)) {
        $owner = Get-PortOwnerPid $port
        if ($owner) { Write-Host "  WARNING :$port still bound by PID $owner" -ForegroundColor Red }
    }
    Get-ChildItem -LiteralPath $RunDir -Filter '*.json' -ErrorAction SilentlyContinue |
        Remove-Item -ErrorAction SilentlyContinue
}

function Invoke-Status {
    param([pscustomobject]$Cfg)

    Write-Host ""
    Write-Host "=== annotation stack status ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Cyan

    # 1. sockets
    foreach ($pair in @(@{N = 'collector'; P = $CollectorPort }, @{N = 'ui'; P = $UiPort })) {
        $owner = Get-PortOwnerPid $pair.P
        if ($owner) {
            $p = Get-ProcInfo $owner
            $exe = if ($p) { $p.ExecutablePath } else { '?' }
            Write-Host ("  {0,-10} :{1}  LISTENING  pid {2}  {3}" -f $pair.N, $pair.P, $owner, $exe) -ForegroundColor Green
        } else {
            Write-Host ("  {0,-10} :{1}  DOWN" -f $pair.N, $pair.P) -ForegroundColor Red
        }
    }

    # 2. what the live processes actually resolved to (not what we assume)
    Write-Host ""
    foreach ($half in @('collector', 'ui')) {
        $f = Join-Path $RunDir "$half.json"
        if (-not (Test-Path -LiteralPath $f)) { Write-Host "  no runtime state for $half"; continue }
        $s = Get-Content -LiteralPath $f -Raw -Encoding utf8 | ConvertFrom-Json
        # A bare "does this PID exist" is not enough: after a reboot Windows
        # reissues low PIDs freely, so an unrelated process would be reported as
        # our live supervisor. Require the identity AND that the process is not
        # NEWER than the state file that claims to describe it.
        $p = Get-ProcInfo ([int]$s.pid)
        $alive = $false
        if ($p -and [int]$s.pid -gt 4 -and $p.Name -eq 'powershell.exe' -and
            ([string]$p.CommandLine) -like "*run-$half*") {
            try {
                $created = ([datetime]$p.CreationDate).ToUniversalTime()
                $stamped = ([datetime]$s.started_utc).ToUniversalTime()
                $alive = $created -le $stamped.AddSeconds(5)
            } catch { $alive = $false }
        }
        $tag = if     ($alive) { 'supervisor alive' }
               elseif ($p)     { 'STALE STATE FILE (pid reused)' }
               else            { 'SUPERVISOR GONE' }
        Write-Host ("  {0,-10} mode={1} {2} (pid {3}, since {4})" -f $half, $s.mode, $tag, $s.pid, $s.started_utc)
        if ($half -eq 'collector') { Write-Host ("             db={0}" -f $s.db) }
        if ($half -eq 'ui')        { Write-Host ("             api={0}" -f $s.api_url) }
        if ($s.mode -ne $Cfg.Mode) {
            Write-Host ("             WARNING: running in '{0}' but you asked about '{1}'" -f $s.mode, $Cfg.Mode) -ForegroundColor Yellow
        }
    }

    # 3. unauthenticated liveness — proves almost nothing, so say so
    Write-Host ""
    $h = Invoke-Probe "http://127.0.0.1:$CollectorPort/health" $null
    Write-Host ("  /health            -> {0} {1}" -f $h.Code, ($h.Body -replace '\s+', ' '))
    $u = Invoke-Probe "http://127.0.0.1:$UiPort/_stcore/health" $null
    Write-Host ("  /_stcore/health    -> {0} {1}" -f $u.Code, ($u.Body -replace '\s+', ' '))

    # 4. the checks that DO prove something: auth + real rows
    $admin = Get-AdminToken $Cfg.TokensFile
    if (-not $admin) {
        Write-Host "  /admin/progress    -> SKIPPED (no admin token in $($Cfg.TokensFile))" -ForegroundColor Yellow
    } else {
        $pr = Invoke-Probe "http://127.0.0.1:$CollectorPort/admin/progress?split=$($Cfg.Split)" $admin
        if ($pr.Code -eq 200) {
            Write-Host ("  /admin/progress    -> 200 {0}" -f ($pr.Body -replace '\s+', ' ')) -ForegroundColor Green
        } else {
            Write-Host ("  /admin/progress    -> {0} {1}" -f $pr.Code, $pr.Body) -ForegroundColor Red
            Write-Host "     a 401 here with a 200 from /health means the token table is empty (auth.py:39-42)" -ForegroundColor Yellow
        }
    }

    $r = Get-DbCounts $Cfg.Db
    switch ($r.State) {
        'ok' {
            $c = $r.Counts
            Write-Host ("  db rows            -> annotations={0} events={1} splits=[{2}] annotators=[{3}]" -f $c.annotations, $c.events, ($c.splits -join ','), ($c.annotators -join ','))
        }
        'dberror'   { Write-Host ("  db UNREADABLE at {0}: {1}" -f $Cfg.Db, $r.Detail) -ForegroundColor Red }
        'probefail' { Write-Host ("  db PROBE FAILED (the database may be fine): {0}" -f $r.Detail) -ForegroundColor Red }
        'nofile'    { Write-Host ("  db rows            -> NO DATABASE at {0}" -f $Cfg.Db) -ForegroundColor Red }
    }
    if ($r.State -ne 'nofile') {
        $fi = Get-Item -LiteralPath $Cfg.Db
        # The byte count is the second diagnostic: a real initialised but empty
        # schema is ~20 KB, not 0.
        Write-Host ("  db file            -> {0} bytes, last written {1}" -f $fi.Length, $fi.LastWriteTime)
    }

    $strays = Get-StrayLocalAnnotations
    if ($strays.Count -gt 0) {
        Write-Host ""
        Write-Host ("  LOCAL-MODE WORK FOUND: {0} file(s) in DATA\gold_standard\annotations\" -f $strays.Count) -ForegroundColor Red
        foreach ($s in $strays) {
            Write-Host ("    {0}  ({1} bytes, {2})" -f $s.Name, $s.Length, $s.LastWriteTime) -ForegroundColor Red
        }
        Write-Host "    These bypassed the collector (a UI visit with no ?token=) and sit where" -ForegroundColor Red
        Write-Host "    gold_standard/iaa.py reads. Import or archive them; do not ignore them." -ForegroundColor Red
    }

    # 5. token leak check on the logs
    foreach ($half in @('collector', 'ui')) {
        $log = Join-Path $LogDir "$half.log"
        if (-not (Test-Path -LiteralPath $log)) { continue }
        $hits = @(Select-String -LiteralPath $log -Pattern 'token=' -SimpleMatch -ErrorAction SilentlyContinue)
        if ($hits.Count -gt 0) {
            Write-Host ("  LOG LEAK: {0} contains {1} 'token=' occurrence(s). Rotate it out and keep --log-level warning." -f $log, $hits.Count) -ForegroundColor Red
        }
    }

    # 6. autostart
    Write-Host ""
    $any = $false
    foreach ($t in @($TaskCollector, $TaskUi)) {
        $task = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $t -ErrorAction SilentlyContinue
        if (-not $task) { continue }
        $any = $true
        $info = Get-ScheduledTaskInfo -TaskPath $TaskFolder -TaskName $t
        $res = $info.LastTaskResult
        $note = if ($res -eq 267009) { '(267009 = still running: healthy)' }
                elseif ($res -eq 267011) { '(267011 = has not run yet)' }
                else { '' }
        Write-Host ("  task {0,-22} {1,-8} lastRun={2} result={3} {4}" -f $t, $task.State, $info.LastRunTime, $res, $note)
    }
    if (-not $any) {
        Write-Host "  autostart          -> NOT INSTALLED (run: .\run_annotation_stack.ps1 install -Mode $($Cfg.Mode))" -ForegroundColor Yellow
    }
    Write-Host ""
}

function Invoke-Install {
    param([pscustomobject]$Cfg)

    $self = Join-Path $Repo 'run_annotation_stack.ps1'
    $ps   = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $me   = "$env:USERDOMAIN\$env:USERNAME"

    # The one refusal that belongs at install time. A scheduled task inherits
    # User-scope variables, and GS_API_TOKEN would make every annotator
    # authenticate as one identity (app.py:101) and UPSERT over each other.
    foreach ($scope in @('User', 'Machine')) {
        if ([Environment]::GetEnvironmentVariable('GS_API_TOKEN', $scope)) {
            Write-Error "GS_API_TOKEN is set in the $scope environment. A scheduled task would inherit it and collapse all three annotators onto one identity. Remove it before installing."
            exit 2
        }
    }

    # Defaults that would bite a laptop: a task does NOT start on battery, is
    # STOPPED when the mains plug is pulled, and is killed after 72 hours.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)
    $settings.IdleSettings.StopOnIdleEnd = $false
    $settings.IdleSettings.RestartOnIdle = $false
    $settings.RunOnlyIfNetworkAvailable  = $false

    # Interactive/Limited needs no elevation, but it DOES need someone to sign
    # in after a reboot. See the README section this script is documented in.
    $principal = New-ScheduledTaskPrincipal -UserId $me -LogonType Interactive -RunLevel Limited

    $plan = @(
        @{ Name = $TaskCollector; Half = 'run-collector'; Delay = 'PT20S'
           Desc = "Gold-standard annotation collector (uvicorn, .venv) - mode=$($Cfg.Mode)" },
        @{ Name = $TaskUi;        Half = 'run-ui';        Delay = 'PT50S'
           Desc = "Gold-standard annotator UI (streamlit, .venv-ml) - mode=$($Cfg.Mode)" }
    )

    foreach ($p in $plan) {
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $me
        $trigger.Delay = $p.Delay
        $argStr = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}" {1} -Mode {2}' -f $self, $p.Half, $Cfg.Mode
        $action = New-ScheduledTaskAction -Execute $ps -Argument $argStr -WorkingDirectory $Repo
        Register-ScheduledTask -TaskName $p.Name -TaskPath $TaskFolder `
            -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
            -Description $p.Desc -Force | Out-Null
        Write-Host "  registered $TaskFolder$($p.Name)  (mode=$($Cfg.Mode), delay $($p.Delay))" -ForegroundColor Green
    }

    # Registration succeeding does NOT mean the tasks can start. The task path
    # deliberately omits -Init, so it refuses on a missing DB, tokens file or
    # sample — all three of which are state, not code. Check them and tell the
    # truth. (Not Test-Preflight: its port checks would refuse the install
    # whenever a stack is already running, which is the normal case.)
    $pending = @()
    foreach ($need in @(
        @{ What = 'tokens file'; Path = $Cfg.TokensFile },
        @{ What = 'sample CSV';  Path = (Join-Path $Repo ("DATA\gold_standard\{0}.csv" -f $Cfg.Split)) },
        @{ What = 'database';    Path = $Cfg.Db })) {
        if (-not (Test-Path -LiteralPath $need.Path)) {
            $pending += ("{0}: {1}" -f $need.What, $need.Path)
        }
    }

    Write-Host ""
    if ($pending.Count -eq 0) {
        Write-Host "Installed. Both halves now start when $me signs in." -ForegroundColor Cyan
    } else {
        Write-Host "Installed, but the tasks will REFUSE to start at logon until these exist:" -ForegroundColor Yellow
        foreach ($p in $pending) { Write-Host "  - $p" -ForegroundColor Yellow }
        Write-Host "  fix with:  .\run_annotation_stack.ps1 start -Mode $($Cfg.Mode) -Init" -ForegroundColor Yellow
    }
    Write-Host "  verify:    .\run_annotation_stack.ps1 status"
    Write-Host "  remove:    .\run_annotation_stack.ps1 uninstall"
    Write-Host ""
    Write-Host "What this does NOT cover (measured on this machine, 2026-11-17):" -ForegroundColor Yellow
    Write-Host "  1. A reboot nobody signs in after. There is no autologon, and Windows Update" -ForegroundColor Yellow
    Write-Host "     active hours are 09->01, so an unattended restart can land at 01:00-09:00" -ForegroundColor Yellow
    Write-Host "     and leave the stack down until someone signs in. Sign in and run 'status'" -ForegroundColor Yellow
    Write-Host "     before each annotation day." -ForegroundColor Yellow
    Write-Host "  2. Battery. The active power scheme never sleeps on AC but sleeps after 180 s" -ForegroundColor Yellow
    Write-Host "     on battery, and this machine's only sleep state disconnects the network, so" -ForegroundColor Yellow
    Write-Host "     an unplugged laptop becomes unreachable while the processes stay healthy." -ForegroundColor Yellow
    Write-Host "     Keep it on mains for the pilot." -ForegroundColor Yellow
    Write-Host "  3. LAN reachability of :8501. The firewall allowance that matches today is" -ForegroundColor Yellow
    Write-Host "     scoped to C:\Python313\python.exe on the Public profile. Rebuilding a venv" -ForegroundColor Yellow
    Write-Host "     or joining a network Windows calls Private can break it, and a hidden task" -ForegroundColor Yellow
    Write-Host "     has no desktop to show the firewall prompt on. Test from a second device." -ForegroundColor Yellow
    Write-Host "  4. A hung-but-listening process. Nothing here detects that; 'status' would" -ForegroundColor Yellow
    Write-Host "     show everything green. Watch /admin/progress during the pilot, not /health." -ForegroundColor Yellow
    Write-Host "  5. Disk failure. D: and C: are partitions of ONE physical NVMe, so this" -ForegroundColor Yellow
    Write-Host "     survives a reboot, not a dead disk. Run 'backup' daily during the pilot." -ForegroundColor Yellow
}

function Invoke-Uninstall {
    foreach ($t in @($TaskCollector, $TaskUi)) {
        $task = Get-ScheduledTask -TaskPath $TaskFolder -TaskName $t -ErrorAction SilentlyContinue
        if (-not $task) { Write-Host "  $t not registered"; continue }
        if ($task.State -eq 'Running') { Stop-ScheduledTask -TaskPath $TaskFolder -TaskName $t }
        Unregister-ScheduledTask -TaskPath $TaskFolder -TaskName $t -Confirm:$false
        Write-Host "  unregistered $TaskFolder$t"
    }
}

function Invoke-Backup {
    param([pscustomobject]$Cfg)

    if (-not (Test-Path -LiteralPath $Cfg.Db)) {
        Write-Error "no database at $($Cfg.Db)"; exit 1
    }
    New-DirIfMissing $BakDir
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $dst = Join-Path $BakDir ("{0}_{1}.sqlite" -f $Cfg.Split, $stamp)

    # VACUUM INTO, not a file copy. store.py closes its connection per request,
    # so there is usually no -wal on disk and `cp` LOOKS correct — but copy it
    # mid-save and you get a database that opens cleanly, passes
    # integrity_check, and is silently missing the newest commits.
    $code = @'
import pathlib, sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
c = sqlite3.connect(src)
c.execute("PRAGMA busy_timeout=15000")
c.execute("VACUUM INTO ?", (dst,))
c.close()
print("ok", dst)
'@
    New-DirIfMissing $RunDir
    $tmp = Join-Path $RunDir '_vacuum.py'
    Set-Content -LiteralPath $tmp -Value $code -Encoding utf8
    & $PyWeb -I $tmp $Cfg.Db $dst
    if ($LASTEXITCODE -ne 0) { Write-Error "backup failed"; exit 1 }

    $before = Get-DbCounts $Cfg.Db
    $after  = Get-DbCounts $dst
    if ($before.State -ne 'ok' -or $after.State -ne 'ok') {
        Write-Host ("  WARNING: could not verify the snapshot (live={0}, snapshot={1}). The file was written but its row counts are unconfirmed." -f $before.State, $after.State) -ForegroundColor Red
        Write-Host "  -> $dst" -ForegroundColor Yellow
        return
    }
    $b = $before.Counts
    $a = $after.Counts
    Write-Host ("  live     annotations={0} events={1}" -f $b.annotations, $b.events)
    Write-Host ("  snapshot annotations={0} events={1}" -f $a.annotations, $a.events)
    if ($b.annotations -ne $a.annotations) {
        Write-Host "  (rows were written during the snapshot; that is expected on a live DB)" -ForegroundColor Yellow
    }
    Write-Host "  -> $dst" -ForegroundColor Green
}

# ----------------------------------------------------------------- dispatch
$cfg = Get-ModeConfig $Mode

switch ($Action) {
    'run-collector' { Invoke-Runner -Half 'collector' -Cfg $cfg }
    'run-ui'        { Invoke-Runner -Half 'ui'        -Cfg $cfg }
    'start'         { Invoke-Start  -Cfg $cfg }
    'stop'          { Invoke-Stop }
    'restart'       {
        # -Mode defaults to 'pilot', but 'stop' is mode-independent. A bare
        # 'restart' against a running dry run would therefore stop the dry run
        # and start the PILOT — crossing the contamination boundary in one word.
        # Read the live mode before Invoke-Stop deletes the state file.
        $liveMode = $null
        $stateFile = Join-Path $RunDir 'collector.json'
        if (Test-Path -LiteralPath $stateFile) {
            try { $liveMode = (Get-Content -LiteralPath $stateFile -Raw -Encoding utf8 | ConvertFrom-Json).mode } catch { }
        }
        $portsBound = (Get-PortOwnerPid $CollectorPort) -or (Get-PortOwnerPid $UiPort)
        $modeGiven = $PSBoundParameters.ContainsKey('Mode')

        if (-not $liveMode -and $portsBound -and -not $modeGiven) {
            Write-Host "refusing: something is listening on :$CollectorPort/:$UiPort but there is no runtime state in $RunDir, so the live mode is unknown. Run 'status' first, or 'restart -Mode <pilot|dryrun>' explicitly." -ForegroundColor Red
            exit 2
        }
        if ($liveMode -and $liveMode -ne $Mode) {
            if ($modeGiven) {
                Write-Host "refusing: the live stack is '$liveMode' but you asked for '$Mode'. Crossing the mode boundary needs an explicit 'stop' then 'start -Mode $Mode'." -ForegroundColor Red
                exit 2
            }
            Write-Host "no -Mode given; restarting in the live mode '$liveMode'" -ForegroundColor Yellow
            $cfg = Get-ModeConfig $liveMode
        }
        Invoke-Stop
        Start-Sleep -Seconds 2
        Invoke-Start -Cfg $cfg
    }
    'status'        { Invoke-Status -Cfg $cfg }
    'install'       { Invoke-Install -Cfg $cfg }
    'uninstall'     { Invoke-Uninstall }
    'backup'        { Invoke-Backup -Cfg $cfg }
}
