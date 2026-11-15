# Download, verify, and install vork on Windows x64 without gh or admin rights.
#   & ([scriptblock]::Create((irm https://github.com/voyager-so/vork/releases/latest/download/install.ps1)))
# Environment: VERSION, PREFIX (default %LOCALAPPDATA%\Programs\vork), VORK_REPO, VORK_DOWNLOAD_BASE;
# VORK_SKIP_SETUP=1 skips `vork setup` (GPU-free CI); VORK_NO_MODIFY_PATH=1 leaves the user PATH alone.
# VORK_MODIFY_PATH=1 opts a custom PREFIX into PATH edits; VORK_NO_MODIFY_PATH=1 always disables edits.
param(
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = if ($env:VORK_REPO) { $env:VORK_REPO } else { 'voyager-so/vork' }
# Default version is generated from Cargo.toml by scripts/sync-version.
$version = if ($env:VERSION) { $env:VERSION } else { '0.6.0' }
# Contract: PREFIX -> a fully qualified, lexically normalized install directory,
# without trailing separators, or a refusal before downloads/writes. Relative and
# drive-relative forms use the PowerShell filesystem location (not process cwd);
# root-relative forms use that location's drive/share root. No path must exist.
function Resolve-InstallPrefix([string]$value) {
    $refusal = 'PREFIX must name an install directory, not an empty path or a drive or share root. Use a folder such as D:\vork. Nothing was downloaded or changed.'
    if ([string]::IsNullOrWhiteSpace($value) -or $value -match '^[\\/]+$') { throw $refusal }
    $location = Get-Location
    if ($location.Provider.Name -ne 'FileSystem') { throw 'PREFIX requires a PowerShell filesystem location. Nothing was downloaded or changed.' }
    $basePath = $location.ProviderPath
    if ([IO.Path]::DirectorySeparatorChar -eq '/') {
        # Keep native paths usable for GPU-free installer tests on Unix.
        if (-not $value.StartsWith('/')) { $value = Join-Path $basePath $value }
    } elseif ($value -match '^[A-Za-z]:[\\/]' -or $value -match '^\\\\[^\\/]+[\\/][^\\/]+') {
        # Drive-absolute and UNC paths already carry their own root.
    } elseif ($value -match '^[\\/]') {
        $value = Join-Path ([IO.Path]::GetPathRoot($basePath)) $value.TrimStart('\', '/')
    } else {
        # C:vork follows the same explicit location rule as vork; never use the
        # process's hidden per-drive directory or leave a colon inside a child.
        $value = Join-Path $basePath ($value -replace '^[A-Za-z]:', '')
    }
    $destination = [IO.Path]::GetFullPath($value)
    $root = [IO.Path]::GetPathRoot($destination)
    $destination = $destination.TrimEnd('\', '/')
    if ($destination -eq $root.TrimEnd('\', '/')) { throw $refusal }
    return $destination
}
$prefix = if (Test-Path Env:PREFIX) { $env:PREFIX } else { Join-Path $env:LOCALAPPDATA 'Programs\vork' }
$prefix = Resolve-InstallPrefix $prefix
if ($version -notmatch '^[A-Za-z0-9._-]+$') { throw 'Invalid VERSION' }
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($arch -ne 'AMD64') { throw "Supported platforms: Windows x64, Apple Silicon macOS and Linux x64; found $arch" }
$archive = "vork-$version-windows-x64.zip"
$base = if ($env:VORK_DOWNLOAD_BASE) { $env:VORK_DOWNLOAD_BASE.TrimEnd('/') } else { "https://github.com/$repo/releases/download/v$version" }
function Quote-ShellLiteral([string]$value) {
    "'" + $value.Replace("'", "''") + "'"
}
$bin = $prefix
$quotedBin = Quote-ShellLiteral $bin
if ($DryRun) {
    "Archive: $base/$archive"
    "Checksums: $base/SHA256SUMS"
    "Destination: $prefix"
    "Binary: $bin\vork.exe"
    return
}
function Get-Release([string]$name, [string]$target) {
    $url = "$base/$name"
    if ($url -match '^file:') {
        try { Copy-Item -LiteralPath ([Uri]$url).LocalPath -Destination $target; return }
        catch { throw "Could not copy $url to $target ($($_.Exception.Message)). Check free space and permissions; your existing install was not changed." }
    }
    for ($attempt = 0; $attempt -lt 4; $attempt++) {
        $response = $null; $inputStream = $null; $outputStream = $null
        try {
            # Header/connect and idle-read deadlines, never a whole-download limit.
            $request = [Net.HttpWebRequest]::Create($url)
            $request.Timeout = 30000
            $request.ReadWriteTimeout = 120000
            $response = $request.GetResponse()
            $inputStream = $response.GetResponseStream()
            $outputStream = [IO.File]::Open($target, [IO.FileMode]::Create)
            $buffer = New-Object byte[] 65536
            $received = 0L
            while (($count = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $outputStream.Write($buffer, 0, $count)
                $received += $count
            }
            if ($response.ContentLength -ge 0 -and $received -ne $response.ContentLength) {
                throw [Net.WebException]::new("Interrupted download: received $received of $($response.ContentLength) bytes")
            }
            return
        } catch {
            if ($attempt -eq 3) {
                throw "Could not download $url to $target ($($_.Exception.Message)). Check connection, free space and permissions. Headers wait at most 30 seconds; idle reads wait at most 120 seconds. Your existing install was not changed. Run this installer again: https://github.com/$repo/releases"
            }
            Write-Warning "Download failed ($($_.Exception.Message)); retrying $url"
        } finally {
            if ($outputStream) { $outputStream.Dispose() }
            if ($inputStream) { $inputStream.Dispose() }
            if ($response) { $response.Dispose() }
        }
        Start-Sleep -Seconds ([math]::Pow(2, $attempt))
    }
}
# Add a folder to the user PATH, keeping unexpanded entries such as %USERPROFILE%.
function Add-UserPath([string]$folder) {
    $key = [Microsoft.Win32.Registry]::CurrentUser.CreateSubKey('Environment')
    try {
        $raw = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (($raw -split ';') -contains $folder) { return }
        $key.SetValue('Path', ($folder + $(if ($raw) { ";$raw" })), [Microsoft.Win32.RegistryValueKind]::ExpandString)
    } finally { $key.Close() }
    # Setting any user variable tells Explorer to reload the environment for new windows.
    [Environment]::SetEnvironmentVariable('VORK_INSTALL_NOTIFY', '1', 'User')
    [Environment]::SetEnvironmentVariable('VORK_INSTALL_NOTIFY', $null, 'User')
}
$work = Join-Path ([IO.Path]::GetTempPath()) ("vork-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$stage = "$prefix.new-$PID"
$retired = "$prefix.old-$PID"
try {
    $zip = Join-Path $work $archive
    Get-Release $archive $zip
    Get-Release 'SHA256SUMS' (Join-Path $work 'SHA256SUMS')
    $rows = @(Get-Content (Join-Path $work 'SHA256SUMS') | Where-Object {
            $f = -split $_; $f.Count -eq 2 -and ($f[1] -eq $archive -or $f[1] -eq "*$archive") })
    if ($rows.Count -ne 1) { throw "Missing or ambiguous checksum for $archive; nothing was installed." }
    $expected = (-split $rows[0])[0]
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash
    if ($actual -ne $expected) { throw "Checksum mismatch for ${archive}: the download is damaged or altered. Nothing was installed; run this again." }
    "${archive}: OK"
    # Refuse traversal paths before extracting a verified archive.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $z = [IO.Compression.ZipFile]::OpenRead($zip)
    try {
        foreach ($e in $z.Entries) {
            $n = $e.FullName -replace '\\', '/'
            if ($n -match '^/|^[A-Za-z]:|(^|/)\.\.(/|$)') { throw 'Unsafe archive path' }
        }
    } finally { $z.Dispose() }
    # Build the new tree beside the old one and swap it in whole: files a release dropped
    # do not survive an upgrade, and a failed install leaves the old one working.
    $running = @(Get-Process -Name vork -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $bin 'vork.exe') })
    if ($running.Count) { throw "vork is still running from $prefix (process $($running[0].Id)). Let it finish or close it, then run this again. Your existing install was not changed." }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $prefix) | Out-Null
    try { Expand-Archive -LiteralPath $zip -DestinationPath $stage }
    catch { throw "Could not extract vork into $stage ($($_.Exception.Message)). Check free space and write permissions: the expanded release must fit beside $prefix, and the archive occupies $work. Your existing install was not changed." }
    $stageExe = Join-Path $stage 'vork.exe'
    try { $stagedVersionOutput = & $stageExe --version 2>&1 }
    catch { throw "Cannot execute $stageExe ($($_.Exception.Message)). Check platform compatibility, runtime libraries and execution permissions. Your existing install was not changed." }
    if ($LASTEXITCODE -ne 0) { throw "Staged vork --version failed (exit $LASTEXITCODE): $stagedVersionOutput; your existing install was not changed." }
    if (Test-Path -LiteralPath $prefix) {
        try { Move-Item -LiteralPath $prefix -Destination $retired }
        catch { throw "Could not replace $prefix ($($_.Exception.Message)) Close anything using that folder and run this again. Your existing install was not changed." }
    }
    Move-Item -LiteralPath $stage -Destination $prefix
    $exe = Join-Path $bin 'vork.exe'
    $quotedExe = Quote-ShellLiteral $exe
    & $exe --version
    if ($LASTEXITCODE -ne 0) { throw 'vork --version failed' }
    # Let a new window find vork. Custom prefixes require explicit opt-in.
    $pathNote = $null
    if ($env:VORK_NO_MODIFY_PATH -eq '1' -or ($env:PREFIX -and $env:VORK_MODIFY_PATH -ne '1')) {
        if (-not (($env:Path -split ';') -contains $bin)) {
            $pathNote = "Add it to PATH with:`n  [Environment]::SetEnvironmentVariable('Path', $quotedBin + ';' + [Environment]::GetEnvironmentVariable('Path', 'User'), 'User')"
        }
    } else {
        Add-UserPath $bin
        if (-not (($env:Path -split ';') -contains $bin)) { $env:Path = "$bin;$env:Path" }
        $pathNote = "New PowerShell windows find vork. In a window that is already open, run:`n  `$env:Path = $quotedBin + ';' + `$env:Path"
    }
    if ($env:VORK_SKIP_SETUP -eq '1') {
        'Skipping vork setup (VORK_SKIP_SETUP=1); run vork setup before rendering.'
    } else {
        'Preparing shaders for this GPU; native progress reports the current phase...'
        & $exe setup
        if ($LASTEXITCODE -ne 0) {
            if ($pathNote) { $pathNote }
            throw "vork is installed at $exe, but setup failed. Follow the report above, then run: & $quotedExe setup"
        }
    }
    "Installed to $exe."
    if ($pathNote) { $pathNote }
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    $keepRetired = $false
    if ((Test-Path -LiteralPath $retired) -and -not (Test-Path -LiteralPath $prefix)) {
        try { Move-Item -LiteralPath $retired -Destination $prefix }
        catch {
            $keepRetired = $true
            Write-Warning "Could not restore the previous install ($($_.Exception.Message)); it is preserved at $retired"
        }
    }
    if (-not $keepRetired) { Remove-Item -LiteralPath $retired -Recurse -Force -ErrorAction SilentlyContinue }
}
