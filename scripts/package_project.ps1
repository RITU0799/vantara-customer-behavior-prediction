param(
    [string]$OutputPath = (Join-Path (Split-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path -Parent) 'Vantara_Customer_Behavior_Prediction_Project.zip')
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$archivePath = [System.IO.Path]::GetFullPath($OutputPath)
$excludedDirectories = @('.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', '.pytest-temp', 'tmp', 'htmlcov', 'logs', 'node_modules')
$excludedFileNames = @('.env', '.coverage', 'secrets.toml')

if (Test-Path -LiteralPath $archivePath) {
    throw "Output archive already exists; choose another path or move it first: $archivePath"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Create)
$pendingDirectories = [System.Collections.Generic.Stack[System.IO.DirectoryInfo]]::new()
$pendingDirectories.Push([System.IO.DirectoryInfo]::new($projectRoot))
$rootName = Split-Path $projectRoot -Leaf

try {
    while ($pendingDirectories.Count -gt 0) {
        $directory = $pendingDirectories.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory.FullName -Force) {
            if ($item.PSIsContainer) {
                if ($item.Name -notin $excludedDirectories) {
                    $pendingDirectories.Push([System.IO.DirectoryInfo]::new($item.FullName))
                }
                continue
            }

            if ($item.Name -in $excludedFileNames -or $item.Name -like '*.db' -or $item.Name -like '*.sqlite*' -or $item.Name -like '*.zip') {
                continue
            }

            $relativePath = $item.FullName.Substring($projectRoot.Length).TrimStart('\').Replace('\', '/')
            $entryName = "$rootName/$relativePath"
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive,
                $item.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
    }
}
finally {
    $archive.Dispose()
}

Write-Output "Created: $archivePath"
