# Exports the deck to PDF with PowerPoint itself, and (optionally) one PNG per slide for checking.
#   pwsh docs/pitch/export_deck.ps1 -Pptx <folder>\FieldVision_model_story.pptx [-PngDir <folder>]
param(
    [Parameter(Mandatory = $true)][string]$Pptx,
    [string]$PngDir
)
$Pptx = (Resolve-Path $Pptx).Path
$pdf = [IO.Path]::ChangeExtension($Pptx, '.pdf')
$wasRunning = Get-Process POWERPNT -ErrorAction SilentlyContinue
$app = New-Object -ComObject PowerPoint.Application
try {
    $deck = $app.Presentations.Open($Pptx, -1, 0, 0)   # read-only, no window
    $deck.SaveAs($pdf, 32)                             # 32 = PDF
    if ($PngDir) {
        if (Test-Path $PngDir) { Remove-Item -Recurse -Force $PngDir }
        $deck.Export($PngDir, 'PNG', 1280, 720)
        # PowerPoint names the files in the language of the installation; give them plain names
        Get-ChildItem $PngDir | ForEach-Object {
            if ($_.Name -match '(\d+)\.png$') { Rename-Item $_.FullName ('s{0:d2}.png' -f [int]$Matches[1]) }
        }
    }
    $deck.Close()
}
finally {
    if (-not $wasRunning) { $app.Quit() }
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
"{0}: {1} KB" -f (Split-Path $pdf -Leaf), [int]((Get-Item $pdf).Length / 1KB)
