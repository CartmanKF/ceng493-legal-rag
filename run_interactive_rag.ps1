$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Turkish Legal RAG test modu"
Write-Host "Soru yaz, cikmak icin exit yaz."
Write-Host ""

while ($true) {
    $question = Read-Host "Soru"
    if ($question -eq "exit") {
        break
    }

    & ".\.conda-envs\legal-rag-gpu\python.exe" -m src.legal_rag.cli ask `
        --dataset Datasets_Ceng493_legal_rag `
        --artifacts artifacts `
        --mode fine_tuned `
        --question "$question"

    Write-Host ""
}
