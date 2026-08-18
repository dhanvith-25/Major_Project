$env:SATARKA_PORT = if ($env:SATARKA_PORT) { $env:SATARKA_PORT } else { "8081" }
python -m app.main
