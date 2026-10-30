API FastAPI (v2). Defina `ATLAS_KEY` e, se quiser, `ATLAS_ORIGINS` (domínios permitidos, separados por vírgula).
Endpoints: `GET /health`, `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel` (os três últimos exigem o cabeçalho `X-Atlas-Key`).
Local: `pip install -r requirements.txt` + PyTorch/Diffusers com CUDA; `ATLAS_KEY=segredo uvicorn api:app --port 7860`.
