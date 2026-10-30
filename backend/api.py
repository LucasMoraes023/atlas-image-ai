# ATLAS IMAGE AI: API de geração (FastAPI). Roda no Colab; o notebook a publica por um túnel HTTPS.
import base64, gc, io, os, queue, random, secrets, threading, time, uuid
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

KEY = os.environ.get("ATLAS_KEY", "")
ORIGINS = [o for o in os.environ.get("ATLAS_ORIGINS", "*").split(",") if o]
app = FastAPI(title="ATLAS IMAGE AI API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Atlas-Key"])

MODELS = {
    "SDXL 1.0 base": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "kind": "sdxl"},
    "SDXL-Turbo": {"repo": "stabilityai/sdxl-turbo", "kind": "turbo"},
    "FLUX.1 schnell": {"repo": "black-forest-labs/FLUX.1-schnell", "kind": "flux"},
}
SIZES = {"Quadrado 1:1": (1024, 1024), "Retrato 3:4": (896, 1152), "Paisagem 4:3": (1152, 896), "Vertical 9:16": (768, 1344), "Wide 16:9": (1344, 768)}
STYLES = {
    "Fotografia": "photo, 35mm, natural light, sharp focus, high detail",
    "Retrato": "portrait, soft studio light, shallow depth of field, detailed face",
    "Paisagem": "wide landscape, golden hour, dramatic sky, high detail",
    "Anime": "anime style, clean line art, vibrant colors",
    "Ilustração": "digital illustration, bold shapes, painterly, detailed",
    "Fantasia": "fantasy art, epic, magical atmosphere, intricate details",
    "Arte conceitual": "concept art, cinematic composition, matte painting",
    "Render 3D": "3d render, global illumination, ultra detailed",
}


class Req(BaseModel):
    prompt: str = Field(min_length=1, max_length=1800)
    negative_prompt: str = Field(default="", max_length=800)
    model: str = "SDXL 1.0 base"
    style: str = "Nenhum"
    size: str = "Quadrado 1:1"
    steps: int = Field(default=30, ge=1, le=50)
    guidance: float = Field(default=7, ge=0, le=15)
    seed: int = Field(default=-1, ge=-1, le=2**32 - 1)
    count: int = Field(default=1, ge=1, le=4)
    translate: bool = True


def auth(x_atlas_key: str = Header(default="")):
    if not KEY or not secrets.compare_digest(x_atlas_key, KEY):
        raise HTTPException(401, "Chave inválida. Use o link de conexão que o notebook mostrou.")


JOBS, Q = {}, queue.Queue()
ST = {"pipe": None, "name": None, "tok": None, "mdl": None}


def to_en(text):
    if ST["mdl"] is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        n = "Helsinki-NLP/opus-mt-ROMANCE-en"
        ST["tok"], ST["mdl"] = AutoTokenizer.from_pretrained(n), AutoModelForSeq2SeqLM.from_pretrained(n)
    b = ST["tok"]([text], return_tensors="pt", truncation=True, max_length=256)
    out = ST["mdl"].generate(**b, max_new_tokens=256, num_beams=4)
    return ST["tok"].decode(out[0], skip_special_tokens=True)


def load_model(name):
    if ST["name"] == name:
        return ST["pipe"]
    ST["pipe"], ST["name"] = None, None
    gc.collect(); torch.cuda.empty_cache()
    m = MODELS[name]
    if m["kind"] == "flux":
        from diffusers import BitsAndBytesConfig as DQ, FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TQ, T5EncoderModel
        t = FluxTransformer2DModel.from_pretrained(m["repo"], subfolder="transformer", torch_dtype=torch.float16,
            quantization_config=DQ(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
        t5 = T5EncoderModel.from_pretrained(m["repo"], subfolder="text_encoder_2", torch_dtype=torch.float16, quantization_config=TQ(load_in_8bit=True))
        p = FluxPipeline.from_pretrained(m["repo"], transformer=t, text_encoder_2=t5, torch_dtype=torch.float16)
        p.enable_model_cpu_offload()
    else:
        from diffusers import AutoPipelineForText2Image
        p = AutoPipelineForText2Image.from_pretrained(m["repo"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    if m["kind"] == "sdxl":
        from diffusers import DPMSolverMultistepScheduler
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    ST["pipe"], ST["name"] = p, name
    return p


def run(job, req):
    job.update(status="running", message="Preparando…")
    notes, kind = [], MODELS[req.model]["kind"]
    text, neg = req.prompt.strip(), req.negative_prompt.strip()
    if req.translate:
        job["message"] = "Traduzindo para inglês…"
        try:
            text, neg = to_en(text), (to_en(neg) if neg else "")
        except Exception as e:
            notes.append(f"Tradução indisponível ({type(e).__name__}); enviei o texto original.")
    if req.style in STYLES:
        text = f"{text}, {STYLES[req.style]}"
    job["processed_prompt"] = text
    job["message"] = "Carregando o modelo (na primeira vez demora)…"
    pipe = load_model(req.model)
    w, h = SIZES.get(req.size, SIZES["Quadrado 1:1"])
    steps, cfg, extra = req.steps, req.guidance, {}
    if kind == "turbo":
        w, h, steps, cfg = w // 2, h // 2, min(steps, 4), 0.0
        notes.append("SDXL-Turbo: sem orientação, sem prompt negativo, até 4 etapas.")
    elif kind == "flux":
        steps, cfg, extra = (steps if steps <= 8 else 4), 0.0, {"max_sequence_length": 256}
        notes.append("FLUX schnell: sem orientação, sem prompt negativo, até 8 etapas.")
    else:
        extra = {"negative_prompt": neg or None}
        n = len(pipe.tokenizer(text).input_ids)
        if n > 77:
            notes.append(f"O prompt tem {n} tokens e o modelo lê só 77; o final foi cortado.")
    for i in range(req.count):
        if job["cancel"]:
            break
        s = random.randint(0, 2**32 - 1) if req.seed < 0 else (req.seed + i) % 2**32

        def tick(p, step, t, kw, i=i):
            job["progress"] = (i + (step + 1) / steps) / req.count
            job["message"] = f"Imagem {i + 1} de {req.count} · etapa {step + 1}/{steps}"
            if job["cancel"]:
                p._interrupt = True
            return kw

        gen = torch.Generator("cpu" if kind == "flux" else "cuda").manual_seed(s)
        img = pipe(prompt=text, width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=gen, callback_on_step_end=tick, **extra).images[0]
        if job["cancel"]:
            break
        buf = io.BytesIO(); img.save(buf, format="PNG")
        job["images"].append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        job["seeds"].append(s)
    job["progress"] = 1.0
    job["status"] = "cancelled" if job["cancel"] else "done"
    job["message"] = "Cancelado." if job["cancel"] else " ".join(["Concluído."] + notes)


def worker():
    while True:
        job, req = Q.get()
        try:
            run(job, req)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            job.update(status="error", error="GPU sem memória. Use SDXL-Turbo ou um formato menor.")
        except Exception as e:
            job.update(status="error", error=f"{type(e).__name__}: {e}")


threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "model": ST["name"]}


@app.post("/jobs", dependencies=[Depends(auth)])
def create(req: Req):
    if not torch.cuda.is_available():
        raise HTTPException(503, "GPU indisponível. Ative a T4 no Colab.")
    if req.model not in MODELS:
        raise HTTPException(400, "Modelo inválido.")
    if Q.qsize() >= 3:
        raise HTTPException(429, "Fila cheia. Aguarde terminar.")
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"id": jid, "status": "queued", "progress": 0.0, "message": "Na fila…", "images": [], "seeds": [], "processed_prompt": "", "error": None, "cancel": False, "created": time.time()}
    for old in sorted(JOBS.values(), key=lambda j: j["created"])[:-8]:
        JOBS.pop(old["id"], None)
    Q.put((JOBS[jid], req))
    return {"id": jid}


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "Tarefa não encontrada.")
    out = {k: v for k, v in job.items() if k not in ("cancel", "created")}
    if job["status"] != "done":
        out["images"] = []
    return out


@app.post("/jobs/{jid}/cancel", dependencies=[Depends(auth)])
def cancel(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "Tarefa não encontrada.")
    JOBS[jid]["cancel"] = True
    return {"ok": True}
