# ATLAS IMAGE AI: API de geração (FastAPI). Roda no Colab; o notebook a publica por um túnel HTTPS.
import base64, binascii, gc, io, os, queue, random, secrets, threading, time, uuid
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import torch

KEY = os.environ.get("ATLAS_KEY", "")
ORIGINS = [o for o in os.environ.get("ATLAS_ORIGINS", "*").split(",") if o]
app = FastAPI(title="ATLAS IMAGE AI API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Atlas-Key"])

MODELS = {
    "SDXL 1.0 base": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "kind": "sdxl"},
    "Playground v2.5": {"repo": "playgroundai/playground-v2.5-1024px-aesthetic", "kind": "playground"},
    "RealVisXL V5.0": {"repo": "SG161222/RealVisXL_V5.0", "kind": "sdxl"},
    "Animagine XL 4.0": {"repo": "cagliostrolab/animagine-xl-4.0", "kind": "sdxl"},
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
    init_image: str = Field(default="", max_length=14_000_000)
    strength: float = Field(default=0.6, ge=0.05, le=1.0)
    upscale: float = Field(default=1.0, ge=1.0, le=2.0)

    @field_validator("init_image", mode="before")
    @classmethod
    def _validate_image_payload(cls, v):
        if not v:
            return ""
        if not isinstance(v, str) or len(v) > 14_000_000:
            raise ValueError("Imagem muito grande. Limite de envio: 10 MB.")
        if not v.startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
            raise ValueError("Formato inválido. Envie PNG, JPEG ou WebP.")
        payload = v.split(",", 1)[1]
        if len(payload) > 14_000_000:
            raise ValueError("Imagem muito grande. Limite de envio: 10 MB.")
        return v


def auth(x_atlas_key: str = Header(default="")):
    if not KEY or not x_atlas_key or not secrets.compare_digest(x_atlas_key, KEY):
        raise HTTPException(401, "Chave inválida. Use o link de conexão que o notebook mostrou.")


JOBS, Q = {}, queue.Queue()
ST = {"pipe": None, "name": None, "tok": None, "mdl": None, "i2i": None}


def to_en(text):
    if ST["mdl"] is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        n = "Helsinki-NLP/opus-mt-ROMANCE-en"
        ST["tok"], ST["mdl"] = AutoTokenizer.from_pretrained(n), AutoModelForSeq2SeqLM.from_pretrained(n)
    b = ST["tok"]([text], return_tensors="pt", truncation=True, max_length=256)
    out = ST["mdl"].generate(**b, max_new_tokens=256, num_beams=4)
    return ST["tok"].decode(out[0], skip_special_tokens=True)


LLM = {"tok": None, "mdl": None}
SYS = ("You write prompts for text-to-image models. Translate the user's idea to English if needed, then expand it into ONE rich, concrete prompt "
       "(subject, setting, lighting, composition, camera or lens, style, mood) in under 70 words. Keep every detail the user gave. "
       "Do not add text, logos or extra people unless asked. Output only the prompt.")


def enhance_prompt(text):
    if LLM["mdl"] is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        n = "Qwen/Qwen2.5-1.5B-Instruct"
        LLM["tok"], LLM["mdl"] = AutoTokenizer.from_pretrained(n), AutoModelForCausalLM.from_pretrained(n, torch_dtype=torch.float16)
    t, m = LLM["tok"], LLM["mdl"].to("cuda")
    chat = t.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = t(chat, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9)
    res = t.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"')
    m.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    return res or text


def load_model(name):
    if ST["name"] == name:
        return ST["pipe"]
    ST["pipe"], ST["name"], ST["i2i"] = None, None, None
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
        try:
            p = AutoPipelineForText2Image.from_pretrained(m["repo"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        except Exception:
            p = AutoPipelineForText2Image.from_pretrained(m["repo"], torch_dtype=torch.float16, use_safetensors=True)
        p = p.to("cuda")
        try:
            p.vae.enable_tiling()
        except Exception:
            pass
    if m["kind"] == "sdxl":
        from diffusers import DPMSolverMultistepScheduler
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    ST["pipe"], ST["name"] = p, name
    return p


def run(job, req):
    job.update(status="running", message="Preparando…")
    notes, kind = [], MODELS[req.model]["kind"]
    text, neg = req.prompt.strip(), req.negative_prompt.strip()
    if req.init_image and kind == "flux":
        raise ValueError("O FLUX schnell não aceita imagem de referência. Use SDXL ou Playground.")
    if req.translate:
        job["message"] = "Traduzindo para inglês…"
        try:
            text, neg = to_en(text), (to_en(neg) if neg else "")
        except Exception as e:
            notes.append(f"Tradução indisponível ({type(e).__name__}); enviei o texto original.")
    if req.style in STYLES:
        text = f"{text}, {STYLES[req.style]}"
    job["processed_prompt"] = text
    job["message"] = "Carregando o modelo (na primeira vez baixa ~7 GB: 5 a 15 min; depois, 1 a 3 min)…"
    pipe = load_model(req.model)
    steps, cfg, kw = req.steps, req.guidance, {}
    w, h = SIZES.get(req.size, SIZES["Quadrado 1:1"])
    if kind == "turbo":
        w, h, steps, cfg = w // 2, h // 2, min(steps, 4), 0.0
        notes.append("SDXL-Turbo: sem orientação nem prompt negativo, até 4 etapas.")
    elif kind == "flux":
        steps, cfg = (steps if steps <= 8 else 4), 0.0
        notes.append("FLUX schnell: sem orientação nem prompt negativo, até 8 etapas.")
    elif kind == "playground":
        cfg = min(cfg, 4.0)
        notes.append("Playground v2.5: orientação limitada a 4.")
    if kind in ("sdxl", "playground"):
        try:
            from compel import Compel, ReturnedEmbeddingsType
            c = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2], text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                       returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                       requires_pooled=[False, True], truncate_long_prompts=False)
            cond, pooled = c(text)
            ncond, npooled = c(neg or "")
            cond, ncond = c.pad_conditioning_tensors_to_same_length([cond, ncond])
            kw.update(prompt_embeds=cond, pooled_prompt_embeds=pooled, negative_prompt_embeds=ncond, negative_pooled_prompt_embeds=npooled)
        except Exception as e:
            kw.update(prompt=text, negative_prompt=neg or None)
            notes.append(f"Prompts longos indisponíveis ({type(e).__name__}); o modelo lê só 77 tokens.")
    else:
        kw["prompt"] = text
        if kind == "flux":
            kw["max_sequence_length"] = 256
    runner = pipe
    if req.init_image:
        import PIL.Image as PI
        from diffusers import AutoPipelineForImage2Image
        try:
            raw = base64.b64decode(req.init_image.split(",", 1)[1], validate=True)
            if len(raw) > 10 * 1024 * 1024:
                raise ValueError("Imagem acima do limite de 10 MB.")
            with PI.open(io.BytesIO(raw)) as check:
                if check.format not in ("PNG", "JPEG", "WEBP"):
                    raise ValueError("Formato de imagem não suportado.")
                if check.width * check.height > 16_000_000:
                    raise ValueError("A imagem excede o limite de 16 megapixels.")
                check.verify()
            img0 = PI.open(io.BytesIO(raw)).convert("RGB")
        except (binascii.Error, OSError) as e:
            raise ValueError("Não foi possível ler a imagem. Envie um PNG, JPEG ou WebP válido.") from e
        if req.upscale > 1:
            f = min(req.upscale, 1536 / max(img0.size))
        else:
            f = ((512 * 512 if kind == "turbo" else 1024 * 1024) / (img0.width * img0.height)) ** 0.5
        snap = lambda v: max(64, int(round(v / 8)) * 8)
        kw["image"], kw["strength"] = img0.resize((snap(img0.width * f), snap(img0.height * f)), PI.LANCZOS), req.strength
        if ST["i2i"] is None:
            ST["i2i"] = AutoPipelineForImage2Image.from_pipe(pipe)
        runner = ST["i2i"]
    else:
        kw.update(width=w, height=h)
    for i in range(req.count):
        if job["cancel"]:
            break
        s = random.randint(0, 2**32 - 1) if req.seed < 0 else (req.seed + i) % 2**32

        def tick(p, step, t, kwargs, i=i):
            job["progress"] = (i + (step + 1) / steps) / req.count
            job["message"] = f"Imagem {i + 1} de {req.count} · etapa {step + 1}/{steps}"
            if job["cancel"]:
                p._interrupt = True
            return kwargs

        gen = torch.Generator("cpu" if kind == "flux" else "cuda").manual_seed(s)
        img = runner(num_inference_steps=steps, guidance_scale=cfg, generator=gen, callback_on_step_end=tick, **kw).images[0]
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
            job.update(status="error", error=f"Falha ao processar ({type(e).__name__}). Verifique o modelo, a memória da GPU e tente novamente.")


threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "model": ST["name"], "models": list(MODELS), "queue": Q.qsize()}


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


class PReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)


@app.post("/prompt", dependencies=[Depends(auth)])
def improve(req: PReq):
    try:
        return {"prompt": enhance_prompt(req.prompt.strip())}
    except Exception as e:
        raise HTTPException(500, f"Não foi possível melhorar o prompt ({type(e).__name__}).")
