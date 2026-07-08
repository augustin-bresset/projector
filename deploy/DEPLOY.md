# Deploying the Projector web demo

> This deploy config (`Dockerfile`, `render.yaml`, this file) lives on the
> **`demo` branch**, kept separate from `main`. Point your host at the `demo`
> branch; keep it in sync with `git checkout demo && git merge main` when the
> app changes.

The web server is a plain FastAPI app, packaged as a small Docker image
(`Dockerfile` at the repo root — no pywebview / Qt, only the `api` extra:
`numpy` + `fastapi` + `uvicorn[standard]`). The **same image** runs on any
container host, so you are not locked into one provider. It boots on the
synthetic `demo` source — **zero external data** — and binds `0.0.0.0` on
`$PORT` (hosts inject their own; defaults to 7860 locally).

```bash
docker build -t projector .
docker run --rm -p 7860:7860 projector   # http://127.0.0.1:7860
```

The container runs as uid 1000 with a writable `$HOME`, so session sidecars
(layout, bookmarks, position) land under `~/.local/share/projector` and are
ephemeral. The demo serves a **single shared session**, so concurrent visitors
drive the same viewer.

## Render — free, simplest (recommended)

Render's free web-service tier builds Docker straight from the repo. Two ways:

**Blueprint (infra-as-code, `render.yaml` is already in the repo):**

1. Push the `demo` branch to GitHub (`git push -u origin demo`).
2. Render → **New → Blueprint** → pick the repo, **branch `demo`**. It reads
   `render.yaml` (service `projector-demo`, Docker, free plan) and deploys.

**Or by hand:** Render → **New → Web Service** → connect the repo → **branch
`demo`** → it auto-detects the `Dockerfile` → **Instance type: Free** → Create.

Either way the demo comes up at `https://projector-demo.onrender.com` (the URL
wired into the README badge — rename the service if you want a different host).
Free services **spin down after ~15 min idle** and wake on the next request
(~1 min cold start); 750 instance-hours/month per workspace.

## Google Cloud Run — free, more headroom (alternative)

Always-Free tier (2M requests/month), scales to zero. Needs the `gcloud` CLI and
billing enabled on the project (you stay within the free quota):

```bash
gcloud run deploy projector-demo --source . --port 7860 \
    --allow-unauthenticated --region europe-west1
```

Cloud Run injects `$PORT=8080`; the Dockerfile's CMD already honors it, so the
`--port` flag above just matches the container's `EXPOSE`.

## Any other Docker host / VPS (same image)

```bash
docker run -d -p 80:7860 --restart unless-stopped projector
```

> **Note on Hugging Face Spaces:** HF now restricts Docker (and Gradio) Spaces to
> paid plans — only **Static** Spaces stay free, and Projector needs a Python
> backend (websocket frame stream + numpy engine), so it can't run as a Static
> Space without a large client-side rewrite. Hence Render / Cloud Run above.
