# Deploying the Projector web demo

The web server is a plain FastAPI app, packaged as a small Docker image
(`Dockerfile` at the repo root — no pywebview / Qt, only the `api` extra:
`numpy` + `fastapi` + `uvicorn[standard]`). The **same image** runs on any
container host, so you are not locked into one provider. It boots on the
synthetic `demo` source — **zero external data**.

```bash
docker build -t projector .
docker run --rm -p 7860:7860 projector   # http://127.0.0.1:7860
```

The container runs as uid 1000 with a writable `$HOME`, so session sidecars
(layout, bookmarks, position) land under `~/.local/share/projector` and are
ephemeral.

## Hugging Face Spaces (this branch)

This `hf` branch **is** the Space: it carries the `Dockerfile`, the standard
`.gitattributes` (LFS), and a `README.md` whose YAML frontmatter is the Space
metadata (that block lives only here, never on `main`).

1. Create a new Space → **SDK: Docker** (blank), named `projector`.
2. Push this branch to the Space's git remote as its `main`:

   ```bash
   git remote add space https://huggingface.co/spaces/SmaugC137/projector
   git push space hf:main
   ```

HF builds the `Dockerfile` and serves the app on port 7860 (declared as
`app_port` in the README frontmatter). Free Spaces sleep after inactivity and
wake on the next visit (~30 s). The demo serves a **single shared session**, so
concurrent visitors drive the same viewer.

## Other hosts (same Dockerfile)

- **Render** — New → Web Service → from repo → "Docker"; set the port to 7860.
- **Fly.io** — `fly launch` (detects the Dockerfile), then `fly deploy`.
- **Google Cloud Run** — `gcloud run deploy projector --source . --port 7860`.
- **Any VPS** — `docker run -p 80:7860 projector`.
