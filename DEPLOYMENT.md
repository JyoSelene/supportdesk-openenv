# Deployment Guide — HF Spaces

This guide walks you through deploying SupportDesk-OpenEnv to a Hugging Face Space.

---

## Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create New Space"**
3. Fill in details:
   - **Space name:** `supportdesk-openenv` (or your preferred name)
   - **License:** MIT
   - **SDK:** Docker
   - **Space type:** Public (or private if preferred)
4. Click **"Create Space"**

You'll be taken to your new Space. Copy the **Space URL** — it will be something like:
```
https://your-username-supportdesk-openenv.hf.space
```

---

## Step 2: Clone and Push the Repository

Open your terminal and run:

```bash
cd C:\Users\Admin\Carture\supportdesk-openenv

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial OpenEnv environment submission"

# Add HF Space as remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/supportdesk-openenv

# Push to Space
git push -u space main
```

**Note:** Replace `YOUR_USERNAME` with your actual Hugging Face username.

The Space will automatically build the Docker image and start the container. This usually takes **2–5 minutes**.

---

## Step 3: Wait for Container to Start

1. Go to your Space URL
2. You should see a loading indicator while the Docker image builds
3. Once complete, you'll see the app running on the Space

To check the build status:
- Go to Space settings (⚙️ icon)
- Check **"Logs"** tab to see Docker build output

---

## Step 4: Test the Deployment

Once the Space is live, test the API:

```bash
# Health check
curl https://your-username-supportdesk-openenv.hf.space/health

# Should return:
# {"status": "ok", "active_sessions": 0}

# List tasks
curl https://your-username-supportdesk-openenv.hf.space/tasks

# Should return task metadata
```

---

## Step 5: Run Baseline Inference Against Your Space

Set the environment variables and run inference:

```bash
set SUPPORTDESK_URL=https://your-username-supportdesk-openenv.hf.space
set HF_TOKEN=hf_your_huggingface_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

**Expected output:**
```
[START] task=email_classify env=supportdesk-openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=... reward=... done=... error=...
...
[END] success=true steps=10 score=0.72 rewards=...
[START] task=email_extract ...
...
[END] success=true steps=6 score=0.55 rewards=...
[START] task=email_respond ...
...
[END] success=true steps=4 score=0.48 rewards=...
```

---

## Step 6: Pre-Submission Checklist

Before submitting, verify:

- [ ] **Space is live** — `/health` returns 200
- [ ] **Reset works** — `/reset` returns valid observation
- [ ] **Step works** — `/step` accepts actions and returns rewards
- [ ] **All tasks run** — `inference.py` completes without error
- [ ] **Scores are in range** — All rewards between 0.0–1.0
- [ ] **Docker builds** — `docker build -t supportdesk .` succeeds locally
- [ ] **README is complete** — Explains environment, tasks, API, setup
- [ ] **openenv.yaml is valid** — Run `openenv validate` (if you have openenv-core installed)

---

## Troubleshooting

### Docker build fails
- Check the **Logs** tab in Space settings
- Ensure all dependencies in `requirements.txt` are available for Python 3.11
- Try building locally: `docker build -t supportdesk .`

### Container starts but API doesn't respond
- Check Space logs for errors
- Ensure `PORT=7860` is set in environment
- Verify the Dockerfile exposes port 7860 correctly

### Inference script fails
- Verify `SUPPORTDESK_URL` is correct
- Check that HF Space is responding: `curl https://your-space/health`
- Ensure `HF_TOKEN` and `API_BASE_URL` are set for LLM calls

### Session errors ("Session not found")
- Make sure you're using the correct `session_id` from `/reset`
- Don't reuse session IDs across different Space instances

---

## Environment Variables for the Space

If you need to override defaults, set these in Space settings → Variables:

| Variable | Value |
|----------|-------|
| `PORT` | `7860` |
| `PYTHONUNBUFFERED` | `1` |

The LLM-related variables (`HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`) should be set when you run the inference script locally, not in the Space settings.

---

## What Gets Evaluated

The submission validator will:

1. **Ping your Space URL** — Expects HTTP 200
2. **Call `/reset`** — Expects valid observation back
3. **Call `/step`** — Submits an action, expects reward in [0.0, 1.0]
4. **Validate openenv.yaml** — Checks spec compliance
5. **Run inference script** — Expects [START]/[STEP]/[END] format logs
6. **Check Docker build** — Runs `docker build` on your repo
7. **Enumerate tasks** — Verifies 3+ tasks exist with graders

---

## Final Submission

Once everything is working:

1. Copy your **Space URL**
2. Go to the OpenEnv submission portal
3. Paste your URL and click **Submit**

---

## Support

- **OpenEnv docs:** https://github.com/AlignedAI/openenv-core
- **HF Spaces docs:** https://huggingface.co/docs/hub/spaces
- **Docker reference:** https://docs.docker.com/reference/

Good luck! 🚀
