# phi3-financial — Enterprise PromptOps Pipeline

[![CI](https://github.com/andrelair-platform/phi3-financial/actions/workflows/prompt-eval.yml/badge.svg)](https://github.com/andrelair-platform/phi3-financial/actions/workflows/prompt-eval.yml)
[![Eval: 25/25](https://img.shields.io/badge/eval-25%2F25%20PASS-brightgreen)](https://github.com/andrelair-platform/phi3-financial/actions/workflows/prompt-eval.yml)
[![Prompt: v2.2.4](https://img.shields.io/badge/prompt-v2.2.4-blue)](scripts/eval.py)
[![Supply chain: cosign](https://img.shields.io/badge/supply%20chain-cosign%20signed-green)](https://github.com/sigstore/cosign)

> A production PromptOps pipeline for a domain-restricted AI financial assistant. Prompt versioning, runtime injection, PII masking, and a 25-case behavioral eval suite — all running on a self-hosted Kubernetes cluster.

**Live demo:** [chat.devandre.sbs](https://chat.devandre.sbs) — protected by Authentik SSO  
**Portfolio:** [devandre.sbs/en](https://www.devandre.sbs/en)

---

## Screenshots

| Live chat (sub-1s via Groq) | Langfuse Prompt Management |
|---|---|
| ![Chat](docs/screenshots/screenshot-chat.png) | ![Prompts](docs/screenshots/screenshot-langfuse-prompts.png) |

| Langfuse Eval Traces | Langfuse Eval Dataset |
|---|---|
| ![Traces](docs/screenshots/screenshot-langfuse-traces.png) | ![Dataset](docs/screenshots/screenshot-langfuse-dataset.png) |

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Stack](#stack)
- [CI/CD Pipeline](#cicd-pipeline)
- [Eval Results](#eval-results)
- [Prompt Rollback](#prompt-rollback)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Runtime prompt injection** — LiteLLM `CustomLogger` fetches the production-labelled prompt from Langfuse on every request; 5-min in-process cache, no Langfuse call per request
- **Three-layer fail-open** — live Langfuse → stale cache → ConfigMap env var; LiteLLM never goes down because Langfuse is unreachable
- **25-case behavioral eval suite** — CI deployment gate enforcing 100% pass rate across financial knowledge, off-topic refusal, prompt injection resistance, and mixed-domain edge cases
- **PII/DLP guardrail** — Presidio masks PERSON, PHONE, EMAIL, IP_ADDRESS before logging to Langfuse
- **GitOps prompt sync** — `sync_prompt_to_litellm.py` pushes a GPG-signed commit to `minicloud-gitops`; ArgoCD deploys in ~3 minutes
- **Prompt rollback under 5 minutes** — flip the `production` label in Langfuse UI; no git push, no pod restart required
- **Supply chain security** — eval CI uses Tailscale OAuth + Cosign-signed LiteLLM image from Harbor

---

## Architecture

```
User request
    │
    ▼
LiteLLM (routing + virtual model: phi3-financial)
    │   └── LangfusePromptHandler (async_pre_call_hook)
    │           │  1. Fetch production-labelled prompt from Langfuse (cached 5 min)
    │           │  2. Strip any user-supplied system message (anti-injection)
    │           │  3. Inject prompt as sole system message
    │           └── Fail-open: stale cache → ConfigMap env var → no injection
    │
    ├── Presidio guardrail (pre_call) — PII masking before trace logging
    │
    ▼
Ollama (phi4-mini + Modelfile XML guardrails)
    │
    ▼
Langfuse (trace + score every response)
```

**Prompt update flow:**

```
Langfuse UI → edit prompt → set label 'production'
    │
    ├── (fast path) run sync_prompt_to_litellm.py
    │       └── GPG-signed commit to minicloud-gitops → ArgoCD → ConfigMap updated in ~3 min
    │
    └── (cache path) wait ≤5 min
            └── LangfusePromptHandler fetches new version automatically
```

---

## Key Components

### `langfuse_prompt_handler.py` — LiteLLM CustomLogger

Core of the PromptOps pipeline. Implements `async_pre_call_hook` from LiteLLM's `CustomLogger` interface:

- Fetches the `production`-labelled prompt from Langfuse Prompt Management at request time
- 5-minute in-process cache per LiteLLM worker pod — no Langfuse call on every inference request
- Three-layer fail-open: live Langfuse → stale cache → `LITELLM_SYSTEM_PROMPT` ConfigMap env var
- Strips any user-supplied `system` role message before injecting the production prompt — prevents guardrail bypass via direct API calls

Deployed as a Kubernetes ConfigMap mounted into the LiteLLM pod. Updating the handler logic requires no image rebuild — only a ConfigMap update + rollout restart.

### `ollama_server/Modelfile` — XML Prompt Architecture

System prompt structured in 5 behavioral sections with explicit off-topic taxonomy:

```xml
<system_intent>               — persona and domain scope
<domain_knowledge_constraints>— financial topics IN; explicit taxonomy OUT:
                               medical (diseases, medications)
                               culinary (recipes, cooking)
                               sports/entertainment
                               gambling (casino games, betting odds, wagering)
<safety_and_guardrails>       — credential handling, market manipulation refusal
<jailbreak_defenses>          — 5 named patterns:
                               FIXED RUNTIME, UNTRUSTED INPUT, INJECTION RESPONSE,
                               PROMPT ISOLATION, PERSISTENCE
<output_format>               — response style and length constraints
```

Off-topic taxonomy is explicit (named sub-items) rather than a blanket "stay on topic" — each boundary is independently testable in the eval suite.

### `scripts/eval.py` — 25-case Behavioral Eval Suite

CI deployment gate. Runs on every push to `Modelfile` or `eval.py`, weekly on schedule, and on-demand.

| Case group | Count | What it tests |
|---|---|---|
| Financial knowledge | 10 | P/E ratio, VaR/CVaR, yield curve, Basel III, options, EBITDA, duration |
| Off-topic refusal | 4 | Medical, culinary, gambling/casino, sports betting |
| Prompt injection | 6 | System override, roleplay, credential phishing, translation bypass, prompt extraction |
| Edge / mixed-domain | 5 | Healthcare stocks (answer as investment topic), market manipulation refusal |

Scoring: keyword presence/absence per case — deterministic, no LLM-as-judge, no API cost. Every case logs a Langfuse trace + correctness score for full audit trail in Langfuse.

**Infrastructure retry logic:** three-tier exponential backoff (10s / 30s / 60s) on 5xx/connection errors — handles ~90s downtime from an ArgoCD `Recreate` restart overlapping a CI run.

### `scripts/sync_prompt_to_litellm.py` — PromptOps Sync

Fetches the `production`-labelled prompt from Langfuse, renders it into a Kubernetes ConfigMap YAML, commits to `minicloud-gitops` with a GPG-signed commit, and pushes — ArgoCD deploys within ~3 minutes.

---

## Stack

| Component | Role |
|---|---|
| Ollama (phi4-mini) | Model serving — Modelfile bakes XML guardrails at the Ollama layer |
| LiteLLM | Routing, virtual model `phi3-financial`, CustomLogger callback registration |
| Langfuse | Prompt Management (versioning + labels), LLMOps tracing, eval dataset |
| Kubernetes (k3s) | Workload orchestration — LiteLLM + Ollama in `ai` namespace, 5-node cluster |
| Valkey | Prompt cache backend shared across LiteLLM worker pods |
| pgvector + bge-m3 | RAG layer — financial document embeddings for context injection |
| Presidio | PII/DLP guardrail — masks entities before logging to Langfuse |
| GitHub Actions | Eval CI — blocks merge on any regression below 100% pass rate |

---

## CI/CD Pipeline

Every push to `main` touching `Modelfile` or `scripts/eval.py` triggers `.github/workflows/prompt-eval.yml`:

```
push / PR / weekly schedule
    │
    ├─ 1. Connect to Tailscale (OAuth — TS_OAUTH_CLIENT_ID / TS_OAUTH_SECRET)
    ├─ 2. Trust minicloud CA (raw PEM appended to certifi + system store)
    ├─ 3. pip install langfuse openai
    ├─ 4. curl litellm.devandre.sbs/health — abort if unreachable
    ├─ 5. python scripts/eval.py           — exits 1 on any regression
    └─ 6. git tag v<PROMPT_VERSION>        — on clean main push only
```

**Required secrets** (all org-level on `andrelair-platform` — inherited automatically):

| Secret | Purpose |
|---|---|
| `TS_OAUTH_CLIENT_ID` | Tailscale OAuth client ID — joins tailnet as `tag:ci` |
| `TS_OAUTH_SECRET` | Tailscale OAuth secret |
| `MINICLOUD_CA_CERT` | Raw PEM of the minicloud self-signed CA — trusted for in-cluster HTTPS |
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `LANGFUSE_HOST` | Langfuse instance URL |
| `LITELLM_API_KEY` | LiteLLM virtual key scoped to the `phi3-financial` model |
| `LITELLM_BASE_URL` | LiteLLM public URL (`https://litellm.devandre.sbs`) |

> **MINICLOUD_CA_CERT is stored as raw PEM** — never pipe it through `base64 -d`. Use `echo "${{ secrets.MINICLOUD_CA_CERT }}" | sudo tee ...` directly.

---

## Eval Results

```
25/25 PASS (100%)  —  PROMPT_VERSION 2.2.4
Langfuse dataset:  phi3-financial-evals
Eval runtime:      ~7 min (Groq llama-3.1-8b-instant primary, Ollama fallback)
```

Prompt rollback time: **under 5 minutes** via Langfuse label change.

---

## Prompt Rollback

No git push, no pod restart required for the runtime path:

1. Langfuse UI → Prompt Management → `phi3-financial-system` → set old version label to `production`
2. Wait up to 5 min for the in-process cache to expire  
   *(or `kubectl rollout restart deployment/litellm -n ai` for immediate effect)*
3. All new requests automatically use the rolled-back prompt

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionDeniedError: "Your request was blocked."` from Python openai SDK | Cloudflare WAF blocks `User-Agent: AsyncOpenAI/Python*` | Pass `default_headers={"User-Agent": "your-app/1.0"}` in the `OpenAI()` constructor |
| Langfuse returns empty body / `JSONDecodeError` | minicloud CA not trusted — `base64 -d` on raw PEM secret | Use `echo "$MINICLOUD_CA_CERT" \| sudo tee ...` with no decode |
| Eval times out (>60 min) | Retry backoff triggered: 10s+30s+60s per case × 25 cases | Check LiteLLM pod health; a `Recreate` restart overlapping a run will trigger retries |
| `APIConnectionError` on internal URL | `tag:ci` Tailscale ACL does not route to `10.0.0.200:443` | Use the Cloudflare public URL (`litellm.devandre.sbs`) not the internal nip.io one |

---

## License

[MIT](LICENSE) © andrelair-platform
