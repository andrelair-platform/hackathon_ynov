# phi3-financial — Enterprise PromptOps Pipeline

Domain-restricted AI financial assistant with a production PromptOps architecture. Built on a self-hosted Kubernetes cluster (k3s, 5 nodes) using Ollama for model serving, LiteLLM for routing and runtime prompt injection, and Langfuse for prompt management and LLMOps tracing.

**Live demo:** [chat.devandre.sbs](https://chat.devandre.sbs) — protected by Authentik SSO  
**Case study:** [devandre.sbs/en](https://www.devandre.sbs/en)

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
    ▼
Ollama (phi4-mini + Modelfile guardrails)
    │
    ▼
Langfuse (trace + score every response)
```

**Rollback in under 5 minutes — no git push, no pod restart:**

1. Langfuse UI → Prompt Management → `phi3-financial-system` → set old version as `production`
2. Wait up to 5 min for the cache to expire (or `kubectl rollout restart deployment/litellm -n ai`)
3. All new requests automatically use the rolled-back prompt

---

## Key Components

### `langfuse_prompt_handler.py` — LiteLLM CustomLogger

The core of the PromptOps pipeline. Implements `async_pre_call_hook` from LiteLLM's `CustomLogger` interface to intercept every request to the `phi3-financial` virtual model:

- Fetches the `production`-labelled prompt from Langfuse at request time
- 5-minute in-process cache per LiteLLM worker (no Langfuse call on every request)
- Three-layer fail-open chain: live Langfuse → stale cache → ConfigMap env var
- Strips any user-supplied `system` role message before injecting the production prompt (prevents guardrail bypass via API)

Deployed as a Kubernetes ConfigMap mounted into the LiteLLM pod — updating it requires no image rebuild.

### `ollama_server/Modelfile` — XML Prompt Architecture

System prompt structured in 5 behavioral sections:

```xml
<system_intent>          — persona and scope
<domain_knowledge_constraints> — financial topics IN, explicit off-topic taxonomy OUT
<safety_and_guardrails>  — credential handling, market manipulation refusal
<jailbreak_defenses>     — 5 named patterns (FIXED RUNTIME, UNTRUSTED INPUT,
                           INJECTION RESPONSE, PROMPT ISOLATION, PERSISTENCE)
<output_format>          — response style and length constraints
```

Off-topic taxonomy is explicit, not a blanket "stay on topic": medical (diseases, medications), culinary (recipes, cooking), sports/entertainment, gambling (casino games, betting odds, wagering strategies) — each with named sub-items so the eval suite can test each boundary independently.

### `scripts/eval.py` — 25-case Behavioral Eval Suite

CI deployment gate enforcing 100% pass rate. Runs on every push touching the Modelfile or eval script, weekly on schedule, and on-demand.

| Case group | Count | What it tests |
|---|---|---|
| Financial questions | 10 | Core domain coverage: P/E ratio, VaR/CVaR, yield curve, Basel III, options |
| Off-topic refusal | 4 | Medical, culinary, gambling/casino, sports betting |
| Prompt injection | 6 | System override, roleplay, credential phishing, translation bypass, prompt extraction |
| Edge / mixed-domain | 5 | Healthcare stocks (must answer as investment topic), market manipulation refusal |

Scoring: keyword presence/absence per case (no LLM-as-judge — deterministic, fast, cheap). Each case logs a Langfuse trace + correctness score for full audit trail.

**Infrastructure retry logic:** Three-tier exponential backoff (10s / 30s / 60s) on 5xx/connection errors — added after root-causing a 90-second CI outage from an ArgoCD `Recreate` restart overlapping an eval run. Fixed at two layers: LiteLLM deployment switched to `RollingUpdate` + retry logic in eval harness.

### `scripts/sync_prompt_to_litellm.py` — PromptOps Sync

Fetches the `production`-labelled prompt from Langfuse, renders it into a Kubernetes ConfigMap YAML, commits to `minicloud-gitops` with a GPG-signed commit, and pushes — triggering ArgoCD to deploy the change within ~3 minutes.

Rollback workflow:
```
Langfuse UI → flip label → run sync_prompt_to_litellm.py → ArgoCD deploys → done
```

No pod restart required for the runtime path (LangfusePromptHandler cache expires in 5 min). Pod restart only needed for immediate effect.

---

## Stack

| Component | Role |
|---|---|
| Ollama (phi4-mini) | Model serving — Modelfile bakes in XML guardrails at the Ollama layer |
| LiteLLM | Routing, virtual model `phi3-financial`, CustomLogger callback registration |
| Langfuse | Prompt Management (versioning + labels), LLMOps tracing, eval dataset |
| Kubernetes (k3s) | Workload orchestration — LiteLLM + Ollama in `ai` namespace |
| Valkey | Prompt cache backend (shared across LiteLLM workers) |
| pgvector + bge-m3 | RAG layer — financial document embeddings for context injection |
| Presidio | PII/DLP guardrail — strips sensitive entities before logging to Langfuse |
| GitHub Actions | Eval CI — blocks merge on any regression below 100% pass rate |

---

## CI

```
push / PR touching Modelfile or eval.py
  → Connect to Tailscale (OAuth)
  → Trust minicloud CA
  → pip install langfuse openai
  → python scripts/eval.py         # exits 1 if any case fails
  → git tag v<PROMPT_VERSION>      # on clean main push only
```

Weekly baseline run (Monday 06:00 UTC) catches silent model weight updates shipping under the same Ollama tag without a version bump.

---

## Eval results

```
25/25 PASS (100%)  —  PROMPT_VERSION 2.2.3
Langfuse dataset:  phi3-financial-evals
```

Prompt rollback time: **under 5 minutes** via Langfuse label change — no git push, no pod restart required.
