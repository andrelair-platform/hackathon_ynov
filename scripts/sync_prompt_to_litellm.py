#!/usr/bin/env python3
"""
sync_prompt_to_litellm.py
=========================
Fetches the production-labelled phi3-financial-system prompt from Langfuse,
updates the ConfigMap in minicloud-gitops, and pushes to git so ArgoCD deploys
the change within ~3 minutes.

Usage:
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://langfuse.devandre.sbs    # Cloudflare Tunnel URL
    # Optional overrides:
    export GITOPS_REPO=~/Developer/cloudplateform/minicloud-gitops
    python scripts/sync_prompt_to_litellm.py

Rollback workflow:
    1. In Langfuse UI → Prompt Management → phi3-financial-system
    2. Click the version to roll back to → set label "production"
    3. Re-run this script
    4. ArgoCD auto-syncs the ConfigMap within ~3 minutes
    5. Restart the LiteLLM deployment to reload env vars:
       kubectl --context minicloud rollout restart deployment/litellm -n ai

Exit codes:
    0 — success (ConfigMap updated + git push done, or prompt unchanged)
    1 — error
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from langfuse import Langfuse

# ── Configuration ─────────────────────────────────────────────────────────────

LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST       = os.environ["LANGFUSE_HOST"]

PROMPT_NAME   = "phi3-financial-system"
LABEL         = "production"
GITOPS_REPO   = Path(os.environ.get("GITOPS_REPO",
    Path(__file__).parent.parent.parent / "minicloud-gitops"))
CONFIGMAP_FILE = GITOPS_REPO / "manifests" / "ai" / "14-phi3-financial-prompt-configmap.yaml"

GIT_USER_NAME  = "AndreLiar"
GIT_USER_EMAIL = "andrelaurelyvan.kanmegnetabouguie@ynov.com"

# ── Helpers ────────────────────────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[git] ERROR: {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout.strip()


def _extract_version_from_labels(labels: list[str]) -> str:
    """Return the vX.Y.Z label if present, else 'unknown'."""
    for label in labels:
        if re.match(r"^v\d+\.\d+\.\d+$", label):
            return label
    return "unknown"


def fetch_production_prompt() -> tuple[str, str]:
    """Return (prompt_text, version_label) from Langfuse."""
    lf = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )
    prompt = lf.get_prompt(PROMPT_NAME, label=LABEL)
    text    = prompt.prompt
    version = _extract_version_from_labels(getattr(prompt, "labels", []))
    print(f"[langfuse] Fetched '{PROMPT_NAME}' (label={LABEL}, version={version})")
    return text, version


def build_configmap(prompt_text: str, version: str) -> str:
    """Render the ConfigMap YAML with the given prompt embedded."""
    # Indent each line of the prompt by 4 spaces for the YAML block scalar
    indented = textwrap.indent(prompt_text, "    ")
    # Ensure trailing newline after block scalar
    if not indented.endswith("\n"):
        indented += "\n"

    return (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: phi3-financial-prompt\n"
        "  namespace: ai\n"
        "  annotations:\n"
        "    # This ConfigMap is the runtime source of truth for the phi3-financial system prompt.\n"
        "    # It is synced from Langfuse Prompt Management (label: production) by running:\n"
        "    #   python scripts/sync_prompt_to_litellm.py   (from hackathon_ynov repo)\n"
        "    # To roll back: flip the production label to an older version in Langfuse UI,\n"
        "    # then re-run sync_prompt_to_litellm.py — ArgoCD deploys within ~3 minutes.\n"
        f"    prompt.minicloud/source: \"langfuse:{PROMPT_NAME}:{LABEL}\"\n"
        f"    prompt.minicloud/version: \"{version}\"\n"
        "data:\n"
        "  PHI3_FINANCIAL_SYSTEM_PROMPT: |\n"
        f"{indented}"
    )


def main() -> int:
    print(f"\n=== sync_prompt_to_litellm  [{PROMPT_NAME}:{LABEL}] ===\n")

    # 1. Fetch production prompt from Langfuse
    prompt_text, version = fetch_production_prompt()

    # 2. Render target ConfigMap content
    new_content = build_configmap(prompt_text, version)

    # 3. Compare with current file
    if CONFIGMAP_FILE.exists():
        current = CONFIGMAP_FILE.read_text()
        # Strip only the annotation line that would change (version/source) for comparison
        # by comparing prompt body content
        old_prompt_block = ""
        new_prompt_block = ""
        for block, text in [("old", current), ("new", new_content)]:
            start = text.find("  PHI3_FINANCIAL_SYSTEM_PROMPT: |")
            if start != -1:
                if block == "old":
                    old_prompt_block = text[start:]
                else:
                    new_prompt_block = text[start:]

        if old_prompt_block == new_prompt_block:
            print("[sync] Prompt content unchanged — nothing to push.")
            return 0
        print("[sync] Prompt content changed — updating ConfigMap.")
    else:
        print(f"[sync] ConfigMap file not found at {CONFIGMAP_FILE} — creating.")

    # 4. Write updated ConfigMap
    CONFIGMAP_FILE.write_text(new_content)
    print(f"[sync] Written: {CONFIGMAP_FILE}")

    # 5. Git commit and push
    run(["git", "config", "user.name",  GIT_USER_NAME],  cwd=GITOPS_REPO)
    run(["git", "config", "user.email", GIT_USER_EMAIL], cwd=GITOPS_REPO)

    run(["git", "add", str(CONFIGMAP_FILE)], cwd=GITOPS_REPO)

    status = run(["git", "status", "--porcelain"], cwd=GITOPS_REPO)
    if not status:
        print("[git] Nothing staged — ConfigMap file already matches git HEAD.")
        return 0

    commit_msg = (
        f"feat(ai): sync phi3-financial system prompt {version} from Langfuse\n\n"
        f"Source: Langfuse '{PROMPT_NAME}' label=production version={version}\n"
        f"Synced by: scripts/sync_prompt_to_litellm.py"
    )
    run(["git", "commit", "-S", "-m", commit_msg], cwd=GITOPS_REPO)
    run(["git", "push", "origin", "main"], cwd=GITOPS_REPO)

    print(f"\n[sync] Done — ArgoCD will deploy the updated ConfigMap in ~3 minutes.")
    print(f"       To force LiteLLM to reload: kubectl --context minicloud rollout restart deployment/litellm -n ai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
