#!/usr/bin/env python3
"""Create + deploy the Trainsect HF Space (habib-analyst/trainsect-demo).

Auth: surrogate credential via dynamic_credentials helper, sent as
``Authorization: Bearer hsurr:...`` ONLY to huggingface.co.
Transport: curl subprocess (Python HTTP clients hang on huggingface.co
through this VM's egress proxy). Never prints, logs, or persists the
real credential.
"""
import json
import os
import subprocess
import sys
import urllib.parse

sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
from dynamic_credentials import dynamic_credential_entry  # noqa: E402

CRED = "custom.huggingface"
SPACE = "habib-analyst/trainsect-demo"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE}"
CURL = ["curl", "-s", "--max-time", "300"]
HERE = os.path.dirname(os.path.abspath(__file__))


def _surrogate() -> str:
    return dynamic_credential_entry(CRED)["surrogate"]


def _curl(args, **kw):
    out = subprocess.run(CURL + args, capture_output=True, text=True, **kw)
    return out


def space_exists() -> bool:
    out = _curl(["-o", "/dev/null", "-w", "%{http_code}",
                 f"https://huggingface.co/api/spaces/{SPACE}"])
    return out.stdout.strip() == "200"


def create_space() -> None:
    payload = json.dumps(
        {"type": "space", "name": "trainsect-demo",
         "sdk": "gradio", "private": False}
    )
    out = _curl(["-X", "POST", "https://huggingface.co/api/repos/create",
                 "-H", f"Authorization: Bearer {_surrogate()}",
                 "-H", "Content-Type: application/json",
                 "-d", payload])
    print("create_space ->", out.stdout[:500])
    if out.returncode != 0 or '"error"' in out.stdout:
        raise RuntimeError(f"space creation failed: {out.stdout[:500]}")


def upload_file(local_path: str, repo_path: str) -> None:
    url = ("https://huggingface.co/api/spaces/"
           f"{SPACE}/upload/main/{urllib.parse.quote(repo_path)}")
    out = _curl(["-X", "POST", url,
                 "-H", f"Authorization: Bearer {_surrogate()}",
                 "-H", "Content-Type: application/octet-stream",
                 "--data-binary", f"@{local_path}"])
    body = out.stdout.strip()
    ok = out.returncode == 0 and ("oid" in body or "sha" in body or body == "")
    print(f"upload {repo_path} -> {'OK' if ok else 'CHECK: ' + body[:300]}")
    if not ok:
        raise RuntimeError(f"upload failed for {repo_path}: {body[:300]}")


def main() -> None:
    if space_exists():
        print(f"Space already exists: {SPACE_URL}")
    else:
        create_space()
        print(f"Space created: {SPACE_URL}")
    files = ["app.py", "requirements.txt", "README.md",
             "demo_report.json", "demo_transcript.txt"]
    for f in files:
        upload_file(os.path.join(HERE, f), f)
    pkg = os.path.join(HERE, "trainsect")
    for fn in sorted(os.listdir(pkg)):
        if fn.endswith(".py"):
            upload_file(os.path.join(pkg, fn), f"trainsect/{fn}")
    print(f"Deployed: {SPACE_URL}")


if __name__ == "__main__":
    main()
