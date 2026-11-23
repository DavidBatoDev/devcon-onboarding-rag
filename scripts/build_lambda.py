"""Build the two Lambda packages (arm64, Python 3.13) into server/build/.

    build/api/       FastAPI app + deps, started by Lambda Web Adapter via run.sh
    build/indexer/   indexer handler + shared modules + deps + bundled tiktoken BPE

Terraform zips these directories (archive_file). Works on Linux CI and on
Windows: wheels are fetched for manylinux2014_aarch64 regardless of the host.

Usage (from server/):  python scripts/build_lambda.py [api|indexer|all]
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
BUILD = SERVER / "build"
# Lambda python3.13 = Amazon Linux 2023 (glibc 2.34): manylinux_2_28 wheels are fine
# (numpy >= 2.3 publishes only those for aarch64), manylinux2014 as the fallback tag.
PLATFORM = ["--platform", "manylinux_2_28_aarch64", "--platform", "manylinux2014_aarch64",
            "--implementation", "cp", "--python-version", "3.13", "--only-binary=:all:"]
# Pure-Python packages published only as sdists: installed without the platform pin.
SDIST_ONLY = {"docx2txt"}

API_FILES = ["main.py", "run.sh", "app"]
INDEXER_FILES = ["indexer", "app/__init__.py", "app/core/config.py",
                 "app/services/chunking.py", "app/services/index_format.py", "app/services/llm.py"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "storage")


def pip_install(requirements: Path, target: Path) -> None:
    lines = [l.split("#")[0].strip() for l in requirements.read_text().splitlines()]
    reqs = [l for l in lines if l and not l.startswith("-")]
    binary = [r for r in reqs if r.split("==")[0].lower() not in SDIST_ONLY]
    sdist = [r for r in reqs if r.split("==")[0].lower() in SDIST_ONLY]
    pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
           "--no-compile", "--target", str(target)]
    subprocess.run(pip + PLATFORM + binary, check=True)
    if sdist:
        subprocess.run(pip + ["--no-deps"] + sdist, check=True)


def copy_sources(files, target: Path) -> None:
    for rel in files:
        src = SERVER / rel
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def normalise_scripts(target: Path) -> None:
    run = target / "run.sh"
    if run.exists():  # LF endings or bash fails on Lambda ("\r: command not found")
        run.write_bytes(run.read_bytes().replace(b"\r\n", b"\n"))


def prune(target: Path) -> None:
    for p in list(target.rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)
    for name in ("bin",):  # console-script shims built for the host OS
        shutil.rmtree(target / name, ignore_errors=True)


def bundle_tiktoken(target: Path) -> None:
    """Pre-download the cl100k_base BPE into the package so the indexer never
    fetches it at runtime (Lambda env: TIKTOKEN_CACHE_DIR=/var/task/tiktoken_cache)."""
    cache = target / "tiktoken_cache"
    cache.mkdir(exist_ok=True)
    env = {**os.environ, "TIKTOKEN_CACHE_DIR": str(cache)}
    subprocess.run([sys.executable, "-c", "import tiktoken; tiktoken.get_encoding('cl100k_base')"],
                   check=True, env=env)


def build(name: str) -> None:
    target = BUILD / name
    for _ in range(5):  # Windows: AV/indexer locks can make the first rmtree partial
        shutil.rmtree(target, ignore_errors=True)
        if not target.exists():
            break
        time.sleep(1)
    target.mkdir(parents=True, exist_ok=True)
    if name == "api":
        pip_install(SERVER / "requirements.txt", target)
        copy_sources(API_FILES, target)
        normalise_scripts(target)
    else:
        pip_install(SERVER / "requirements-indexer.txt", target)
        copy_sources(INDEXER_FILES, target)
        bundle_tiktoken(target)
    prune(target)
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"built {target.relative_to(SERVER)}  ({size / 1e6:.1f} MB unzipped)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (["api", "indexer"] if which == "all" else [which]):
        build(n)
