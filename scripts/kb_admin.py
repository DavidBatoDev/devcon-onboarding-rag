"""Knowledge-base operations against s3://debbie-kb-<account> (profile `default`).

  status                      live version, previous, versions present, docs/ listing
  upload-index <local-dir>    upload a locally built/exported version dir to index/<name>/
  promote <version>           point index/current.json at a version (rollback = promote the old one)
  seed-docs                   upload the KB source texts (app/storage/corpus/docs.jsonl) to docs/
                              as .txt + docs/manifest.json (titles, Drive file ids/links)

The API picks up a pointer change within INDEX_REFRESH_SECONDS (5 min) on warm
instances and immediately on cold ones. Usage (from server/):
  .venv/Scripts/python scripts/kb_admin.py status
"""

import argparse
import json
import re
import sys
from pathlib import Path

import boto3

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))
from app.services import index_format as fmt  # noqa: E402

REGION, ACCOUNT = "ap-southeast-1", "484907520476"
BUCKET = f"debbie-kb-{ACCOUNT}"


def s3():
    session = boto3.Session(profile_name="default", region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        sys.exit("refusing: profile `default` is not the DEBBIE account")
    return session.client("s3")


def status(c):
    try:
        ptr = json.loads(c.get_object(Bucket=BUCKET, Key="index/current.json")["Body"].read())
    except c.exceptions.NoSuchKey:
        ptr = None
    versions = sorted({o["Key"].split("/")[1] for o in
                       c.list_objects_v2(Bucket=BUCKET, Prefix="index/").get("Contents", [])
                       if o["Key"].count("/") >= 2})
    docs = [o["Key"] for o in c.list_objects_v2(Bucket=BUCKET, Prefix="docs/").get("Contents", [])]
    print(json.dumps({"pointer": ptr, "versions": versions, "docs": docs}, indent=2))


def upload_index(c, local_dir):
    vdir = Path(local_dir)
    manifest = json.loads((vdir / fmt.MANIFEST_FILE).read_text(encoding="utf-8"))
    fmt.validate(manifest, (vdir / fmt.CHUNKS_FILE).read_bytes(), (vdir / fmt.EMBEDDINGS_FILE).read_bytes(),
                 expected_dim=manifest.get("dim", 1536))
    for name, ctype in ((fmt.CHUNKS_FILE, "application/x-ndjson"), (fmt.EMBEDDINGS_FILE, "application/octet-stream"),
                        (fmt.MANIFEST_FILE, "application/json")):
        c.put_object(Bucket=BUCKET, Key=f"index/{vdir.name}/{name}", Body=(vdir / name).read_bytes(),
                     ContentType=ctype)
    print(f"uploaded index/{vdir.name}/ ({manifest['count']} chunks)")


def promote(c, version):
    m = json.loads(c.get_object(Bucket=BUCKET, Key=f"index/{version}/{fmt.MANIFEST_FILE}")["Body"].read())
    get = lambda n: c.get_object(Bucket=BUCKET, Key=f"index/{version}/{n}")["Body"].read()  # noqa: E731
    fmt.validate(m, get(fmt.CHUNKS_FILE), get(fmt.EMBEDDINGS_FILE), expected_dim=m.get("dim", 1536))
    try:
        prev = json.loads(c.get_object(Bucket=BUCKET, Key="index/current.json")["Body"].read()).get("version")
    except c.exceptions.NoSuchKey:
        prev = None
    c.put_object(Bucket=BUCKET, Key="index/current.json", ContentType="application/json",
                 Body=json.dumps({"version": version, "previous": prev}).encode())
    print(f"current.json -> {version} (was {prev})")


def seed_docs(c):
    sidecar = {}
    for line in open(SERVER / "app" / "storage" / "corpus" / "docs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        title = re.sub(r"\.docx$", "", d["title"]).strip()
        name = re.sub(r'[\\/:*?"<>|]+', "-", title)[:120] + ".txt"
        c.put_object(Bucket=BUCKET, Key=f"docs/{name}", Body=d["text"].encode("utf-8"),
                     ContentType="text/plain; charset=utf-8")
        sidecar[name] = {"file_id": d["file_id"], "title": title, "drive_link": d.get("drive_link", ""),
                         "modified_time": d.get("modified_time", "")}
    c.put_object(Bucket=BUCKET, Key="docs/manifest.json", ContentType="application/json",
                 Body=json.dumps(sidecar, indent=2, ensure_ascii=False).encode("utf-8"))
    print(f"seeded {len(sidecar)} documents + docs/manifest.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("seed-docs")
    sub.add_parser("upload-index").add_argument("local_dir")
    sub.add_parser("promote").add_argument("version")
    a = ap.parse_args()
    c = s3()
    {"status": lambda: status(c), "seed-docs": lambda: seed_docs(c),
     "upload-index": lambda: upload_index(c, a.local_dir), "promote": lambda: promote(c, a.version)}[a.cmd]()


if __name__ == "__main__":
    main()
