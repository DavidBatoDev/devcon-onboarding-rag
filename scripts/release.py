"""Release gate for debbie-api: smoke-test a published Lambda version directly,
then point the `live` alias (what API Gateway invokes) at it.

  release.py                  smoke-test the newest published version, then move `live`
  release.py --version 7      same, for a specific version
  release.py --rollback       move `live` back to the version it pointed at before
  release.py --check-only     smoke-test without moving the alias

The smoke test invokes the version with InvokeWithResponseStream and an API
Gateway-shaped event (exactly what the STREAM integration sends), so a version
that can't start, can't read its index/secret, or can't stream an answer never
receives traffic. Credentials: default chain (profile `default` locally, OIDC in CI).
"""

import argparse
import json
import sys
import time

import boto3

FUNCTION, ALIAS, REGION = "debbie-api", "live", "ap-southeast-1"


def event(method, path, body=None, accept="application/json"):
    headers = {"Accept": accept, "Content-Type": "application/json", "Host": "release-smoke.local"}
    return {
        "resource": "/api/v1/{proxy+}", "path": path, "httpMethod": method,
        "headers": headers, "multiValueHeaders": {k: [v] for k, v in headers.items()},
        "queryStringParameters": None, "multiValueQueryStringParameters": None,
        "pathParameters": {"proxy": path.split("/api/v1/", 1)[-1]}, "stageVariables": None,
        "requestContext": {"resourcePath": "/api/v1/{proxy+}", "httpMethod": method, "path": f"/prod{path}",
                           "stage": "prod", "requestId": f"release-{int(time.time())}",
                           "identity": {"sourceIp": "192.0.2.10", "userAgent": "release-smoke"}},
        "body": json.dumps(body) if body is not None else None, "isBase64Encoded": False,
    }


def invoke(lam, version, ev):
    resp = lam.invoke_with_response_stream(FunctionName=FUNCTION, Qualifier=version,
                                           Payload=json.dumps(ev).encode())
    raw = b""
    for chunk in resp["EventStream"]:
        if "PayloadChunk" in chunk:
            raw += chunk["PayloadChunk"]["Payload"]
        if "InvokeComplete" in chunk and chunk["InvokeComplete"].get("ErrorCode"):
            raise RuntimeError(f"invoke error: {chunk['InvokeComplete']}")
    prelude, sep, body = raw.partition(b"\x00" * 8)
    if not sep:
        raise RuntimeError(f"no streaming prelude in response: {raw[:200]!r}")
    return json.loads(prelude), body.decode("utf-8", "replace")


def smoke(lam, version, ask=True):
    meta, body = invoke(lam, version, event("GET", "/api/v1/health"))
    health = json.loads(body)
    if meta.get("statusCode") != 200 or health.get("status") != "healthy" or not health.get("index_version"):
        raise RuntimeError(f"/health failed: {meta.get('statusCode')} {body[:300]}")
    print(f"  health ok: index {health['index_version']}, model {health.get('model')}")
    if not ask:
        return
    meta, body = invoke(lam, version, event("POST", "/api/v1/ask/stream",
                                            {"query": "How do I apply for the DEVCON internship?",
                                             "structured": True, "nocache": True},
                                            accept="application/x-ndjson"))
    events = [json.loads(l) for l in body.split("\n") if l.strip()]
    types = [e["type"] for e in events]
    if meta.get("statusCode") != 200 or not types or types[0] != "meta" or types[-1] != "done" \
            or "token" not in types:
        raise RuntimeError(f"/ask/stream failed: {meta.get('statusCode')} events={types[:8]} {body[:300]}")
    grounded = next(e for e in events if e["type"] == "sources")["grounded"]
    print(f"  ask/stream ok: {types.count('token')} token events, grounded={grounded}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--no-ask", action="store_true", help="skip the (one OpenAI call) answer test")
    a = ap.parse_args()
    lam = boto3.client("lambda", region_name=REGION)
    alias = lam.get_alias(FunctionName=FUNCTION, Name=ALIAS)
    current = alias["FunctionVersion"]
    previous = (alias.get("Description") or "").removeprefix("previous=") or None

    if a.rollback:
        if not previous:
            sys.exit("no previous version recorded on the alias")
        target = previous
    else:
        versions = [v["Version"] for v in lam.list_versions_by_function(FunctionName=FUNCTION)["Versions"]
                    if v["Version"] != "$LATEST"]
        target = a.version or max(versions, key=int)

    print(f"{FUNCTION}: live={current} target={target}")
    try:
        smoke(lam, target, ask=not a.no_ask)
    except Exception as e:
        print(f"SMOKE TEST FAILED for version {target}: {e}\n`live` stays on {current}.")
        sys.exit(1)
    if a.check_only or target == current:
        print("alias unchanged")
        return
    lam.update_alias(FunctionName=FUNCTION, Name=ALIAS, FunctionVersion=target,
                     Description=f"previous={current}")
    print(f"`live` -> {target} (rollback: release.py --rollback -> {current})")


if __name__ == "__main__":
    main()
