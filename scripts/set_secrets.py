"""Write DEBBIE's SSM parameters from your machine (never via Terraform, so the
values are not in Terraform state or the repo).

  /debbie/prod/openai_api_key   SecureString  <- OPENAI_API_KEY from server/.env (or --openai-key-env)
  /debbie/prod/alert_email      String        <- --alert-email (SNS alarm/budget subscription)

Usage (from server/, AWS profile `default`):
  .venv/Scripts/python scripts/set_secrets.py --alert-email you@example.com
  .venv/Scripts/python scripts/set_secrets.py --rotate-openai   # after creating a new key
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from dotenv import dotenv_values

REGION = "ap-southeast-1"
ACCOUNT = "484907520476"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert-email")
    ap.add_argument("--rotate-openai", action="store_true", help="only (re)write the OpenAI key")
    ap.add_argument("--openai-key-env", default="OPENAI_API_KEY")
    args = ap.parse_args()

    session = boto3.Session(profile_name="default", region_name=REGION)
    if session.client("sts").get_caller_identity()["Account"] != ACCOUNT:
        sys.exit("refusing: profile `default` is not the DEBBIE account")
    ssm = session.client("ssm")

    env = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ.get(args.openai_key_env) or env.get(args.openai_key_env)
    if not key:
        sys.exit(f"{args.openai_key_env} not found in the environment or server/.env")
    ssm.put_parameter(Name="/debbie/prod/openai_api_key", Value=key, Type="SecureString",
                      Overwrite=True, Description="OpenAI API key for debbie-api / debbie-indexer")
    print("wrote /debbie/prod/openai_api_key (SecureString, aws/ssm key)")

    if args.alert_email and not args.rotate_openai:
        ssm.put_parameter(Name="/debbie/prod/alert_email", Value=args.alert_email, Type="String",
                          Overwrite=True, Description="Email for DEBBIE alarms and budget")
        print("wrote /debbie/prod/alert_email")


if __name__ == "__main__":
    main()
