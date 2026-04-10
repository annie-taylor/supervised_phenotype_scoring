#!/usr/bin/env python3
"""
upload_batch.py — Upload exported spectrograms and audio to S3 for hosted scoring.

After running export_batch.py, use this script to push the PNG and WAV files to
an S3 bucket and update config.json with the resulting base URLs so that
ranking_app.py can serve them remotely.

The public manifest (UIDs only, no bird identity) is also uploaded as
  s3://<bucket>/<prefix>/manifest.json

After upload, batch_dir/config_hosted.json is written with updated
image_base_url, audio_base_url, and mode="hosted" — ready to pass to
ranking_app.py for external scoring sessions.

Usage
-----
    python upload_batch.py E:/scoring/batches/pk24bu3_wh88br85_20260410 \\
        --bucket my-scoring-bucket \\
        --prefix scoring/pk24bu3_wh88br85

    # Dry-run (list what would be uploaded):
    python upload_batch.py <batch_dir> --bucket <bucket> --dry-run

Dependencies
------------
    pip install boto3
    AWS credentials must be configured: ~/.aws/credentials or env vars
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tables

# ── S3 upload helpers ─────────────────────────────────────────────────────────────

def get_s3_client():
    try:
        import boto3
        return boto3.client("s3")
    except ImportError:
        raise SystemExit(
            "boto3 is not installed.  Run:  pip install boto3\n"
            "Then configure AWS credentials via:\n"
            "  aws configure\n"
            "or environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )


def upload_file(s3, local_path: Path, bucket: str, key: str,
                content_type: str, dry_run: bool) -> str:
    """Upload one file; return the public HTTPS URL."""
    if dry_run:
        print(f"  [dry-run] would upload {local_path.name} → s3://{bucket}/{key}")
    else:
        s3.upload_file(
            str(local_path), bucket, key,
            ExtraArgs={
                "ContentType": content_type,
                "ACL": "public-read",
            }
        )
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def upload_directory(s3, local_dir: Path, bucket: str, prefix: str,
                     extension: str, content_type: str,
                     dry_run: bool, force: bool) -> tuple[int, int]:
    """
    Upload all files with `extension` from local_dir to s3://bucket/prefix/.
    Returns (n_uploaded, n_skipped).
    """
    files = sorted(local_dir.glob(f"*{extension}"))
    n_up = n_skip = 0

    # Fetch existing keys if not forcing (to skip already-uploaded files)
    existing: set[str] = set()
    if not force and not dry_run:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                existing.add(obj["Key"])

    for f in files:
        key = f"{prefix}/{f.name}"
        if key in existing:
            n_skip += 1
            continue
        upload_file(s3, f, bucket, key, content_type, dry_run)
        n_up += 1

    return n_up, n_skip


# ── Manifest upload ───────────────────────────────────────────────────────────────

def read_uid_list(h5_path: Path) -> list[str]:
    uid_list = []
    with tables.open_file(str(h5_path), mode="r") as h5:
        for row in h5.root.manifest.iterrows():
            uid_list.append(row["uid"].decode())
    return uid_list


def upload_manifest(s3, h5_path: Path, bucket: str, prefix: str,
                    dry_run: bool) -> None:
    uids = read_uid_list(h5_path)
    payload = json.dumps({"uids": uids}, indent=2).encode()

    key = f"{prefix}/manifest.json"
    if dry_run:
        print(f"  [dry-run] would upload manifest.json ({len(uids)} UIDs) "
              f"→ s3://{bucket}/{key}")
        return

    import boto3
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
        ACL="public-read",
    )
    print(f"  manifest.json uploaded ({len(uids)} UIDs)")


# ── Config writer ─────────────────────────────────────────────────────────────────

def write_hosted_config(batch_dir: Path, bucket: str, prefix: str) -> Path:
    """
    Write batch_dir/config_hosted.json with image/audio base URLs and mode=hosted.
    """
    base = f"https://{bucket}.s3.amazonaws.com/{prefix}"
    hosted = {
        "mode":           "hosted",
        "image_base_url": f"{base}/spectrograms",
        "audio_base_url": f"{base}/audio",
        "results_url":    f"{base}/sessions",
    }
    out = batch_dir / "config_hosted.json"
    out.write_text(json.dumps(hosted, indent=2) + "\n", encoding="utf-8")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload exported batch assets to S3 for hosted scoring."
    )
    parser.add_argument("batch_dir")
    parser.add_argument("--bucket",  required=True,
                        help="S3 bucket name")
    parser.add_argument("--prefix",  default=None,
                        help="S3 key prefix (default: batch directory name)")
    parser.add_argument("--region",  default=None,
                        help="AWS region (default: from ~/.aws/config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be uploaded without actually uploading")
    parser.add_argument("--force",   action="store_true",
                        help="Re-upload even if file already exists in S3")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip audio WAV upload (spectrograms only)")
    args = parser.parse_args()

    batch_dir  = Path(args.batch_dir).resolve()
    h5_path    = batch_dir / "batch.h5"
    export_dir = batch_dir / "export"
    spec_dir   = export_dir / "spectrograms"
    audio_dir  = export_dir / "audio"
    prefix     = args.prefix or batch_dir.name

    # Validate
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found.")
        return
    if not spec_dir.exists():
        print(f"ERROR: {spec_dir} not found.  Run export_batch.py first.")
        return

    s3 = get_s3_client()
    if args.region:
        import boto3
        s3 = boto3.client("s3", region_name=args.region)

    print(f"Batch:  {batch_dir.name}")
    print(f"Bucket: s3://{args.bucket}/{prefix}")
    if args.dry_run:
        print("(dry-run — nothing will actually be uploaded)\n")

    # ── Spectrograms ──────────────────────────────────────────────────────────
    n_pngs = len(list(spec_dir.glob("*.png")))
    print(f"\nUploading spectrograms ({n_pngs} PNGs) ...")
    n_up, n_skip = upload_directory(
        s3, spec_dir, args.bucket,
        prefix=f"{prefix}/spectrograms",
        extension=".png",
        content_type="image/png",
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"  Uploaded: {n_up}   Skipped (already exists): {n_skip}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    if not args.no_audio and audio_dir.exists():
        n_wavs = len(list(audio_dir.glob("*.wav")))
        print(f"\nUploading audio ({n_wavs} WAVs) ...")
        n_up, n_skip = upload_directory(
            s3, audio_dir, args.bucket,
            prefix=f"{prefix}/audio",
            extension=".wav",
            content_type="audio/wav",
            dry_run=args.dry_run,
            force=args.force,
        )
        print(f"  Uploaded: {n_up}   Skipped (already exists): {n_skip}")

    # ── Public manifest ───────────────────────────────────────────────────────
    print("\nUploading public manifest ...")
    upload_manifest(s3, h5_path, args.bucket, prefix, args.dry_run)

    # ── Write hosted config ───────────────────────────────────────────────────
    if not args.dry_run:
        cfg_path = write_hosted_config(batch_dir, args.bucket, prefix)
        print(f"\nHosted config written: {cfg_path}")
        print("To run the scoring app in hosted mode:")
        print(f"  python ranking_app.py {batch_dir} --mode hosted")
    else:
        base = f"https://{args.bucket}.s3.amazonaws.com/{prefix}"
        print(f"\n[dry-run] hosted config would set:")
        print(f"  image_base_url: {base}/spectrograms")
        print(f"  audio_base_url: {base}/audio")

    print("\nDone.")


if __name__ == "__main__":
    main()
