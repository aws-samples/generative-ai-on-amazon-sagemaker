"""Exact token counts with the base model's own tokenizer (tokenizer.json from the Hub artifact)."""
from __future__ import annotations

import json
from functools import lru_cache

import boto3

from .config import BASE_MODEL_ID, WORK_DIR


@lru_cache(maxsize=1)
def tokenizer(region: str):
    from tokenizers import Tokenizer
    path = WORK_DIR / "tokenizer" / BASE_MODEL_ID / "tokenizer.json"
    if not path.exists():
        sm = boto3.client("sagemaker", region_name=region)
        doc = json.loads(sm.describe_hub_content(HubName="SageMakerPublicHub", HubContentType="Model",
                                                 HubContentName=BASE_MODEL_ID)["HubContentDocument"])
        uri = doc["HostingArtifactUri"].rstrip("/") + "/tokenizer.json"
        bucket, key = uri[5:].split("/", 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3", region_name=region).download_file(bucket, key, str(path))
    return Tokenizer.from_file(str(path))


def count(texts: list[str], region: str) -> list[int]:
    return [len(e.ids) for e in tokenizer(region).encode_batch(list(texts))]
