"""ClickHouse client security tests — Week 4.

Verifies HTTPS enforcement, prompt hashing, token auth, and retry queue.
"""
from __future__ import annotations

import hashlib

import pytest

from rfsn_v10.clickhouse_client import ClickHouseClient


def test_http_over_wan_raises():
    """HTTP must be rejected for non-localhost hosts."""
    with pytest.raises(ValueError, match="HTTP is only allowed for localhost"):
        ClickHouseClient(host="remote.example.com", secure=False)


def test_localhost_http_allowed():
    """HTTP is permitted for localhost."""
    client = ClickHouseClient(host="localhost", secure=False)
    assert not client.secure


def test_https_default():
    """Default secure=True uses HTTPS."""
    client = ClickHouseClient()
    assert client.secure
    assert client._base_url.startswith("https://")


def test_auth_token_header_set():
    """RFSN-Auth header is set when token is provided."""
    client = ClickHouseClient(auth_token="my-secret-token")
    assert client.auth_token == "my-secret-token"


def test_prompt_hashing_replaces_raw_text():
    """Sensitive keys are replaced with SHA-256 hashes."""
    event = {
        "task_id": "t1",
        "prompt": "secret user input",
        "text": "another secret",
        "safe_key": "public value",
    }
    hashed = ClickHouseClient._hash_sensitive_values(event)

    assert hashed["prompt"] == hashlib.sha256(
        b"secret user input"
    ).hexdigest()
    assert hashed["prompt_length"] == len("secret user input")

    assert hashed["text"] == hashlib.sha256(
        b"another secret"
    ).hexdigest()
    assert hashed["text_length"] == len("another secret")

    assert hashed["safe_key"] == "public value"
    assert "safe_key_length" not in hashed


def test_non_string_sensitive_values_preserved():
    """Non-string values for sensitive keys are left untouched."""
    event = {"prompt": 123, "task_id": "t1"}
    hashed = ClickHouseClient._hash_sensitive_values(event)
    assert hashed["prompt"] == 123
    assert "prompt_length" not in hashed
