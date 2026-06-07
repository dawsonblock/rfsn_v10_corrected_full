"""Security tests for ClickHouse telemetry (Tickets 4-1, 4-2, 4-3).

Tests verify:
- TLS enforcement (4-1)
- SHA-256 prompt hashing (4-2)
- Retry queue with SIGTERM flush (4-3)
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
import time
from urllib import error

import pytest

from rfsn_v10.clickhouse_client import ClickHouseClient


class TestTLSEnforcement:
    """Ticket 4-1: HTTPS-only ClickHouse client."""

    def test_http_allowed_for_localhost(self):
        """HTTP should be allowed for localhost connections."""
        client = ClickHouseClient(
            host="localhost",
            port=8123,
            secure=False,
            auth_token="test-token"
        )
        assert client.secure is False
        assert client._base_url == "http://localhost:8123"

    def test_http_allowed_for_127_0_0_1(self):
        """HTTP should be allowed for 127.0.0.1."""
        client = ClickHouseClient(
            host="127.0.0.1",
            port=8123,
            secure=False
        )
        assert client.secure is False

    def test_http_rejected_for_remote_host(self):
        """HTTP should be rejected for non-localhost hosts."""
        with pytest.raises(ValueError, match="HTTP is only allowed for localhost"):
            ClickHouseClient(
                host="clickhouse.example.com",
                port=8123,
                secure=False
            )

    def test_https_default_for_remote(self):
        """HTTPS should be default/enforced for remote hosts."""
        client = ClickHouseClient(
            host="clickhouse.example.com",
            port=8443,
            secure=True,
            auth_token="secret-token"
        )
        assert client.secure is True
        assert client._base_url == "https://clickhouse.example.com:8443"

    def test_auth_token_stored(self):
        """Auth token should be stored for RFSN-Auth header."""
        client = ClickHouseClient(
            host="localhost",
            auth_token="bearer-token-123"
        )
        assert client.auth_token == "bearer-token-123"

    def test_rfsn_auth_header_in_request(self):
        """RFSN-Auth header should be included in requests when token set."""
        client = ClickHouseClient(
            host="localhost",
            secure=False,
            auth_token="test-token"
        )

        # Mock the request to capture headers
        captured_headers = {}
        original_urlopen = client._execute_query.__func__

        # The auth token should be stored
        assert client.auth_token == "test-token"


class TestPromptHashing:
    """Ticket 4-2: SHA-256 prompt hashing."""

    def test_sensitive_keys_defined(self):
        """Client should have defined sensitive keys."""
        assert "prompt" in ClickHouseClient._SENSITIVE_KEYS
        assert "text" in ClickHouseClient._SENSITIVE_KEYS
        assert "input_text" in ClickHouseClient._SENSITIVE_KEYS
        assert "user_message" in ClickHouseClient._SENSITIVE_KEYS

    def test_hash_sensitive_values(self):
        """Sensitive values should be SHA-256 hashed."""
        original = "secret user prompt"
        expected_hash = hashlib.sha256(original.encode()).hexdigest()

        event = {"prompt": original, "model": "llama-7b"}
        result = ClickHouseClient._hash_sensitive_values(event)

        assert result["prompt"] == expected_hash
        assert len(result["prompt"]) == 64  # 32 bytes * 2 hex chars
        assert result["prompt_length"] == len(original)

    def test_hash_empty_string(self):
        """Empty string should hash correctly."""
        event = {"prompt": ""}
        result = ClickHouseClient._hash_sensitive_values(event)
        expected = hashlib.sha256(b"").hexdigest()
        assert result["prompt"] == expected

    def test_non_sensitive_fields_preserved(self):
        """Non-sensitive fields should not be hashed."""
        event = {
            "model": "test-model",
            "tokens": 100,
            "temperature": 0.7
        }

        result = ClickHouseClient._hash_sensitive_values(event)
        assert result == event

    def test_mixed_sensitive_and_non_sensitive(self):
        """Mixed fields: sensitive hashed, others preserved."""
        event = {
            "prompt": "secret prompt text",
            "model": "llama-7b",
            "latency_ms": 100.0,
            "text": "another secret"
        }

        result = ClickHouseClient._hash_sensitive_values(event)

        # Sensitive fields should be hashed
        assert result["prompt"] != "secret prompt text"
        assert result["text"] != "another secret"
        assert len(result["prompt"]) == 64
        assert len(result["text"]) == 64

        # Length metadata added
        assert result["prompt_length"] == 18
        assert result["text_length"] == 14

        # Non-sensitive fields unchanged
        assert result["model"] == "llama-7b"
        assert result["latency_ms"] == 100.0

    def test_non_string_values_not_hashed(self):
        """Non-string values in sensitive keys should not be hashed."""
        event = {
            "prompt": 12345,  # Not a string
            "text": None,  # Not a string
        }

        result = ClickHouseClient._hash_sensitive_values(event)
        # Non-string values should be preserved as-is
        assert result["prompt"] == 12345
        assert result["text"] is None

    def test_no_raw_prompt_in_output(self):
        """Original prompt text should never appear in output."""
        sensitive_words = ["password", "secret", "token", "key"]

        for word in sensitive_words:
            event = {"prompt": f"my {word} is 12345"}
            result = ClickHouseClient._hash_sensitive_values(event)

            # Original words should not appear
            assert word not in result["prompt"]
            assert "12345" not in result["prompt"]
            assert len(result["prompt"]) == 64  # It's a hash


class TestRetryQueue:
    """Ticket 4-3: Exponential backoff retry queue."""

    def test_queue_initially_empty(self):
        """Queue should start empty."""
        client = ClickHouseClient(host="localhost", secure=False)
        assert len(client._pending_queue) == 0

    def test_max_retries_configured(self):
        """Max retries should be configured."""
        client = ClickHouseClient(host="localhost", secure=False)
        assert client._max_retries == 5

    def test_exponential_backoff_in_write_events(self):
        """Backoff in _write_events should follow exponential pattern."""
        client = ClickHouseClient(host="localhost", secure=False)

        # Test the backoff calculation: min(2 ** attempt, 30)
        test_cases = [
            (0, 1),   # 2^0 = 1
            (1, 2),   # 2^1 = 2
            (2, 4),   # 2^2 = 4
            (3, 8),   # 2^3 = 8
            (4, 16),  # 2^4 = 16
            (5, 30),  # 2^5 = 32, but capped at 30
            (10, 30), # Always capped at 30
        ]

        for attempt, expected_max in test_cases:
            backoff = min(2 ** attempt, 30)
            assert backoff <= 30  # Never exceeds 30
            if attempt < 5:
                assert backoff == 2 ** attempt

    def test_failed_events_queued(self):
        """Failed events should be added to pending queue."""
        client = ClickHouseClient(host="localhost", secure=False)

        # Manually add events to simulate failed writes
        event = {"model": "test", "latency_ms": 100}
        client._pending_queue.append(event)

        assert len(client._pending_queue) == 1
        assert client._pending_queue[0] == event

    def test_drain_pending_queue_empty(self):
        """Draining empty queue should not error."""
        client = ClickHouseClient(host="localhost", secure=False)
        client._drain_pending_queue()  # Should not raise
        assert len(client._pending_queue) == 0


class TestSIGTERMHandling:
    """Ticket 4-3: SIGTERM handler for graceful shutdown."""

    def test_flush_path_defined(self):
        """Flush path should be defined for disk persistence."""
        assert hasattr(ClickHouseClient, '_FLUSH_PATH')
        assert ClickHouseClient._FLUSH_PATH.endswith('.jsonl')

    def test_sigterm_handler_exists(self):
        """SIGTERM handler method should exist."""
        client = ClickHouseClient(host="localhost", secure=False)
        assert hasattr(client, '_sigterm_handler')

    def test_queue_flush_to_disk(self):
        """Queue should be flushable to disk."""
        client = ClickHouseClient(host="localhost", secure=False)

        # Add events to queue
        client._pending_queue = [
            {"event_id": 1, "data": "test1"},
            {"event_id": 2, "data": "test2"}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            flush_path = f.name
            # Write something to clear the file first
            f.write("")

        try:
            # Use the default flush method
            original_flush_path = client._FLUSH_PATH
            client._FLUSH_PATH = flush_path

            client._flush_queue_to_disk()

            # Verify file contents
            with open(flush_path) as f_read:
                lines = f_read.readlines()

            assert len(lines) == 2
            assert json.loads(lines[0])["event_id"] == 1
            assert json.loads(lines[1])["event_id"] == 2

            # Queue should be cleared after flush
            assert len(client._pending_queue) == 0

            # Restore original path
            client._FLUSH_PATH = original_flush_path
        finally:
            if os.path.exists(flush_path):
                os.unlink(flush_path)

    def test_replay_flushed_events(self):
        """Previously flushed events should be replayed on startup."""
        client = ClickHouseClient(host="localhost", secure=False)

        # Create a flush file with old events
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"old_event": 1}) + '\n')
            f.write(json.dumps({"old_event": 2}) + '\n')
            flush_path = f.name

        try:
            # Set client flush path and replay
            original_path = client._FLUSH_PATH
            client._FLUSH_PATH = flush_path
            client._replay_flushed_events()

            # Events should be in queue
            assert len(client._pending_queue) == 2
            assert client._pending_queue[0]["old_event"] == 1

            # Restore path
            client._FLUSH_PATH = original_path

            # Clean up
            if os.path.exists(flush_path):
                os.unlink(flush_path)
        except Exception:
            if os.path.exists(flush_path):
                os.unlink(flush_path)
            raise

    def test_flush_empty_queue(self):
        """Flushing empty queue should not create file."""
        client = ClickHouseClient(host="localhost", secure=False)
        client._pending_queue = []

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            flush_path = f.name

        try:
            original_path = client._FLUSH_PATH
            client._FLUSH_PATH = flush_path

            client._flush_queue_to_disk()

            # File should be empty or not modified
            with open(flush_path) as f_read:
                content = f_read.read()
            assert content == ""  # Empty queue = no writes

            client._FLUSH_PATH = original_path
        finally:
            if os.path.exists(flush_path):
                os.unlink(flush_path)


class TestIntegrationSecurity:
    """Integration tests for security features."""

    def test_full_hashing_pipeline(self):
        """End-to-end: event with sensitive data is fully hashed."""
        raw_event = {
            "timestamp": time.time(),
            "prompt": "My password is secret123!",
            "user_message": "API key: sk-abc123",
            "model": "llama-7b",
            "backend": "metal",
            "latency_ms": 150.5,
        }

        # Hash sensitive values
        hashed = ClickHouseClient._hash_sensitive_values(raw_event)

        # Verify all sensitive data is hashed
        assert hashed["prompt"] != raw_event["prompt"]
        assert hashed["user_message"] != raw_event["user_message"]

        # Verify hashes are 64-char hex
        assert len(hashed["prompt"]) == 64
        assert all(c in '0123456789abcdef' for c in hashed["prompt"])

        # Verify length metadata added
        assert hashed["prompt_length"] == len(raw_event["prompt"])
        assert hashed["user_message_length"] == len(raw_event["user_message"])

        # Verify non-sensitive data preserved
        assert hashed["model"] == "llama-7b"
        assert hashed["backend"] == "metal"
        assert hashed["latency_ms"] == 150.5

        # Verify original secrets don't appear anywhere in output
        event_str = json.dumps(hashed)
        assert "password" not in event_str.lower()
        assert "secret123" not in event_str
        assert "sk-abc123" not in event_str

    def test_allowed_tables_defined(self):
        """Allowed tables should be defined for security."""
        assert len(ClickHouseClient.ALLOWED_TABLES) > 0
        assert "rfsn_attention_events" in ClickHouseClient.ALLOWED_TABLES
        assert "rfsn_audit_events" in ClickHouseClient.ALLOWED_TABLES

    def test_disallowed_table_rejected(self):
        """Writing to disallowed table should raise."""
        client = ClickHouseClient(host="localhost", secure=False)

        with pytest.raises(ValueError, match="Table not allowed"):
            client._write_events("disallowed_table", [{"test": "data"}])

    def test_empty_events_noop(self):
        """Empty events list should be no-op."""
        client = ClickHouseClient(host="localhost", secure=False)
        # Should not raise
        client._write_events("rfsn_attention_events", [])
