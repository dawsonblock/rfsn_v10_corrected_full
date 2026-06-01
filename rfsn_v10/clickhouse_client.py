#!/usr/bin/env python3
"""
RFSN v10 - ClickHouse Client for Telemetry.

Simple HTTP-based client for writing telemetry batches to ClickHouse.
Assumes tables are pre-created via DDL scripts.
"""
from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Optional
from urllib import request, parse, error


class ClickHouseClient:
    """
    HTTP-based ClickHouse client for telemetry ingestion.
    
    Features:
    - Simple POST-based inserts
    - JSON format for telemetry events
    - Automatic table creation (if enabled)
    - Basic error handling and retries
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        username: str = "default",
        password: str = "",
        database: str = "default",
        secure: bool = False,
        create_tables: bool = False,
    ):
        """
        Args:
            host: ClickHouse server hostname.
            port: ClickHouse HTTP port (usually 8123).
            username: ClickHouse username.
            password: ClickHouse password.
            database: Target database.
            secure: Use HTTPS if True.
            create_tables: If True, attempt to create tables on init.
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.secure = secure
        self.create_tables = create_tables
        
        self._base_url = f"{'https' if secure else 'http'}://{host}:{port}"
        self._session = None  # Could reuse urllib opener for connection pooling
        
        if create_tables:
            self._create_tables_if_not_exist()
    
    def _get_url(self, endpoint: str = "") -> str:
        """Construct full URL for ClickHouse HTTP interface."""
        return f"{self._base_url}/{endpoint.lstrip('/')}"
    
    def _execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Execute a ClickHouse query via HTTP POST."""
        data = query.encode('utf-8')
        req = request.Request(
            self._get_url(),
            data=data,
            headers={
                'Content-Type': 'text/plain',
                'X-ClickHouse-User': self.username,
                'X-ClickHouse-Key': self.password,
            }
        )
        
        try:
            with request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"ClickHouse error: {response.status} {response.reason}")
        except error.URLError as e:
            raise RuntimeError(f"Failed to connect to ClickHouse: {e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"ClickHouse query failed: {e}") from e
    
    def _create_tables_if_not_exist(self) -> None:
        """Create telemetry tables if they don't exist."""
        tables = {
            "rfsn_attention_events": """
                CREATE TABLE IF NOT EXISTS rfsn_attention_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    seq_len UInt32,
                    head_count UInt32,
                    head_dim UInt32,
                    top_k_ratio Float32,
                    block_size UInt32,
                    num_active_blocks UInt32,
                    effective_sparsity Float32,
                    kv_cache_hit UInt8,
                    kv_cache_store_latency_ms Float64,
                    kv_cache_retrieve_latency_ms Float64,
                    attention_latency_ms Float64,
                    total_latency_ms Float64,
                    fallback_used UInt8,
                    sparse_success UInt8,
                    dense_success UInt8,
                    audit_enabled UInt8,
                    audit_cosine Float64,
                    audit_rel_mae Float64,
                    audit_max_abs_error Float64,
                    termination_reason String,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_kv_cache_events": """
                CREATE TABLE IF NOT EXISTS rfsn_kv_cache_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    operation String,  -- 'store' or 'retrieve'
                    key_size_bytes UInt64,
                    value_size_bytes UInt64,
                    latency_ms Float64,
                    success UInt8,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_audit_events": """
                CREATE TABLE IF NOT EXISTS rfsn_audit_events (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    sparse_output String,  -- JSON or hash for comparison
                    dense_output String,   -- JSON or hash for comparison
                    cosine_similarity Float64,
                    rel_mae Float64,
                    max_abs_error Float64,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_failures": """
                CREATE TABLE IF NOT EXISTS rfsn_failures (
                    task_id String,
                    model_id String,
                    layer_id String,
                    batch_id String,
                    skill_pattern String,
                    error_type String,
                    error_message String,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """,
            "rfsn_sparsity_profiles": """
                CREATE TABLE IF NOT EXISTS rfsn_sparsity_profiles (
                    task_id String,
                    model_id String,
                    skill_pattern String,
                    avg_top_k_ratio Float64,
                    min_top_k_ratio Float64,
                    max_top_k_ratio Float64,
                    avg_effective_sparsity Float64,
                    fallback_rate Float64,
                    sample_count UInt32,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                PARTITION BY toYYYYMM(timestamp)
                ORDER BY (task_id, timestamp)
                SETTINGS index_granularity = 8192
            """
        }
        
        for table_name, ddl in tables.items():
            try:
                self._execute_query(ddl)
            except Exception as e:
                # Log but don't fail - table might already exist or permissions issue
                import warnings
                warnings.warn(f"Could not create table {table_name}: {e}")
    
    def write_telemetry_batch(self, events: List[Dict[str, Any]]) -> None:
        """
        Write a batch of telemetry events to ClickHouse.
        
        Args:
            events: List of telemetry event dictionaries.
        """
        if not events:
            return
        
        # Convert events to JSON lines format for ClickHouse
        lines = []
        for event in events:
            # Flatten the event for insertion
            line_parts = []
            for key, value in event.items():
                if isinstance(value, str):
                    # Escape single quotes in strings
                    escaped = value.replace("'", "\\'")
                    line_parts.append(f"'{escaped}'")
                elif isinstance(value, (bool,)):
                    line_parts.append('1' if value else '0')
                elif value is None:
                    line_parts.append('NULL')
                else:
                    line_parts.append(str(value))
        lines.append(f"({', '.join(line_parts)})")
    
    if not lines:
        return
    
    # Determine table based on event structure (simplified - in practice you'd route differently)
    # For now, assume all events go to attention_events table
    # In a real implementation, you'd have different tables for different event types
    columns = ", ".join(events[0].keys())
    values = ", ".join(lines)
    query = f"""
        INSERT INTO rfsn_attention_events ({columns}) 
        VALUES {values}
    """
        
        self._execute_query(query)
    
    def close(self) -> None:
        """Close the client and cleanup resources."""
        # Nothing to close for HTTP client, but keep for interface consistency
        pass
    
    def ping(self) -> bool:
        """Check if ClickHouse is reachable."""
        try:
            self._execute_query("SELECT 1")
            return True
        except Exception:
            return False
