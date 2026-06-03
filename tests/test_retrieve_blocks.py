#!/usr/bin/env python3
"""Test sparse block retrieval from KV manager."""
from __future__ import annotations

import tempfile

import mlx.core as mx
import pytest

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


class TestRetrieveBlocks:
    def test_retrieve_blocks_basic(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RFSNTurboQuantKVManager(
                cache_dir=td,
                k_bits=4,
                v_bits=4,
                group_size=64,
                use_wht=True,
                use_incoherent_signs=True,
                prefer_metal_kernels=True,
                prefer_fused_kernel=True,
            )
            shape = (1, 4, 256, 64)
            x = mx.random.normal(shape).astype(mx.float16)
            mgr.store("key", x, x, 256)

            # Request blocks 0 and 2 (skipping block 1)
            k, v = mgr.retrieve_blocks("key", [0, 2], block_size=64)
            assert k is not None
            assert v is not None
            # 2 blocks * 64 tokens = 128 tokens
            assert k.shape == (1, 4, 128, 64)
            assert v.shape == (1, 4, 128, 64)

    def test_retrieve_blocks_equivalence_vs_full(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RFSNTurboQuantKVManager(
                cache_dir=td,
                k_bits=4,
                v_bits=4,
                group_size=64,
                use_wht=True,
                use_incoherent_signs=True,
                prefer_metal_kernels=True,
                prefer_fused_kernel=True,
            )
            shape = (1, 4, 256, 64)
            x = mx.random.normal(shape).astype(mx.float16)
            mgr.store("key", x, x, 256)

            k_full, v_full = mgr.retrieve("key")
            k_sparse, v_sparse = mgr.retrieve_blocks(
                "key", [0, 1], block_size=64
            )

            # First 128 tokens should match exactly
            assert mx.allclose(k_sparse, k_full[:, :, :128, :], atol=1e-4)
            assert mx.allclose(v_sparse, v_full[:, :, :128, :], atol=1e-4)

    def test_retrieve_blocks_empty_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RFSNTurboQuantKVManager(
                cache_dir=td,
                k_bits=4,
                v_bits=4,
            )
            x = mx.random.normal((1, 4, 64, 64)).astype(mx.float16)
            mgr.store("key", x, x, 64)
            with pytest.raises(ValueError, match="block_indices must not be empty"):
                mgr.retrieve_blocks("key", [], block_size=64)

    def test_retrieve_blocks_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RFSNTurboQuantKVManager(
                cache_dir=td,
                k_bits=4,
                v_bits=4,
            )
            x = mx.random.normal((1, 4, 64, 64)).astype(mx.float16)
            mgr.store("key", x, x, 64)
            with pytest.raises(ValueError, match="block index"):
                mgr.retrieve_blocks("key", [1], block_size=64)

    def test_retrieve_blocks_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RFSNTurboQuantKVManager(
                cache_dir=td,
                k_bits=4,
                v_bits=4,
            )
            result = mgr.retrieve_blocks("missing", [0], block_size=64)
            assert result is None
