#!/usr/bin/env python3
"""Platform abstraction layer for RFSN v10.

Provides cross-platform support for Metal, CUDA, and CPU backends.
"""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Any

import numpy as np


class PlatformType(Enum):
    """Supported platform types."""

    METAL = "metal"
    CUDA = "cuda"
    CPU = "cpu"


class PlatformCapabilities:
    """Platform capabilities."""

    def __init__(
        self,
        has_shared_memory: bool = False,
        max_threads_per_block: int = 1,
        max_shared_memory_bytes: int = 0,
        supports_warp_operations: bool = False,
    ):
        """Initialize platform capabilities.

        Args:
            has_shared_memory: Whether platform supports shared memory
            max_threads_per_block: Maximum threads per block
            max_shared_memory_bytes: Maximum shared memory in bytes
            supports_warp_operations: Whether platform supports warp operations
        """
        self.has_shared_memory = has_shared_memory
        self.max_threads_per_block = max_threads_per_block
        self.max_shared_memory_bytes = max_shared_memory_bytes
        self.supports_warp_operations = supports_warp_operations


class Backend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def get_platform_type(self) -> PlatformType:
        """Get the platform type.

        Returns:
            PlatformType enum
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> PlatformCapabilities:
        """Get platform capabilities.

        Returns:
            PlatformCapabilities instance
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available.

        Returns:
            True if available
        """
        pass

    @abstractmethod
    def dequantize_wht_sign(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
    ) -> np.ndarray:
        """Execute dequantization, WHT, and sign application.

        Args:
            packed: Packed quantized data
            scales: Quantization scales
            n_values: Number of values
            bits: Quantization bit width
            seed: Random seed for signs
            use_wht: Apply WHT transform
            use_incoherent_signs: Apply incoherent signs

        Returns:
            Dequantized array
        """
        pass


class MetalBackend(Backend):
    """Metal backend for Apple Silicon."""

    def __init__(self):
        """Initialize Metal backend."""
        try:
            import mlx

            self.mlx = mlx
            self._available = True
            self._capabilities = PlatformCapabilities(
                has_shared_memory=True,
                max_threads_per_block=1024,
                max_shared_memory_bytes=32768,
                supports_warp_operations=True,
            )
        except ImportError:
            self._available = False
            self._capabilities = PlatformCapabilities()

    def get_platform_type(self) -> PlatformType:
        """Get platform type."""
        return PlatformType.METAL

    def get_capabilities(self) -> PlatformCapabilities:
        """Get platform capabilities."""
        return self._capabilities

    def is_available(self) -> bool:
        """Check if Metal is available."""
        return self._available

    def dequantize_wht_sign(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
    ) -> np.ndarray:
        """Execute dequantization using Metal kernels."""
        if not self._available:
            raise RuntimeError("Metal backend not available")

        # Import here to avoid import errors
        from rfsn_v10.kv_manager import _reconstruct_packed_dequant_wht_sign_fused

        # Convert to MLX arrays
        packed_mlx = self.mlx.array(packed)
        scales_mlx = self.mlx.array(scales)

        # Call fused kernel
        result = _reconstruct_packed_dequant_wht_sign_fused(
            packed_mlx,
            scales_mlx,
            n_values,
            packed.shape,
            bits,
            seed,
            use_wht,
            use_incoherent_signs,
        )

        return np.array(result)


class CUDABackend(Backend):
    """CUDA backend for NVIDIA GPUs."""

    def __init__(self):
        """Initialize CUDA backend."""
        try:
            import torch

            self.torch = torch
            self._available = torch.cuda.is_available()
            if self._available:
                self._capabilities = PlatformCapabilities(
                    has_shared_memory=True,
                    max_threads_per_block=1024,
                    max_shared_memory_bytes=49152,
                    supports_warp_operations=True,
                )
            else:
                self._capabilities = PlatformCapabilities()
        except ImportError:
            self._available = False
            self._capabilities = PlatformCapabilities()

    def get_platform_type(self) -> PlatformType:
        """Get platform type."""
        return PlatformType.CUDA

    def get_capabilities(self) -> PlatformCapabilities:
        """Get platform capabilities."""
        return self._capabilities

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._available

    def dequantize_wht_sign(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
    ) -> np.ndarray:
        """Execute dequantization using CUDA kernels."""
        if not self._available:
            raise RuntimeError("CUDA backend not available")

        # Placeholder for CUDA implementation
        # This would require implementing CUDA kernels
        # For now, fall back to CPU path
        return self._cpu_dequantize(
            packed, scales, n_values, bits, seed, use_wht, use_incoherent_signs
        )

    def _cpu_dequantize(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool,
        use_incoherent_signs: bool,
    ) -> np.ndarray:
        """CPU fallback dequantization."""
        # Import CPU implementation
        from rfsn_v10.kv_manager import _reconstruct_packed_dequant_wht_sign

        return _reconstruct_packed_dequant_wht_sign(
            packed, scales, n_values, packed.shape, bits, seed, use_wht, use_incoherent_signs
        )


class CPUBackend(Backend):
    """CPU backend for fallback."""

    def __init__(self):
        """Initialize CPU backend."""
        self._available = True
        self._capabilities = PlatformCapabilities(
            has_shared_memory=False,
            max_threads_per_block=1,
            max_shared_memory_bytes=0,
            supports_warp_operations=False,
        )

    def get_platform_type(self) -> PlatformType:
        """Get platform type."""
        return PlatformType.CPU

    def get_capabilities(self) -> PlatformCapabilities:
        """Get platform capabilities."""
        return self._capabilities

    def is_available(self) -> bool:
        """Check if CPU is available."""
        return True

    def dequantize_wht_sign(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
    ) -> np.ndarray:
        """Execute dequantization on CPU."""
        from rfsn_v10.kv_manager import _reconstruct_packed_dequant_wht_sign

        return _reconstruct_packed_dequant_wht_sign(
            packed, scales, n_values, packed.shape, bits, seed, use_wht, use_incoherent_signs
        )


class PlatformManager:
    """Manages platform backends."""

    def __init__(self):
        """Initialize platform manager."""
        self.backends: list[Backend] = []
        self._detect_backends()

    def _detect_backends(self) -> None:
        """Detect available backends."""
        # Try Metal first (Apple Silicon)
        metal = MetalBackend()
        if metal.is_available():
            self.backends.append(metal)

        # Try CUDA (NVIDIA)
        cuda = CUDABackend()
        if cuda.is_available():
            self.backends.append(cuda)

        # Always add CPU as fallback
        self.backends.append(CPUBackend())

    def get_best_backend(self) -> Backend:
        """Get the best available backend.

        Returns:
            Best backend instance
        """
        if not self.backends:
            return CPUBackend()

        # Return first available backend (Metal > CUDA > CPU)
        return self.backends[0]

    def get_backend(self, platform_type: PlatformType) -> Optional[Backend]:
        """Get backend for specific platform type.

        Args:
            platform_type: Desired platform type

        Returns:
            Backend instance or None if not available
        """
        for backend in self.backends:
            if backend.get_platform_type() == platform_type:
                return backend
        return None

    def get_available_platforms(self) -> list[PlatformType]:
        """Get list of available platform types.

        Returns:
            List of PlatformType enums
        """
        return [backend.get_platform_type() for backend in self.backends]

    def execute_dequantize_wht_sign(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        n_values: int,
        bits: int,
        seed: int,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
        preferred_platform: Optional[PlatformType] = None,
    ) -> Tuple[np.ndarray, PlatformType]:
        """Execute dequantization on best available backend.

        Args:
            packed: Packed quantized data
            scales: Quantization scales
            n_values: Number of values
            bits: Quantization bit width
            seed: Random seed for signs
            use_wht: Apply WHT transform
            use_incoherent_signs: Apply incoherent signs
            preferred_platform: Preferred platform type

        Returns:
            Tuple of (result array, platform used)
        """
        backend = None

        if preferred_platform:
            backend = self.get_backend(preferred_platform)

        if backend is None:
            backend = self.get_best_backend()

        result = backend.dequantize_wht_sign(
            packed, scales, n_values, bits, seed, use_wht, use_incoherent_signs
        )

        return result, backend.get_platform_type()


def get_platform_manager() -> PlatformManager:
    """Get the global platform manager instance.

    Returns:
        PlatformManager instance
    """
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = PlatformManager()
    return _platform_manager


_platform_manager: Optional[PlatformManager] = None


def detect_platform() -> PlatformType:
    """Detect the current platform.

    Returns:
        Detected platform type
    """
    system = platform.system()

    if system == "Darwin":
        # Check for Apple Silicon
        try:
            import mlx

            return PlatformType.METAL
        except ImportError:
            pass

    # Check for CUDA
    try:
        import torch

        if torch.cuda.is_available():
            return PlatformType.CUDA
    except ImportError:
        pass

    return PlatformType.CPU
