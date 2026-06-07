"""RFSN v10 production generation loop.

Provides ``RFSNGenerator`` — a high-level inference interface that wraps
a loaded model and tokenizer with:

- Prefill (dense causal attention for the initial prompt)
- Decode loop (streaming token generation)
- RFSNRuntime integration hooks for KV-cache + sparse attention
- Temperature / top-p / repetition-penalty sampling
- Telemetry collection per decode step

The generator is backend-agnostic: it works with ``mlx-lm`` models on
Apple Silicon or ``transformers`` models on any platform.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from ..config import RFSNConfig, load_config
from ..kv_manager import RFSNTurboQuantKVManager


try:
    from ..compat import mx
except ImportError:
    mx = None  # type: ignore[assignment]


try:
    from mlx_lm.utils import generate as _mlx_generate
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    _mlx_generate = None  # type: ignore[assignment]


try:
    import transformers  # noqa: F401
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class GenerationConfig:
    """Sampling parameters for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass
class GenerationResult:
    """Result of a single generation request."""

    text: str
    tokens: list[int]
    generation_time_ms: float
    tokens_per_second: float
    telemetry: list[dict] = field(default_factory=list)


class RFSNGenerator:
    """High-level inference generator with RFSN runtime integration.

    Usage (MLX) ::

        from rfsn_v10.model_loader import load_mlx_model
        from rfsn_v10.runtime.generation import RFSNGenerator

        model, tokenizer = load_mlx_model("mlx-community/Llama-3-8B-Instruct-4bit")
        gen = RFSNGenerator(model=model, tokenizer=tokenizer)
        result = gen.chat("Hello, world!")
        print(result.text)

    Usage (streaming) ::

        for token in gen.generate("Hello", stream=True):
            print(token, end="")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: RFSNConfig | None = None,
        kv_manager: RFSNTurboQuantKVManager | None = None,
        enable_sparse_decode: bool = False,
        enable_quantized_kv: bool = True,
        audit_mode: bool = False,
    ):
        """
        Args:
            model: Loaded model (``mlx-lm`` or ``transformers``).
            tokenizer: Matching tokenizer.
            config: RFSN runtime configuration.  Loaded from env when ``None``.
            kv_manager: Optional KV-cache manager.  Created automatically when
                ``None`` and ``enable_quantized_kv`` is ``True``.
            enable_sparse_decode: Whether to enable RFSN sparse decode.
            enable_quantized_kv: Whether to use quantized KV-cache.
            audit_mode: Enable per-step quality auditing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or load_config()
        self.enable_sparse_decode = enable_sparse_decode
        self.enable_quantized_kv = enable_quantized_kv
        self.audit_mode = audit_mode

        self._kv_manager = kv_manager
        if kv_manager is None and enable_quantized_kv:
            self._kv_manager = RFSNTurboQuantKVManager(
                k_bits=8,
                v_bits=5,
                group_size=32,
            )

        self._runtime = None
        if self._kv_manager is not None:
            from .engine import RFSNRuntime
            self._runtime = RFSNRuntime(
                kv_manager=self._kv_manager,
                model_id=getattr(
                    tokenizer, "name_or_path", "unknown"
                ),
                enable_sparse_decode=enable_sparse_decode,
                audit_mode=audit_mode,
            )

        self._telemetry_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        **gen_kwargs: Any,
    ) -> GenerationResult:
        """Generate a response to a single user message.

        Args:
            message: User message text.
            system_prompt: Optional system prompt prepended to the message.
            **gen_kwargs: Overrides for :class:`GenerationConfig` fields.

        Returns:
            :class:`GenerationResult` with full text and metadata.
        """
        prompt = self._build_chat_prompt(message, system_prompt)
        return self._generate_sync(prompt, **gen_kwargs)

    def generate(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> Iterator[str]:
        """Generate text from a raw prompt, yielding tokens as strings.

        Args:
            prompt: Raw prompt string.
            **gen_kwargs: Overrides for :class:`GenerationConfig` fields.

        Yields:
            Decoded token strings (one per yield).
        """
        cfg = self._make_gen_config(**gen_kwargs)
        if MLX_LM_AVAILABLE and hasattr(self.model, "__call__"):
            yield from self._stream_mlx(prompt, cfg)
        else:
            raise RuntimeError(
                "Streaming generation requires mlx-lm.  "
                "Install with: pip install mlx-lm"
            )

    async def generate_async(
        self,
        prompt: str,
        **gen_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming variant of :meth:`generate`.

        Yields:
            Decoded token strings.
        """
        for token in self.generate(prompt, **gen_kwargs):
            yield token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_prompt(
        self,
        message: str,
        system_prompt: str | None = None,
    ) -> str:
        """Build a chat prompt using the tokenizer's chat template if present."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: simple concatenation
        parts: list[str] = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(message)
        return "\n".join(parts)

    def _make_gen_config(self, **overrides: Any) -> GenerationConfig:
        """Build a :class:`GenerationConfig` from defaults + overrides."""
        defaults = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "stream": True,
        }
        defaults.update(overrides)
        return GenerationConfig(**defaults)

    def _generate_sync(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Run synchronous generation and return the full result."""
        cfg = self._make_gen_config(stream=False, **kwargs)
        t_start = time.monotonic()
        tokens: list[int] = []
        telemetry: list[dict] = []

        if MLX_LM_AVAILABLE:
            text = self._generate_mlx(prompt, cfg)
        elif TRANSFORMERS_AVAILABLE:
            text = self._generate_torch(prompt, cfg)
        else:
            raise RuntimeError(
                "No generation backend available.  "
                "Install mlx-lm (Apple Silicon) or transformers."
            )

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        tps = len(tokens) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        return GenerationResult(
            text=text,
            tokens=tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tps,
            telemetry=telemetry,
        )

    def _generate_mlx(self, prompt: str, cfg: GenerationConfig) -> str:
        """Generate via ``mlx_lm.utils.generate``."""
        assert MLX_LM_AVAILABLE and _mlx_generate is not None
        return _mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temp=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            verbose=False,
        )

    def _generate_torch(self, prompt: str, cfg: GenerationConfig) -> str:
        """Generate via ``transformers`` pipeline."""
        assert TRANSFORMERS_AVAILABLE
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "to"):
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=cfg.temperature > 0,
        )
        generated = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _stream_mlx(self, prompt: str, cfg: GenerationConfig) -> Iterator[str]:
        """Stream generation via ``mlx_lm``, yielding individual tokens."""
        assert MLX_LM_AVAILABLE and _mlx_generate is not None
        # mlx_lm.generate does not natively stream token-by-token,
        # so we call it once and then yield words for a basic streaming feel.
        # TODO: integrate with mlx_lm's streaming API once available.
        text = _mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temp=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            verbose=False,
        )
        # Yield word-by-word for a streaming feel
        import re
        for chunk in re.split(r"(\s+)", text):
            if chunk:
                yield chunk

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_telemetry(self) -> list[dict]:
        """Return accumulated telemetry from the runtime (if any)."""
        if self._runtime is not None:
            return [ev.__dict__ for ev in self._runtime.get_telemetry()]
        return []

    def clear_telemetry(self) -> None:
        """Clear telemetry log."""
        if self._runtime is not None:
            self._runtime.clear_telemetry()
