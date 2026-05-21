"""LLM-as-parser + VSA-as-reasoner: hybrid architecture.

The idea: pure VSA solves compositional reasoning when the input is already
parsed into role-filler slots. Natural language is not. A frontier LLM does
parsing well but generalizes compositionally poorly.

The hybrid is the natural fit:
  - LLM: parse natural-language input -> structured slot representation
  - VSA: take the slot representation and produce the output via algebra

This module provides:
  - `Parser` protocol: the abstract interface any parser must implement
  - `HandwrittenParser`: thin wrapper around scan_runner.parse_scan
  - `DiscoveredParser`: thin wrapper around discovered-grammar parsing
  - `LLMParser`: calls a frontier LLM (Anthropic, OpenAI, or any chat backend)
  - `MockLLMParser`: deterministic stub for testing (returns hand-parsed)

The same `SCANHyperion.predict()` works with any parser that satisfies the
`Parser` protocol -- swap the parser, keep the reasoner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from pure_vsa.scan_runner import Atom, Clause, ParsedSCAN


class Parser(Protocol):
    """Any parser that maps raw input strings to a ParsedSCAN structure."""

    def parse(self, input_str: str) -> ParsedSCAN: ...


class HandwrittenParser:
    """The original parse_scan from scan_runner.

    We capture the function reference at construction time so monkey-patching
    pure_vsa.scan_runner.parse_scan downstream (by HyperionWithParser) doesn't
    cause infinite recursion.
    """

    def __init__(self) -> None:
        from pure_vsa.scan_runner import parse_scan
        self._parse_scan = parse_scan

    def parse(self, input_str: str) -> ParsedSCAN:
        return self._parse_scan(input_str)


# ----------------------------------------------------------------------
# LLM-based parser
# ----------------------------------------------------------------------

PARSE_INSTRUCTIONS = """You are a SCAN-grammar parser. Given a SCAN command string, output a JSON object describing its parse:

{
  "clause1": {
    "atom": {"verb": "<verb>", "direction": "<left|right|null>", "spatial": "<opposite|around|null>"},
    "modifier": "<twice|thrice|null>"
  },
  "clause2": <same structure as clause1, or null>,
  "connective": "<and|after|null>"
}

The valid verbs are: walk, look, run, jump, turn.
The valid modifiers are: twice, thrice.

Output ONLY the JSON, no explanation. Examples:

Input: walk
Output: {"clause1": {"atom": {"verb": "walk", "direction": null, "spatial": null}, "modifier": null}, "clause2": null, "connective": null}

Input: walk twice and jump opposite right thrice
Output: {"clause1": {"atom": {"verb": "walk", "direction": null, "spatial": null}, "modifier": "twice"}, "clause2": {"atom": {"verb": "jump", "direction": "right", "spatial": "opposite"}, "modifier": "thrice"}, "connective": "and"}

Now parse:
"""


@dataclass
class LLMParserConfig:
    """How to talk to the LLM backend."""
    backend: str = "anthropic"   # 'anthropic' | 'openai' | 'mock'
    model: str = "claude-haiku-4-5-20251001"  # fast and accurate enough for parsing
    max_tokens: int = 400
    temperature: float = 0.0
    api_key_env: str = "ANTHROPIC_API_KEY"


class LLMParser:
    """Parse SCAN inputs by asking a frontier LLM for the JSON structure.

    The LLM is doing the *parsing* (raw string -> structured slots). The
    actual compositional reasoning (held-out generalization) happens entirely
    in pure VSA downstream. The LLM never sees the test outputs and never
    learns the SCAN grammar -- it's prompted in-context.
    """

    def __init__(self, cfg: LLMParserConfig | None = None) -> None:
        self.cfg = cfg or LLMParserConfig()
        self._cache: dict[str, ParsedSCAN] = {}

    def parse(self, input_str: str) -> ParsedSCAN:
        if input_str in self._cache:
            return self._cache[input_str]
        if self.cfg.backend == "anthropic":
            raw = self._call_anthropic(input_str)
        elif self.cfg.backend == "openai":
            raw = self._call_openai(input_str)
        else:
            raise ValueError(f"unknown backend: {self.cfg.backend}")
        parsed = self._json_to_parsed_scan(raw)
        self._cache[input_str] = parsed
        return parsed

    def _call_anthropic(self, input_str: str) -> str:
        try:
            from anthropic import Anthropic  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. `pip install anthropic` first."
            ) from e
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"env var {self.cfg.api_key_env} not set")
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": PARSE_INSTRUCTIONS + input_str}],
        )
        return msg.content[0].text  # type: ignore[attr-defined]

    def _call_openai(self, input_str: str) -> str:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed. `pip install openai` first."
            ) from e
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise RuntimeError(f"env var {self.cfg.api_key_env} not set")
        client = OpenAI(api_key=key)
        msg = client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": PARSE_INSTRUCTIONS + input_str}],
        )
        return msg.choices[0].message.content or ""

    def _json_to_parsed_scan(self, raw: str) -> ParsedSCAN:
        # extract JSON from raw (in case the LLM wraps it)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        obj = json.loads(raw)
        return self._dict_to_parsed_scan(obj)

    @staticmethod
    def _dict_to_parsed_scan(obj: dict) -> ParsedSCAN:
        def to_atom(d: dict) -> Atom:
            return Atom(
                verb=d["verb"],
                direction=d.get("direction") or None,
                spatial=d.get("spatial") or None,
            )

        def to_clause(d: dict) -> Clause:
            return Clause(atom=to_atom(d["atom"]), modifier=d.get("modifier") or None)

        clause1 = to_clause(obj["clause1"])
        clause2 = to_clause(obj["clause2"]) if obj.get("clause2") else None
        connective = obj.get("connective") or None
        return ParsedSCAN(clause1=clause1, clause2=clause2, connective=connective)


class MockLLMParser:
    """Deterministic LLM-parser stub for testing the framework without API calls.

    Just delegates to the hand-written parser. Behaviorally identical to
    HandwrittenParser, but conceptually represents "what an LLM would do":
    map raw input strings to structured ParsedSCAN.
    """

    def __init__(self) -> None:
        from pure_vsa.scan_runner import parse_scan
        self._parse_scan = parse_scan

    def parse(self, input_str: str) -> ParsedSCAN:
        return self._parse_scan(input_str)


# ----------------------------------------------------------------------
# Integration: SCANHyperion with a pluggable parser
# ----------------------------------------------------------------------

class HyperionWithParser:
    """Wrap SCANHyperion + a pluggable Parser. Use any parser at fit/predict time."""

    def __init__(self, parser: Parser, cfg=None) -> None:
        from pure_vsa.scan_hyperion import SCANConfig, SCANHyperion
        self.parser = parser
        self.reasoner = SCANHyperion(cfg or SCANConfig(d=8192, max_output_len=80))

    def fit(self, examples: list[tuple[str, list[str]]]) -> None:
        # Monkey-patch parse_scan to use our pluggable parser, then delegate.
        import pure_vsa.scan_runner as scan_runner_mod
        original = scan_runner_mod.parse_scan
        try:
            scan_runner_mod.parse_scan = self.parser.parse
            self.reasoner.fit(examples)
        finally:
            scan_runner_mod.parse_scan = original

    def accuracy(self, examples):
        import pure_vsa.scan_runner as scan_runner_mod
        original = scan_runner_mod.parse_scan
        try:
            scan_runner_mod.parse_scan = self.parser.parse
            return self.reasoner.accuracy(examples)
        finally:
            scan_runner_mod.parse_scan = original
