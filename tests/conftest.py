"""Force deterministic offline pipeline before importing ``src`` (pytest loads this first)."""

from __future__ import annotations

import os

# Override repo .env so tests never hit remote LLM APIs.
os.environ["LLM_PROVIDER"] = "fake"
