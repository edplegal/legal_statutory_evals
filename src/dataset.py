"""Dataset loading utilities for landlord-tenant evaluation prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetItem:
    """Represents one evaluation item."""

    id: str
    scenario: str
    jurisdiction: str
    variant: str
    turns: List[Dict[str, str]]
    reference_law: str
    expected_legal_points: List[str]
    manual_labels: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for serialization."""
        return {
            "id": self.id,
            "scenario": self.scenario,
            "jurisdiction": self.jurisdiction,
            "variant": self.variant,
            "turns": self.turns,
            "reference_law": self.reference_law,
            "expected_legal_points": self.expected_legal_points,
            "manual_labels": self.manual_labels,
        }


def load_dataset(path: Path) -> List[DatasetItem]:
    """Load dataset items from a JSONL file."""
    items: List[DatasetItem] = []
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(DatasetItem(**data))
    if not items:
        raise ValueError(f"No items loaded from {path}")
    return items
