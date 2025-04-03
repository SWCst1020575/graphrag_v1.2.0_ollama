# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine entities extraction package root."""

from graphrag.index.operations.extract_entities_only.extract_entities import (
    ExtractEntityStrategyType,
    extract_entities_only,
)

__all__ = ["ExtractEntityStrategyType", "extract_entities_only"]
