# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine entities extraction package root."""

from graphrag.index.operations.extract_relationships.extract_relationships import (
    ExtractEntityStrategyType,
    extract_relationships,
)

__all__ = ["ExtractEntityStrategyType", "extract_relationships"]
