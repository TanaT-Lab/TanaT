#!/usr/bin/env python3
"""
Shared Literal type aliases for sequence visualization builders.
"""

from __future__ import annotations

from typing import Literal

# Shared across spanplot, timeline, ... # add to list as needed
GroupBy = Literal["category", "id"]

# Shared across barplot, spanplot, ... # add to list as needed
DisplayUnit = Literal["days", "hours", "minutes", "seconds"]
SortOrder = Literal["alphabetic", "ascending", "descending"]
Orientation = Literal["vertical", "horizontal"]

# Shared across timeline, distribution, ... # add to list as needed
TimeMode = Literal["absolute", "relative"]
