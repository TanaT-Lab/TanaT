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

# Strategy for null values in time index columns (__TIME__, __START__, __END__).
# - "drop": remove affected rows and emit a UserWarning.
# - "raise": raise ValueError immediately if any null is found.
NaTimeIndex = Literal["drop", "raise"]

# Strategy for null values in the entity feature label.
# - "drop": remove rows with a null label and emit a UserWarning.
# - "raise": raise ValueError immediately if any null label is found.
# - "category": replace null labels with "N/A", making missing values
#   an explicit visible category.
NaLabel = Literal["drop", "raise", "category"]
