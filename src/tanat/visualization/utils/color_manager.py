#!/usr/bin/env python3
"""
ColorManager: assigns colors to categorical labels.

Accepts a colormap spec (string, dict, or list of hex strings) and a list
of labels, then returns a dict mapping each label to a hex color string.
"""

from __future__ import annotations

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors

_DEFAULT_CMAP = "tab10"


class ColorManager:
    """Assigns matplotlib colors to categorical labels.

    Colormap spec formats:
    - ``str``           -> named matplotlib colormap (e.g. "tab10", "Set2").
    - ``dict``          -> explicit label-to-color mapping.
    - ``list[str]``     -> explicit color cycle; wraps around if more labels than colors.
    - ``None``          -> defaults to "tab10".
    """

    @staticmethod
    def build(labels: list, spec: str | dict | list | None = None) -> dict:
        """Return a ``{label: hex_color}`` dict for every label in *labels*.

        Args:
            labels: Unique category labels (order determines color assignment
                    when spec is a cmap name or list).
            spec: Color specification (see class docstring).

        Returns:
            Dict mapping each label to a hex color string.

        Raises:
            ValueError: If spec is a dict and some labels are missing.
        """
        if spec is None:
            spec = _DEFAULT_CMAP

        if isinstance(spec, dict):
            missing = [lbl for lbl in labels if lbl not in spec]
            if missing:
                raise ValueError(
                    f"ColorManager: missing colors for labels: {missing}. "
                    "Pass a complete dict or use a colormap name."
                )
            return {lbl: mpl_colors.to_hex(spec[lbl]) for lbl in labels}

        if isinstance(spec, list):
            return {
                lbl: mpl_colors.to_hex(spec[i % len(spec)])
                for i, lbl in enumerate(labels)
            }

        # Named colormap
        cmap = mpl_cm.get_cmap(spec)
        n = max(len(labels), 1)
        return {lbl: mpl_colors.to_hex(cmap(i / n)) for i, lbl in enumerate(labels)}
