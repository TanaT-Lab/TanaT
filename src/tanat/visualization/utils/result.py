#!/usr/bin/env python3
"""
VisualizationResult: wraps a matplotlib Figure for display and export.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class VisualizationResult:
    """Lightweight wrapper around a matplotlib :class:`~matplotlib.figure.Figure`.

    Example::

        result = visualizer.draw(pool)
        result.show()
        result.save("output/barplot.png", dpi=150)
    """

    def __init__(self, fig: Figure) -> None:
        self._fig = fig

    @property
    def figure(self) -> Figure:
        """The underlying matplotlib Figure."""
        return self._fig

    def show(self) -> None:
        """Display the figure (calls :func:`matplotlib.pyplot.show`)."""
        plt.show()

    def save(self, path: str | Path, *, dpi: int = 150, **kwargs) -> None:
        """Save the figure to *path*.

        Args:
            path: Destination file path. Format inferred from extension.
            dpi: Resolution in dots per inch.
            **kwargs: Forwarded to :meth:`~matplotlib.figure.Figure.savefig`.
        """
        self._fig.savefig(path, dpi=dpi, **kwargs)

    def __repr__(self) -> str:
        return f"VisualizationResult(figure={self._fig!r})"
