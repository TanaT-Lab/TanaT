Visualisation
=============

Visualize sequences using :class:`~tanat.visualization.SequenceVisualizer`.

All visualizations work on both :term:`sequence pools <pool>` and individual
:term:`sequences <sequence>`.

All visualizers follow the same **fluent builder** pattern::

    SequenceVisualizer.<type>(<params>) \
        .title("...") \
        .colors("Set2") \
        .draw(pool, entity_feature="status") \
        .show()

Style methods
-------------

All builders share the same chainable style methods:

- ``.title(text)``                    → figure title
- ``.figsize(w, h)``                  → figure dimensions in inches
- ``.grid(show=True)``                → background grid lines
- ``.colors(spec)``                   → color palette (colormap name, dict, or list)
- ``.legend(show, location, title)``  → legend configuration
- ``.marker(alpha, edge_color, ...)`` → marker / bar appearance
- ``.x_axis(label, rotation, ...)``   → horizontal axis
- ``.y_axis(show, label, ...)``       → vertical axis
