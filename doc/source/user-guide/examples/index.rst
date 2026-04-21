Examples Gallery
================

Short, focused examples demonstrating TanaT's core features.
Each example is self-contained and can be adapted to your own data.

.. toctree::
   :hidden:

   container/index
   visualization/index
   zeroing/index
   metric_entity/index
   metric_sequence/index
   metric_trajectory/index
   clustering/index

Data Containers
---------------

Build and explore TanaT's core data structures: event, interval and state sequences,
as well as trajectories that combine multiple sequence types for a single individual.

.. raw:: html

    <div class="tanat-gallery">
        <a href="container/event.html" class="tanat-gallery-card">
            <p><strong>Event Sequences</strong><br>
            Build an EventSequencePool and explore its contents.</p>
        </a>

        <a href="container/state.html" class="tanat-gallery-card">
            <p><strong>State Sequences</strong><br>
            Build a StateSequencePool and explore contiguous states.</p>
        </a>

        <a href="container/interval.html" class="tanat-gallery-card">
            <p><strong>Interval Sequences</strong><br>
            Build an IntervalSequencePool and explore overlapping and gapped intervals.</p>
        </a>

        <a href="container/trajectory.html" class="tanat-gallery-card">
            <p><strong>Trajectories</strong><br>
            Combine multiple sequence types into Trajectory and TrajectoryPool.</p>
        </a>
    </div>

Visualisation
-------------

Visualize sequences using :class:`~tanat.visualization.SequenceVisualizer`.

.. raw:: html

    <div class="tanat-gallery">
        <a href="visualization/timeline.html" class="tanat-gallery-card">
            <p><strong>Timeline</strong><br>
            Sequences aligned on a shared time axis, flat or category stacking.</p>
        </a>

        <a href="visualization/barplot.html" class="tanat-gallery-card">
            <p><strong>Barplot</strong><br>
            Count occurrences, relative frequencies, or total durations per label.</p>
        </a>

        <a href="visualization/spanplot.html" class="tanat-gallery-card">
            <p><strong>Spanplot</strong><br>
            Duration distributions as box, violin, or strip plots.</p>
        </a>

        <a href="visualization/distribution.html" class="tanat-gallery-card">
            <p><strong>Distribution</strong><br>
            State-occupancy areas over time for state sequence pools.</p>
        </a>
    </div>

Temporal Alignment
------------------

Align sequences to a common reference date (T0 / index date) before computing
distances or visualising cross-individual comparisons.

.. raw:: html

    <div class="tanat-gallery">
        <a href="zeroing/sequence_t0.html" class="tanat-gallery-card">
            <p><strong>Sequence Zeroing</strong><br>
            Define a reference point (T0) for sequences.</p>
        </a>

        <a href="zeroing/trajectory_t0.html" class="tanat-gallery-card">
            <p><strong>Trajectory Zeroing</strong><br>
            Define a reference point (T0) for trajectories across multiple sub-pools.</p>
        </a>
    </div>

Metrics
-------

.. raw:: html

    <div style="border-left: 3px solid #e0e0e0; padding-left: 20px; margin: 20px 0;">

Entity Metrics
~~~~~~~~~~~~~~

Distance metrics for individual entities.

.. raw:: html

    <div class="tanat-gallery">
        <a href="metric_entity/hamming.html" class="tanat-gallery-card">
            <p><strong>Hamming Distance</strong><br>
            Calculate distance between categorical states.</p>
        </a>
    </div>

Sequence Metrics
~~~~~~~~~~~~~~~~

Distance metrics for entire temporal sequences.

.. raw:: html

    <div class="tanat-gallery">
        <a href="metric_sequence/linear_pairwise.html" class="tanat-gallery-card">
            <p><strong>LinearPairwise</strong><br>
            Position-wise alignment with aggregation.</p>
        </a>

        <a href="metric_sequence/edit.html" class="tanat-gallery-card">
            <p><strong>Edit Distance</strong><br>
            Needleman-Wunsch alignment with insertions/deletions.</p>
        </a>

        <a href="metric_sequence/lcs.html" class="tanat-gallery-card">
            <p><strong>LCS</strong><br>
            Longest Common Subsequence distance.</p>
        </a>

        <a href="metric_sequence/dtw.html" class="tanat-gallery-card">
            <p><strong>DTW</strong><br>
            Dynamic Time Warping with flexible alignment.</p>
        </a>

        <a href="metric_sequence/softdtw.html" class="tanat-gallery-card">
            <p><strong>SoftDTW</strong><br>
            Differentiable DTW variant.</p>
        </a>

        <a href="metric_sequence/chi2.html" class="tanat-gallery-card">
            <p><strong>Chi²</strong><br>
            Chi-squared distance between distributions.</p>
        </a>
    </div>

Trajectory Metrics
~~~~~~~~~~~~~~~~~~

Distance metrics for multi-sequence trajectories.

.. raw:: html

    <div class="tanat-gallery">
        <a href="metric_trajectory/aggregation.html" class="tanat-gallery-card">
            <p><strong>Aggregation</strong><br>
            Compare trajectories with flexible per-type metrics.</p>
        </a>
    </div>

.. raw:: html

    </div>

Clustering
----------

Partition sequences or trajectories into groups using distance matrices.

.. raw:: html

    <div class="tanat-gallery">
        <a href="clustering/hierarchical.html" class="tanat-gallery-card">
            <p><strong>Hierarchical Clustering</strong><br>
            Nested partitions with dendrograms and four linkage methods.</p>
        </a>

        <a href="clustering/pam.html" class="tanat-gallery-card">
            <p><strong>PAM Clustering</strong><br>
            Medoid-based clustering robust to outliers.</p>
        </a>

        <a href="clustering/clara.html" class="tanat-gallery-card">
            <p><strong>CLARA Clustering</strong><br>
            Scalable medoid-based clustering for large datasets.</p>
        </a>
    </div>
