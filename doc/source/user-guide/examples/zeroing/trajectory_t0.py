"""
Trajectory-level zeroing
=========================

.. note::

   Trajectory-level zeroing is not yet implemented.

   T0 alignment is currently available on individual sequence pools
   (:class:`~tanat.sequence.IntervalSequencePool`,
   :class:`~tanat.sequence.type.event.pool.EventSequencePool`,
   :class:`~tanat.sequence.type.state.pool.StateSequencePool`).
   Support for :class:`~tanat.trajectory.pool.TrajectoryPool` is planned
   for a future release.

   In the meantime, apply ``set_t0`` independently on each component
   sequence pool before building the trajectory pool.

See :doc:`sequence_t0` for the full guide on zeroing sequence pools,
and :doc:`../../../reference/zeroing` for the complete reference.
"""
