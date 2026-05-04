Criteria
========

Criteria are composable filtering primitives that answer two questions:

* **Which IDs match?**:
  :py:meth:`~tanat.sequence.base.pool.SequencePool.which`
  returns the set of matching IDs.
* **Which entity rows match?**:
  :py:meth:`~tanat.sequence.base.pool.SequencePool.filter_entities`
  returns filtered sequence(s) view.

See :doc:`../../../reference/criterion` for the complete reference.
