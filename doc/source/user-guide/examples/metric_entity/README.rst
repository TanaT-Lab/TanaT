Entity Metrics
==============

Entity-level metrics measure the distance between single point-in-time observations
within sequences. 

The implemented metrics are based on the comparison of the entity's 
features and does not handle the timestamp of an entity. In the spirit of TanaT
the timestamps are handle at the sequence level.
More specifically, an entity metric can be apply whatever the nature of the entity (event, 
state or interval).

Sequence-level metrics rely on an entity metric as a building block.

.. note::
   Hamming is the most common choice for categorical features while L2 metrics
   is most suitable with numerical features.
   The combined metric enables to define a metric based several features (numerical
   or categorical) through a linear combinaison of other entity metrics.
