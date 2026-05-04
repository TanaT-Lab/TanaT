# TanaT
**Temporal Analysis of Trajectories** 

*TanaT* is a powerful Python library designed for advanced temporal sequence analysis, with specialized focus on patient care pathways and complex temporal data structures (trajectories).

## Stay Updated

[Subscribe](https://sympa.inria.fr/sympa/subscribe/tanat-community) to our newsletter to get updates, release notes, and example notebooks straight to your inbox!


## What Makes TanaT Different

TanaT bridges the gap between traditional time series analysis and complex temporal sequence modeling by offering:

- **Expressive Data Representation**: Handle event sequences, interval sequences, and state sequences with unified APIs
- **Advanced Distance Metrics**: Specialized metrics for temporal data including DTW, edit distance, and custom metrics
- **Flexible Clustering**: State-of-the-art clustering algorithms adapted for temporal sequences and trajectories
- **Extensible Architecture**: Modular design allowing easy integration of new methods and metrics

## Core Capabilities

### Data Structures
- **Event Sequences**: Point-in-time events with rich feature descriptions
- **Interval Sequences**: Time-spanning events with overlapping support
- **State Sequences**: Continuous state representations with temporal transitions
- **Trajectories**: Multi-dimensional temporal data combining multiple sequence types

### Analysis Methods
- **Distance Computation**: Dynamic Time Warping, Edit Distance, Longest Common Subsequence, and more
- **Clustering**: Specialized algorithms for grouping similar temporal patterns
- **Filtering & Selection**: Advanced criteria-based data selection and manipulation
- **Visualization**: Comprehensive tools for temporal data exploration
- **Survival analysis**: Streamline the preparation of survival targets and time-to-event data

## Scientific Foundation

TanaT draws inspiration from established frameworks:

- **TraMineR** (R): State sequence analysis methodologies
- **aeon** & **tslearn**: Time series analysis best practices

## Architecture Overview

TanaT provides a comprehensive suite of interconnected modules for end-to-end temporal sequence analysis:


| Feature | Description |
|---------|-------------|
| **Simulation** | Easily generate synthetic data to explore the framework's features |
| **Visualization** | Explore and interpret temporal sequences through rich visual representations |
| **Data Wrangling** | Manipulate, filter, and transform temporal data with flexible operations |
| **Survival Analysis** | Integrate time-to-event modeling and survival techniques |
| **Metrics & Clustering** | Apply specialized distance metrics and clustering algorithms for temporal data |
| **Workflow Orchestration** | Organize your analysis, ensuring reproducible code and modular pipelines |

## Resources

- **Documentation**: [Full Documentation](https://tanat-lab.github.io/TanaT/)
- **Source Code**: [GitHub Repository](https://github.com/TanaT-Lab/TanaT)
- **Issues & Support**: [Issue Tracker](https://github.com/TanaT-Lab/TanaT/issues)

## Citation

If you use TanaT in your research, please cite:

```bibtex
@inproceedings{tanat2025,
title={Towards a Library for the Analysis of Temporal Sequences},
authors={Thomas Guyet and Arnaud Duvermy},
booktitle={Proceedings of AALTD, ECML Workshop on Advanced Analytics and Learning on Temporal Data},
year={2025},
pages={16}
}
```

## Affiliation & Support

TanaT is actively developed within the [AIstroSight](https://team.inria.fr/aistrosight/) Inria Team.

The development has been supported by:

* **2024-2025**: [AIRacles Chair](https://www.bernoulli-lab.fr/project/chaire-ai-racles/) (Inria/APHP/CS)
* **2025-present**: PEPR/SafePaw project (Government funding managed by the French National Research Agency under France 2030, reference number ANR-22-PESN-0005)

## Team

**Core Development Team**
- **Arnaud Duvermy** - Architecture & Core Development
- **Thomas Guyet** - Project Leadership & Research Methods

**Contact**: [TanaT](mailto:tanat@inria.fr)

---
*TanaT is open source software designed to advance temporal sequence analysis in research and industry applications.*