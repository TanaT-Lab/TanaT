# Changelog

This page contains the complete changelog for TanaT, documenting all notable changes, new features, bug fixes, and improvements across versions.

## [v0.9.0] - December 2025
**Performance & Scalability** - JIT parallel computing and memory-mapped storage

### Added
- **Numba-optimized metrics**: JIT-compiled kernels
- **Parallel matrix computation**: Pairwise distances computed in parallel with `prange`
- **Memory-mapped matrices**: On-disk storage via `MatrixStorageOptions` for large dataset
- **Enhanced progress display**: Structured, hierarchical progress tracking with timing for clustering and metric operations
- **Faceted plots**: New `facet()` method creates grid visualizations from any static feature for multi-dimensional data exploration

### Notes
- **Compatibility**: Python ≥3.9 and <3.14
- Python 3.14 not recommended due to dependency wheels compatibility issues

---

## [v0.8.0] - November 2025
**Enhanced Validation & Analytics** - Robust validation, metadata inference and statistical methods

### Added
- Improve sequence/trajectory settings validation (Robust Pydantic-based validation, ...)
- Metadata inference/management system
- Sequence type conversion
- Clustering methods : Partitioning Around Medoids, Clustering Large Applications
- Position/Rank-based methods: `head()`, `tail()`, `slice()` methods with negative indexing and step sampling
- Statistical analysis methods: `describe()` method and `statistics` property for computing comprehensive sequence and trajectory statistics (entropy, vocabulary size, transition counts, etc.)

### Notes
- **Compatibility**: Python ≥3.9 and <3.14
- Python 3.14 not recommended due to dependency wheels compatibility issues

---

## [v0.7.0] - August 2025
**Foundation Release** - Core architecture complete, ready for beta testing

### Added
- Core architecture for temporal sequence analysis
- Support for event, interval, and state sequences
- Distance metrics for entities, sequences and trajectories
- Clustering algorithms
- Survival analysis
- Criteria for data wrangling
- Basic visualization tools
- Comprehensive API documentation

### Notes
- This is a preliminary release focusing on the core architecture
- Ready for beta testing and feedback
- Future releases will expand functionality and improve stability
- **Compatibility**: Python >3.9 required
