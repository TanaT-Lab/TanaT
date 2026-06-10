# Changelog

This page contains the complete changelog for TanaT, documenting all notable changes, new features, bug fixes, and improvements across versions.

## [v0.10.2] - June 2026
**Polars casting and filtering fixes**

### Added
- Added a `strict` parameter (defaulting to `True`) to `cast_features()`, allowing lenient casting (`strict=False`) where non-convertible values are nullified instead of crashing the Polars engine.

### Fixed
- Fixed an evaluation order bug where entity filtering (`filter_entities`) was applied before feature casts, causing a Polars `ComputeError` when a scope expression depended on a newly cast type.

## [v0.10.1] - June 2026
**Package structure cleanup**

### Changed
- Subtype registration now relies on explicit `__init__.py` imports 
- `type/__init__.py` files own their subtype imports consistently across all modules

### Dependencies
- Requires `tanat-utils >= 0.0.6`

## [v0.10.0] - June 2026
**Polars Engine & Arrow Store** - Full rewrite of the data layer for scalable, lazy sequence and trajectory manipulation

### Migration
- **Repository moved to GitHub**: TanaT is now hosted at [Github](https://github.com/tanat/tanat).
  The [Inria GitLab repository](https://gitlab.inrclaudeia.fr/tanat/core/tanat) remains archived for historical versions up to v0.9.0.

### Added
- **Polars-based data engine**: all internal data frames are now `polars.LazyFrame`, enabling lazy evaluation, predicate pushdown, and significant memory savings on large datasets
- **Arrow file store**: sequences and trajectories are persisted as Apache Arrow files (`SequenceStore`, `TrajectoryStore`), replacing the previous in-memory/pandas approach
- **Virtual context layer** (`VirtualStore`): lightweight copy-on-write mechanism for type conversions and derived views without materialising intermediate data
- **Ingestion sources**: unified `AbstractSource` interface with `DataFrameSource`, `CsvSource`, `ParquetSource`, and `SqlSource` for SQL query ingestion
- **Cast recipe system** (`tanat.cast`): structured type-cast overrides (`SequenceCastRecipe`, `TrajectoryCastRecipe`) applied lazily at view time, with validation (`probe()`) before execution
- **Frame assembler layer** (`SequenceFrameAssembler`, `TrajectoryFrameAssembler`): centralises construction of view-schema LazyFrames, decoupling store layout from user-facing columns

### Changed
- **Breaking**: API redesign, v0.10.0 is not fully backward-compatible with v0.9.x

### Notes
- **Compatibility**: Python ≥3.10 and <3.14
- Python 3.14 not recommended due to dependency wheel compatibility issues
- Python 3.9 is no longer supported (Polars requires ≥3.10)

---
*For historical versions below, please refer to the [Inria GitLab archive](https://gitlab.inria.fr/tanat/core/tanat).*

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
