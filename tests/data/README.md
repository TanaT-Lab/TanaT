# Test Data

Committed offline files used by the TanaT test suite.  

## Layout

```
data/
  static/                       shared across both temporal variants
    static.csv                  scalar static features
    static_embeddings.parquet   embedding vectors (Array[Float32, 128])

  datetime/                     temporal columns as Datetime[us]
    sequence_main.parquet       core entity features
    sequence_extra.csv          extra entity features + observed_at timestamp

  timestep/                     temporal columns as Float64 (numeric day offset from 2000-01-01)
    sequence_main.parquet       core entity features
    sequence_extra.csv          extra entity features
```

---

## ID Sparsity

Intentionally designed to exercise sparse joins:

| ID range | In sequence data | In static data |
|----------|-----------------|----------------|
| 1–10     | ✓               | ✗              |
| 11–50    | ✓               | ✓              |
| 51–60    | ✗               | ✓              |
| 36–50    | main only        | ✓              |

---

## File Details

### `static/static.csv`
50 rows · IDs 11–60

| Column               | Type    | Description                  |
|----------------------|---------|------------------------------|
| `id`                 | Int64   | Entity identifier            |
| `age`                | UInt8   | Age in years (18–90)         |
| `group`              | String  | Category label (A / B / C / D) |
| `is_active`          | Boolean | Activity flag                |
| `membership_duration`| Int64   | Duration in days (30–1499)   |

### `static/static_embeddings.parquet`
50 rows · IDs 11–60

| Column             | Type              | Description              |
|--------------------|-------------------|--------------------------|
| `id`               | Int64             | Entity identifier        |
| `summary_embedding`| Array[Float32,128]| Static summary embedding |

### `datetime/sequence_main.parquet` · `timestep/sequence_main.parquet`
~267 rows · IDs 1–50 · 3–8 rows per ID

| Column      | datetime type     | timestep type | Description                    |
|-------------|-------------------|---------------|--------------------------------|
| `id`        | Int64             | Int64         | Entity identifier              |
| `start`     | Datetime[us]      | Float64       | Interval start                 |
| `end`       | Datetime[us]      | Float64       | Interval end                   |
| `duration`  | Duration[us]      | Float64       | end − start                    |
| `value`     | Float64           | Float64       | Numeric measurement            |
| `status`    | String            | String        | Label (ok / warn / error / pending) |
| `flag_valid`| Boolean           | Boolean       | Quality flag                   |
| `token_emb` | Array[Float32,32] | Array[Float32,32] | Per-row token embedding    |

### `datetime/sequence_extra.csv` · `timestep/sequence_extra.csv`
~198 rows · IDs 1–35 only

| Column          | datetime type | timestep type | Description               |
|-----------------|---------------|---------------|---------------------------|
| `id`            | Int64         | Int64         | Entity identifier         |
| `start`         | Datetime[us]  | Float64       | Interval start            |
| `end`           | Datetime[us]  | Float64       | Interval end              |
| `duration`      | Int64 (µs)    | Float64       | end − start               |
| `response_time` | Int64         | Int64         | Response time in ms       |
| `observed_at`   | Datetime[us]  | *(absent)*    | Observation timestamp     |

---

## Design Notes

- **Intervals contain overlaps and gaps** : within each ID, consecutive rows have a random mix of: ~30% overlap (next `start` is inside the current interval), ~20% large gap (5–20 days between `end` and next `start`), ~50% small/no gap. This is the core use-case for `IntervalSequencePool`. `StateSequencePool` ignores stored `end` values and infers `T_END = next(T_START)`.
- **`duration` type mismatch is intentional** : `sequence_main.parquet` stores `Duration[us]`, `sequence_extra.csv` stores `Int64` (µs). After builder concat, the pool exposes `Int64`. This exercises the cast feature.
- **Float64 timestep** values represent days elapsed since 2000-01-01 (midnight UTC).
- **Seed 0** : all files are fully reproducible.

