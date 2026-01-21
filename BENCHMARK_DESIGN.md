# Performance Benchmark Design Document

## Executive Summary

This benchmark answers two questions:
1. **How fast does the workload run?** (end-to-end wall-clock time)
2. **Is there a risk of OOM?** (peak RSS relative to available memory)

## Point-by-Point Design Decisions

### 1. What is being measured

**Decision: End-to-end pipeline cost**

**Rationale:**
- The benchmark measures the complete classifier_bin workload from cold start to final visualization
- This includes ingestion, conversion, transformation, aggregation, and rendering
- This reflects real-world usage: users run `sm.run_startup()` followed by notebook cells
- Isolated operation microbenchmarks would not capture:
  - GC pressure from accumulated allocations
  - Framework conversion overhead (polars ↔ pandas)
  - Memory fragmentation from repeated operations
  - Cache effects in realistic workflows

**Tradeoff:** We sacrifice the ability to pinpoint specific slow operations, but gain realistic cost modeling for engineering decisions (e.g., "Can this run in a 4GB container?").

### 2. Runtime measurement

**Decision: Wall-clock time via `time.perf_counter()`**

**Rationale:**
- Wall-clock time reflects real user experience and latency
- Captures everything: CPU, I/O, GC pauses, memory allocation
- Sufficient for answering "Is this too slow?"
- CPU time would hide GC cost, which is critical for memory-intensive workloads

**Methodology:**
- Single timing bracket around each operation
- No warmup runs (reflects cold-start performance)
- Explicit `gc.collect()` before each operation to ensure consistent starting state
- Single run per configuration (multiple dataset sizes tested instead)
- Display output suppressed using `IPython.utils.capture.capture_output(display=True)` to prevent widget clutter

**Explicitly NOT measured:** CPU time, instruction counts, micro-op analysis (not relevant for OOM risk assessment)

### 3. Memory pressure / OOM risk

**Decision: Peak RSS as primary metric**

**Rationale:**
- RSS (Resident Set Size) is what the OS sees and what triggers OOM kills
- Peak RSS directly answers "Will this OOM in a container with X GB limit?"
- Measured via `psutil.Process().memory_info().rss`
- Sampled at 100ms intervals during execution to capture transient peaks

**Secondary metric removed:** tracemalloc
- tracemalloc measures Python-level allocations
- Does NOT capture: C extension allocations (numpy, polars, arrow), memory mapping, shared libraries
- Can significantly underestimate actual memory pressure
- Overhead of tracing can distort measurements
- **Verdict: Removed** - adds complexity without improving OOM risk assessment

**Interpretation guideline:**
- Peak RSS < 50% of available memory: Safe
- Peak RSS 50-80%: Caution (OS pressure, potential swap)
- Peak RSS > 80%: High OOM risk

### 4. Conversion vs. compute cost

**Decision: Report combined, but log ingestion time separately**

**Rationale:**
- **Combined timing reflects real usage**: Users cannot avoid ingestion cost
- However, separating ingestion provides diagnostic value:
  - High ingestion/total ratio → I/O or conversion bottleneck
  - Low ratio → compute-bound workload
- Implementation: Log ingestion time (parquet read + merge) separately, but include in total

**Explicitly NOT separated:**
- Polars ↔ pandas conversions during computation
- GC triggered by allocations
- These are intrinsic to the workload and cannot be optimized away without architectural changes

**Tradeoff:** We accept that pandas/polars interop cost is "baked in" to the measurements. This is realistic - users must pay this cost unless we complete full migration.

### 5. Result presentation

**Decision: Plain tables with absolute values, deltas, and ratios**

**Color rejected with justification:**
- Color gradients are low-signal: "Is red bad? How bad?"
- Not colorblind-friendly
- Loses information when printed or in non-color environments
- Requires subjective interpretation of color scale

**Implementation: 4 separate tables**

1. **Peak RSS (MB) - Config A: Baseline (per_context=False)**
   - Columns: Operation name | 100K | 1M | 10M | % of Total
   - % of Total: each operation's memory as percentage of max for largest dataset

2. **Peak RSS (MB) - Config B: Per-context (per_context=True)**
   - Columns: Operation name | 100K | 1M | 10M | vs Baseline
   - vs Baseline: average overhead percentage across all dataset sizes

3. **Runtime (seconds) - Config A: Baseline**
   - Columns: Operation name | 100K | 1M | 10M | % of Total
   - % of Total: each operation's time as percentage of total time for largest dataset

4. **Runtime (seconds) - Config B: Per-context**
   - Columns: Operation name | 100K | 1M | 10M | vs Baseline
   - vs Baseline: average overhead percentage across all dataset sizes

**Key features:**
- Absolute values with units (MB, seconds)
- Dynamic column ordering (sorted by dataset size)
- Thousands separators for readability (e.g., 1,240 MB)
- Separate baseline and per_context tables for clarity
- Overhead percentages show cost of enabling per_context analysis

### 6. Interpretation and conclusions

**What this benchmark DOES support:**

1. Scaling characteristics: Does memory grow linearly, sub-linearly, or super-linearly with data size?
2. Container sizing: What memory limit is needed for dataset size X?
3. Per-context cost: What is the overhead of enabling per-context analysis?
4. Feasibility assessment: Can we process 10M rows on this hardware?

**What this benchmark DOES NOT support:**

1. ❌ "Polars is faster than pandas" - workload contains both, no isolated comparison
2. ❌ "Operation X is the bottleneck" - end-to-end measurement only
3. ❌ Claims about production performance on different hardware/OS/Python versions
4. ❌ Predictions for data sizes not tested (extrapolation is unreliable)
5. ❌ Multi-core scaling characteristics (single-threaded workload)

**Conclusions will be stated as:**
- "At 10M rows, peak RSS is 4.9 GB, suggesting a minimum 8 GB container (with 40% safety margin)"
- "Per-context analysis adds 35% runtime overhead and 28% memory overhead at 1M rows"

**Conclusions will NOT include:**
- "Perfect constant-time performance!"
- "Sub-linear scaling proves efficiency!"
- "This is 10x faster than before!"

### 7. Scope control

**Decision: Single end-to-end workload, no micro-benchmarks**

**Rationale:**
- Goal is to answer two specific questions about ONE workload
- The classifier_bin workload is well-defined and represents real usage
- Micro-benchmarking individual operations adds complexity without decision value
- If we discover performance issues, we can profile separately (not in this benchmark)

**Explicitly rejected:**
- ❌ Benchmarking individual functions (merge, aggregate, plot)
- ❌ Multiple workload types (regression, multi-class)
- ❌ Parameterized operation selection
- ❌ Profiler integration (flamegraphs, line profiling)

**In scope:**
- ✅ Multiple data sizes: base dataset (~100K) + scaled versions (configurable via `SCALE_SIZES`)
- ✅ Two configurations: per_context OFF vs ON (measured separately, not compared directly)
- ✅ Single workload: classifier_bin notebook flow
- ✅ 18 operations measured individually: startup + 17 notebook operations
- ✅ Two metrics per operation: wall-clock time, peak RSS

**Complexity budget:** ~200 lines of benchmark code, ~50 lines of result reporting per table (4 tables total)

## Workload Definition

The benchmark measures **18 operations individually** from the classifier_bin notebook:

```python
# 1. Startup (ingestion + merge)
sm.run_startup(config_path='.')

# 2-18. All notebook operations (measured individually)
sm.show_info(plot_help=True)
sm.ExploreSubgroups()
sm.feature_alerts()
sm.feature_summary()
sm.cohort_comparison_report()
sm.target_feature_summary()
sm.ExploreModelEvaluation()
sm.plot_model_evaluation({}, 'Readmitted within 30 Days', 'Risk30DayReadmission', (0.10, 0.20), per_context=...)
sm.ExploreFairnessAudit()
sm.show_cohort_summaries(by_target=False, by_score=False)  # Always runs baseline
sm.show_cohort_summaries(by_target=True, by_score=True)    # Always runs per_context
sm.ExploreCohortEvaluation()
sm.ExploreAnalyticsTable()
sm.ExploreOrdinalMetrics()
sm.ExploreCohortOrdinalMetrics()
sm.ExploreCohortLeadTime()
sm.ExploreCohortOutcomeInterventionTimes()
```

**Key implementation details:**
- Each operation is timed and memory-profiled independently
- `show_cohort_summaries` runs TWICE (baseline + per_context) regardless of config to measure both scenarios
- Explore widgets have their "Update" button programmatically clicked to trigger visualization generation
- All display output is suppressed using `capture_output(display=True)` to keep logs clean

## Configuration Management

The benchmark modifies `config.yml` in place for each dataset:

```python
# Backup original config
shutil.copy('config.yml', 'config.yml.backup')

# Modify paths using regex (config has separate fields: data_dir, prediction_path, event_path)
config = re.sub(r'prediction_path:\s*"[^"]*"', f'prediction_path: "{pred_relative}"', config)
config = re.sub(r'event_path:\s*"[^"]*"', f'event_path: "{event_relative}"', config)

# Run benchmark with modified config
sm.run_startup(config_path='.')

# Restore original config
shutil.move('config.yml.backup', 'config.yml')
```

**Rationale:**
- Tests actual file-based configuration (not in-memory dataframe passing)
- Validates real-world workflow where users modify config files
- Backup/restore pattern ensures clean state between runs

## Dataset Scaling

Scaled datasets are generated dynamically based on `SCALE_SIZES` parameter:

```python
SCALE_SIZES = [1_000_000, 10_000_000]  # 1M, 10M rows
```

- Base dataset used as-is (e.g., 104K rows → displayed as "100K" after rounding)
- Scaled datasets replicated with unique IDs: `encounter_id + "_r" + replica_number`
- Size-to-suffix mapping generated dynamically (not hardcoded): `{1_000_000: '1m', 10_000_000: '10m'}`
- Formatted with rounding to nearest 100K for readability

## Memory Sampling Strategy

Peak RSS is sampled during execution:

```python
# Background thread samples RSS every 100ms
import threading

peak_rss = 0
def sample_memory():
    global peak_rss
    while measuring:
        current = psutil.Process().memory_info().rss
        peak_rss = max(peak_rss, current)
        time.sleep(0.1)
```

**Rationale:**
- 100ms sampling catches transient spikes (GC, large allocations)
- Background thread avoids measurement interference
- Alternative (snapshot before/after) would miss intermediate peaks

## Error Handling

Operations that fail are recorded with `None` values for time/memory and flagged in the output:
- Failed operations shown with "✗" in progress log
- Error message truncated to first 50 characters
- Results table includes failed operations for completeness

## Known Limitations

1. **Single-threaded:** Does not measure multi-core scaling
2. **Clean environment:** Does not model memory pressure from other processes
3. **Synthetic scaling:** Replicated data may not reflect real-world data characteristics
4. **No I/O variance:** SSD read times are assumed stable
5. **No GC tuning:** Uses default Python GC parameters

These limitations are acceptable for the stated goals (runtime and OOM risk for this specific workload).

## Open Questions (Not Addressed)

- Is 10M rows sufficient, or should we test 50M/100M? (Out of scope: depends on user requirements)
- Should we test different hardware? (Out of scope: users can run benchmark on their own hardware)
- Should we include profiling? (Out of scope: diagnostic tool, not benchmark)

---

**Review sign-off:** This design prioritizes clarity, realism, and decision-usefulness over theoretical purity or comprehensive coverage. Every design choice is justified in terms of the two core questions.
