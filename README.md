# QuantiPhy Baseline: Parsing, Video Grouping, and EDA

A runnable baseline for the first data-engineering stage of the QuantiPhy Challenge:

1. Normalize the official CSV/Parquet schema.
2. Parse each natural-language question into validated JSON.
3. Parse the numerical prior and optional depth metadata.
4. Group all QA rows by `video_id` into a shared scene-level record.
5. Infer missing units conservatively from same-video context.
6. Generate EDA tables, plots, and a parser review queue.

The included validation CSV is the official 159-row QuantiPhy validation split. Videos are not bundled; place them in a local folder only when you need to attach paths to the grouped records.

## Project structure

```text
quantiphy_baseline/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── validation_dataset.csv
│   ├── interim/
│   └── processed/
│       ├── parsed_questions.jsonl       # generated
│       └── grouped_by_video.jsonl       # generated
├── examples/
│   ├── parsed_question.example.json
│   └── video_group.example.json
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/eda/                         # generated EDA report, CSVs, PNGs
├── scripts/
│   ├── 01_parse_questions.py
│   ├── 02_group_by_video.py
│   ├── 03_run_eda.py
│   └── run_all.py
├── src/quantiphy_baseline/
│   ├── cli.py
│   ├── dataio.py
│   ├── eda.py
│   ├── grouping.py
│   ├── normalizer.py
│   ├── schemas.py
│   └── parsers/
│       ├── depth_parser.py
│       ├── prior_parser.py
│       └── question_parser.py
└── tests/
```

## Installation

```bash
cd quantiphy_baseline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Run the complete pipeline

```bash
python scripts/run_all.py
```

Or:

```bash
make all
```

The default paths are controlled by `configs/default.yaml`.

To associate downloaded videos with groups:

```bash
python scripts/run_all.py --video-dir /path/to/validation_videos
```

The expected video filename is `<video_id>.<extension>`.

## Run each stage separately

```bash
python scripts/01_parse_questions.py \
  --input data/raw/validation_dataset.csv \
  --output data/processed/parsed_questions.jsonl

python scripts/02_group_by_video.py \
  --input data/processed/parsed_questions.jsonl \
  --output data/processed/grouped_by_video.jsonl

python scripts/03_run_eda.py \
  --parsed data/processed/parsed_questions.jsonl \
  --grouped data/processed/grouped_by_video.jsonl \
  --output-dir outputs/eda
```

Installed CLI aliases are also available:

```bash
quantiphy-parse
quantiphy-group
quantiphy-eda
quantiphy-run-all
```

## Parsed question schema

Each JSONL row contains:

```json
{
  "qa_id": "2274",
  "video_id": "internet_0027",
  "raw_question": "What is the velocity of the orange ball at 1.00s in cm/s?",
  "normalized_question": "what is the velocity of the orange ball at 1.00 s in cm/s?",
  "quantity_family": "velocity",
  "quantity_subtype": "velocity",
  "output_dimension": "speed",
  "output_unit": "cm/s",
  "output_unit_source": "question",
  "component": "scalar",
  "temporal": {
    "mode": "instantaneous",
    "time_s": 1.0,
    "start_s": null,
    "end_s": null
  },
  "target_entities": ["the orange ball"],
  "relation_type": "attribute",
  "prior": {
    "description": "billiard ball diameter",
    "property_family": "size",
    "property_subtype": "diameter",
    "value": 57.2,
    "unit": "mm"
  },
  "depth_observations": [],
  "parse_confidence": 1.0,
  "warnings": []
}
```

The parser distinguishes:

- quantity family: `size`, `distance`, `displacement`, `speed`, `velocity`, `acceleration`;
- subtype: `length`, `width`, `height`, `diameter`, `orbital_diameter`, `total_distance`, etc.;
- temporal semantics: `static`, `initial`, `final`, `instantaneous`, `interval`, `whole_video_average`, `whole_video_total`;
- component: `scalar`, `horizontal`, `vertical`, `planar_xy`, `orbital`;
- relation: object attribute, object pair, source-to-target, or traveled path.

Known validation typos are normalized for parsing, while `raw_question` remains unchanged.

## Why grouping happens after parsing

Some information is only recoverable from the set of questions attached to one video. For example, four `orbital diameter` questions in one validation video omit the output unit. Other size questions for the same video consistently use meters, so the grouping stage fills `m` and marks:

```json
"output_unit_source": "video_context"
```

This inference is conservative. It is applied only when one unit has a unique majority and appears at least twice among same-video questions with the same physical dimension.

## EDA outputs

`outputs/eda/` contains:

- `summary.json`: compact machine-readable summary;
- `eda_report.md`: short human-readable report;
- `parsed_questions_flat.csv`: flattened parser output;
- `video_summary.csv`: one row per video;
- count tables for source, video type, inference type, quantity, unit, temporal mode, component, and prior family;
- cross-tabs such as inference type × quantity and quantity × unit;
- `parse_issues.csv`: rows that need manual or LLM review;
- standalone PNG charts.

The EDA includes ground-truth values on a log10 scale because numerical values span multiple orders of magnitude.

## Tests

```bash
pytest
```

The test suite verifies:

- official validation schema cleanup;
- prior parsing, including timestamped priors;
- typo normalization;
- time and interval extraction;
- entity and pair-relation extraction;
- parenthetical measurement descriptions;
- grouping into 24 videos;
- same-video unit inference.

## Important design boundary

This parser is intentionally deterministic. It extracts the structured measurement request but does not solve visual coreference, segmentation, tracking, or physics. `target_entities` are textual grounding prompts for the next stage.

For a production system, add an optional VLM fallback only for rows in `parse_issues.csv`, then validate the returned JSON with the existing Pydantic schema. Do not replace the deterministic path for easy cases.

## Data attribution

QuantiPhy validation dataset: PaulineLi/QuantiPhy-validation, CC BY 4.0. See the official dataset card and repository for competition rules and updates.
