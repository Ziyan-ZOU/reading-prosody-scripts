
# Reading Prosody Analysis Scripts

Developed by **Ziyan Zou**. These Python scripts extract acoustic measurements
and pause information from annotated oral reading recordings.

## Word-level acoustic measurements

`extract_mehdi_f0.py` extracts word duration, mean intensity over the full word
interval, and F0 summaries including minimum, maximum, and range. It divides
each word interval into 10 equal-duration bins and calculates mean F0 from the
voiced frames in each bin.

The audio and TextGrid must have the same base filename, for example
`sample.wav` and `sample.TextGrid`. The script expects these TextGrid tiers:

| Target tier | Parent tier |
| --- | --- |
| 3 `Wordlevel` | 4 `YN` |
| 6 `FocusWord` | 5 `Contrastive` |
| 8 `AmPhrase` | 7 `Ambiguous` |

```bash
python3 -m pip install praat-parselmouth numpy
python3 extract_mehdi_f0.py --input path/to/textgrids --audio-dir path/to/audio --output acoustic_results
```

Outputs: `intervals.csv`, `issues.csv`, and `settings.json`. The output
directory must not already exist.

## Pause and phrase-boundary analysis

`extract_mehdi_pauses.py` reads existing pause annotations and matches them
to expected phrase boundaries within each sentence. It requires long-format
Praat TextGrids with sentence annotations in tier 9, pause annotations in
tier 10, and expected boundaries in tier 11. It runs with Python 3.9+
and uses only the standard library.

```bash
python3 extract_mehdi_pauses.py path/to/textgrids --output pause_results
```

Outputs: `sentences.csv`, `pauses.csv`, `expected_boundaries.csv`,
`boundary_pause_matches.csv`, and `extraction_log.csv`.

## Citation

If you use these scripts, please credit Ziyan Zou and cite the GitHub release
corresponding to the version you used.

Zou, Z. (2026). Reading prosody analysis scripts (Version 1.0.0) [Computer software]. https://github.com/Ziyan-ZOU/reading-prosody-scripts/releases/tag/v1.0.0
