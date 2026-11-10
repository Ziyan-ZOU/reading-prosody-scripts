#!/usr/bin/env python3
"""Word acoustics: Wordlevel->YN, FocusWord->Contrastive, AmPhrase->Ambiguous.

Install: python3 -m pip install praat-parselmouth numpy
Run: python3 extract_mehdi_f0.py --input '/path/to/data'
Matching WAV/AIFF/FLAC files must have the same stem as their TextGrid.
Use --audio-dir if audio lives in a separate directory tree.
Outputs are UTF-8-BOM CSV files (Excel compatible); undefined F0 is blank.
Each labelled interval in tiers 3, 6 and 8 is treated as a WORD.
intervals.csv: duration (seconds), F0 summary (Hz), 10 equal-duration bin
F0 means (Hz), voiced-frame counts per bin, and whole-word intensity (dB).
No frame-level CSV is written. No interpolation across unvoiced frames.
Intensity is averaged in the energy domain, then converted to dB, using
Praat's Intensity: Get mean (energy). It is not calibrated sound level;
recording gain/distance differences can affect between-recording comparisons.
This script never edits audio or TextGrids. See --help for options.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

PAIRS = [(3, 'Wordlevel', 4, 'YN'),
         (6, 'FocusWord', 5, 'Contrastive'),
         (8, 'AmPhrase', 7, 'Ambiguous')]
BASE = ['textgrid', 'target_tier', 'target_interval', 'label',
        'parent_tier', 'sentence_id', 'parent_start_s', 'parent_end_s',
        'start_s', 'end_s', 'duration_s', 'match_status']
STATS = ['n_frames', 'n_voiced', 'voiced_fraction', 'f0_status',
         'f0_mean_hz', 'f0_median_hz', 'f0_min_hz', 'f0_max_hz', 'f0_range_hz']
BIN_FIELDS = [f'f0_bin_{i:02d}_mean_hz' for i in range(1, 11)]
BIN_COUNTS = [f'f0_bin_{i:02d}_n_voiced' for i in range(1, 11)]
INTENSITY_FIELDS = ['intensity_mean_db', 'intensity_status']


def intervals(grid, tier):
    for i in range(1, int(call(grid, 'Get number of intervals', tier)) + 1):
        label = call(grid, 'Get label of interval', tier, i).strip()
        if label:
            yield (i, float(call(grid, 'Get start time of interval', tier, i)),
                   float(call(grid, 'Get end time of interval', tier, i)), label)


def annotations(grid, name):
    """Validate tier index AND name: never silently select a different tier."""
    n = int(call(grid, 'Get number of tiers'))
    for child, cname, parent, pname in PAIRS:
        for index, expected in [(child, cname), (parent, pname)]:
            if index > n or call(grid, 'Get tier name', index).casefold() != expected.casefold():
                raise ValueError(f'Expected tier {index} named {expected!r}; check tier order/names')
            if not call(grid, 'Is interval tier', index):
                raise ValueError(f'Tier {index} must be an IntervalTier')
        parents = list(intervals(grid, parent))
        for i, start, end, label in intervals(grid, child):
            matches = [p for p in parents if p[1] - 1e-6 <= start and end <= p[2] + 1e-6]
            p = matches[0] if len(matches) == 1 else None
            yield dict(textgrid=name, target_tier=cname, target_interval=i,
                       label=label, parent_tier=pname, sentence_id=p[3] if p else '',
                       parent_start_s=p[1] if p else '', parent_end_s=p[2] if p else '',
                       start_s=start, end_s=end, duration_s=end-start,
                       match_status='ok' if p else 'unmatched_or_crossing')


def statistics(values, minimum):
    voiced = values[np.isfinite(values) & (values > 0)]
    n = len(voiced)
    status = 'ok' if n >= minimum else ('no_voiced_frames' if n == 0 else 'too_few_voiced_frames')
    result = dict(n_frames=len(values), n_voiced=n,
                  voiced_fraction=n / len(values) if len(values) else '', f0_status=status)
    for key in STATS[4:]:
        result[key] = ''
    if status == 'ok':
        result.update(f0_mean_hz=float(np.mean(voiced)), f0_median_hz=float(np.median(voiced)),
                      f0_min_hz=float(np.min(voiced)), f0_max_hz=float(np.max(voiced)),
                      f0_range_hz=float(np.ptp(voiced)))
    return result


def binned_f0(times, values, start, end):
    """Assign frame centers to ten half-open bins; retain empty bins as blank."""
    edges = np.linspace(start, end, 11)
    result = {}
    for i in range(10):
        selected = values[(times >= edges[i]) & (times < edges[i + 1])]
        voiced = selected[np.isfinite(selected) & (selected > 0)]
        result[BIN_FIELDS[i]] = float(np.mean(voiced)) if len(voiced) else ''
        result[BIN_COUNTS[i]] = len(voiced)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='TextGrid folder (searched recursively)')
    parser.add_argument('--audio-dir', type=Path, help='Separate audio folder, recursively searched')
    parser.add_argument('--output', type=Path, help='NEW output folder; default: input/acoustic_output')
    parser.add_argument('--floor', type=float, default=75, help='Pitch floor in Hz (default 75; review for your speakers)')
    parser.add_argument('--ceiling', type=float, default=600, help='Pitch ceiling in Hz (default 600; review for your speakers)')
    parser.add_argument('--step', type=float, default=0.01, help='Frame spacing in seconds (default .01)')
    parser.add_argument('--intensity-floor', type=float, default=75, help='Praat intensity minimum pitch, Hz; sets smoothing window (default 75)')
    parser.add_argument('--min-voiced', type=int, default=3, help='Minimum voiced frames for summary statistics')
    parser.add_argument('--channel', type=int, default=0, help='0: mix channels; 1,2,...: select an audio channel')
    parser.add_argument('--annotations-only', action='store_true', help='Check tier/parent matching without audio or F0')
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error('--input must be an existing directory')
    if not 0 < args.floor < args.ceiling or args.step <= 0 or args.min_voiced < 1 or args.channel < 0 or args.intensity_floor <= 0:
        parser.error('Require 0 < floor < ceiling, step > 0, intensity-floor > 0, min-voiced >= 1, channel >= 0')
    if args.audio_dir and not args.audio_dir.is_dir():
        parser.error('--audio-dir must be an existing directory')
    grids = sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.casefold() == '.textgrid')
    if not grids:
        parser.error('No TextGrid files found')
    output = args.output or args.input / 'acoustic_output'
    if output.exists():
        parser.error(f'Output already exists: {output}. Choose another --output to avoid overwriting results.')
    audio_root = args.audio_dir or args.input
    audio_index = {}
    for p in audio_root.rglob('*'):
        if p.is_file() and p.suffix.casefold() in {'.wav', '.aiff', '.aif', '.flac'}:
            audio_index.setdefault(p.stem.casefold(), []).append(p)
    output.mkdir(parents=True)
    issues = []
    count = 0
    fields = BASE if args.annotations_only else BASE + STATS + BIN_FIELDS + BIN_COUNTS + INTENSITY_FIELDS
    with (output / 'intervals.csv').open('w', encoding='utf-8-sig', newline='') as sf:
        summary = csv.DictWriter(sf, fieldnames=fields)
        summary.writeheader()
        for gridfile in grids:
            rel = str(gridfile.relative_to(args.input))
            try:
                grid = parselmouth.read(str(gridfile))
                rows = list(annotations(grid, rel))
                for row in rows:
                    if row['match_status'] != 'ok':
                        issues.append((rel, 'parent_match', f"{row['target_tier']} interval {row['target_interval']} has no unique containing labelled parent"))
                if args.annotations_only:
                    summary.writerows(rows)
                    count += len(rows)
                    continue
                candidates = audio_index.get(gridfile.stem.casefold(), [])
                local = [p for p in candidates if p.parent == gridfile.parent]
                if local:
                    candidates = local
                elif args.audio_dir:
                    expected_parent = args.audio_dir / gridfile.relative_to(args.input).parent
                    mirrored = [p for p in candidates if p.parent == expected_parent]
                    if mirrored:
                        candidates = mirrored
                if len(candidates) != 1:
                    raise ValueError(f'Expected one matching audio, found {len(candidates)}; use identical stems and mirrored folders')
                audio = candidates[0]
                sound = parselmouth.Sound(str(audio))
                if args.channel:
                    if args.channel > sound.n_channels:
                        raise ValueError('Requested channel does not exist')
                    sound = sound.extract_channel(args.channel)
                elif sound.n_channels > 1:
                    sound = sound.convert_to_mono()
                for row in rows:
                    if row['start_s'] < sound.xmin - 1e-6 or row['end_s'] > sound.xmax + 1e-6:
                        raise ValueError('An annotated interval is outside the audio time range')
                # Analyze the entire recording once, retaining acoustic context at boundaries.
                pitch = sound.to_pitch_ac(time_step=args.step, pitch_floor=args.floor,
                                          pitch_ceiling=args.ceiling)
                intensity = sound.to_intensity(minimum_pitch=args.intensity_floor,
                                               time_step=args.step, subtract_mean=True)
                times = pitch.xs()
                frequencies = pitch.selected_array['frequency']
                for row in rows:
                    mask = (times >= row['start_s']) & (times < row['end_s'])
                    vals = frequencies[mask]
                    mean_intensity = float(call(intensity, 'Get mean', row['start_s'], row['end_s'], 'energy'))
                    intensity_ok = np.isfinite(mean_intensity)
                    summary.writerow({**row, **statistics(vals, args.min_voiced),
                                      **binned_f0(times[mask], vals, row['start_s'], row['end_s']),
                                      'intensity_mean_db': mean_intensity if intensity_ok else '',
                                      'intensity_status': 'ok' if intensity_ok else 'undefined'})
                    count += 1
                print(f'OK: {rel} ({len(rows)} labelled intervals)')
            except Exception as exc:
                issues.append((rel, 'file_error', str(exc)))
                print(f'SKIP/ERROR: {rel}: {exc}', file=sys.stderr)
    with (output / 'issues.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['textgrid', 'issue', 'detail'])
        w.writerows(issues)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(parselmouth_version=parselmouth.__version__, method='raw autocorrelation',
                    interval_count=count, issue_count=len(issues),
                    f0_bins=10, bin_definition='Ten equal-duration bins per word; mean of voiced frame centers; at least one voiced frame per bin; counts provided',
                    intensity_method='Whole-recording Intensity, subtract_mean=True; full-word energy-domain mean expressed in dB',
                    note='Whole-recording pitch; interval frames selected by frame center; unvoiced F0 blank; no interpolation or normalization; all child intervals are words')
    (output / 'settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done: {count} intervals; {len(issues)} issues. Results: {output.resolve()}')
    return 1 if any(i[1] == 'file_error' for i in issues) else 0


if __name__ == '__main__':
    sys.exit(main())
