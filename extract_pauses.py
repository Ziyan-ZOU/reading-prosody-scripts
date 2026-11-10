#!/usr/bin/env python3
"""Extract manually annotated pauses; never modify TextGrids.

Usage:
  python3 extract_mehdi_pauses.py /path/to/TextGrids
  python3 extract_mehdi_pauses.py /path/to/file.TextGrid --output /path/to/results

Python 3.9+, standard library only. Recurses into folders.
Tiers are selected by NUMBER: 9=sentence, 10=pause, 11=expected boundary.
Only nonempty annotations are extracted. Long-format Praat IntervalTier and
TextTier (point boundaries in tier 11) are supported; other formats are logged.

Matching is descriptive, NOT a judgment of appropriate phrasing:
  interval boundary: interval overlaps pause, or gap <= --tolerance-ms;
  point boundary: point is in pause, or gap <= --tolerance-ms.
Default tolerance=0 ms. All matching pairs are kept (no forced 1:1 match).
An expected interval is used AS AN INTERVAL, not silently reduced to its end.
No duration threshold is applied to manual pauses. Labels 1,2,... are retained
as labels, not interpreted as categories. All times are seconds except *_ms.
Absent pause counts/totals are 0; means with no pauses are blank (NA in R).

Outputs (UTF-8 BOM CSV):
 sentences.csv: each labeled tier-9 interval, including zero-pause sentences;
 pauses.csv: all labeled tier-10 intervals, including unassigned ones;
 expected_boundaries.csv: all labeled tier-11 entries, including unmatched ones;
 boundary_pause_matches.csv: every matching pair in the same sentence;
 extraction_log.csv: tier names, errors and assignment/overlap warnings.
Sentence assignment requires FULL containment; crossing or ambiguous entries
are retained unassigned and logged, never silently assigned or clipped.
Overlapping pauses are retained; sentence pause total uses their UNION so time
is not double-counted. Mean is the mean of original annotated durations.
"""
import argparse
import csv
from pathlib import Path
import re
import sys

NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
QUOTED = r'"((?:[^\"]|\"\")*)"'

def unquote(s):
    return s.replace('""', '"')

def read_grid(path):
    raw = path.read_bytes()
    text = raw.decode('utf-16') if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else raw.decode('utf-8-sig')
    blocks = re.split(r'(?m)^\s*item\s*\[\d+\]:\s*$', text)[1:]
    if not blocks:
        raise ValueError('Expected long-format Praat TextGrid; save as text in Praat.')
    tiers = []
    for block in blocks:
        name = re.search(r'\bname\s*=\s*' + QUOTED, block)
        kind = re.search(r'\bclass\s*=\s*' + QUOTED, block)
        if not name or not kind:
            raise ValueError('Missing tier name/class')
        entries = []
        if kind[1] == 'IntervalTier':
            chunks = re.split(r'\bintervals\s*\[\d+\]:', block)[1:]
            for i, chunk in enumerate(chunks, 1):
                a = re.search(r'\bxmin\s*=\s*(' + NUM + ')', chunk)
                b = re.search(r'\bxmax\s*=\s*(' + NUM + ')', chunk)
                label = re.search(r'\btext\s*=\s*' + QUOTED, chunk)
                if not (a and b and label):
                    raise ValueError('Malformed interval')
                start, end = float(a[1]), float(b[1])
                if end < start:
                    raise ValueError('Negative interval duration')
                entries.append(dict(index=i, label=unquote(label[1]), start=start, end=end, kind='interval'))
        elif kind[1] == 'TextTier':
            for i, chunk in enumerate(re.split(r'\bpoints\s*\[\d+\]:', block)[1:], 1):
                t = re.search(r'\bnumber\s*=\s*(' + NUM + ')', chunk)
                label = re.search(r'\bmark\s*=\s*' + QUOTED, chunk)
                if not (t and label):
                    raise ValueError('Malformed point')
                entries.append(dict(index=i, label=unquote(label[1]), start=float(t[1]), end=float(t[1]), kind='point'))
        else:
            raise ValueError('Unsupported tier class: ' + kind[1])
        tiers.append((unquote(name[1]), kind[1], entries))
    return tiers

def labeled(entries):
    return [e for e in entries if e['label'].strip()]

def union_duration(entries):
    total, right = 0.0, float('-inf')
    for e in sorted(entries, key=lambda x: x['start']):
        total += max(0.0, e['end'] - max(right, e['start']))
        right = max(right, e['end'])
    return total

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', type=Path, help='TextGrid or directory')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--tolerance-ms', type=float, default=0.0)
    args = ap.parse_args()
    if args.tolerance_ms < 0:
        ap.error('Tolerance must be nonnegative')
    source = args.input.expanduser().resolve()
    if not source.exists():
        ap.error('Input does not exist')
    files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.textgrid')
    if not files:
        ap.error('No TextGrid files found')
    root = source.parent if source.is_file() else source
    output = (args.output.expanduser() if args.output else root / 'pause_extraction').resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {k: [] for k in ('sentences', 'pauses', 'expected_boundaries', 'boundary_pause_matches', 'extraction_log')}
    def log(file, status, detail):
        tables['extraction_log'].append(dict(file=file, status=status, detail=detail))
    failed = 0
    for path in files:
        file = path.relative_to(root).as_posix()
        try:
            tiers = read_grid(path)
            if len(tiers) < 11:
                raise ValueError('Fewer than 11 tiers')
            if tiers[8][1] != 'IntervalTier' or tiers[9][1] != 'IntervalTier':
                raise ValueError('Tiers 9 and 10 must be IntervalTier')
        except Exception as exc:
            log(file, 'error', str(exc)); failed += 1; continue
        log(file, 'tiers', '; '.join(f'{n}: {tiers[n-1][0]}' for n in (9, 10, 11)))
        ss, ps, bs = (labeled(tiers[i][2]) for i in (8, 9, 10))
        def row(e, prefix):
            owners = [s for s in ss if s['start'] - 1e-9 <= e['start'] and e['end'] <= s['end'] + 1e-9]
            owner = owners[0] if len(owners) == 1 else None
            if owner is None:
                log(file, 'warning', f'{prefix} tier entry {e["index"]}: {len(owners)} containing sentences; retained unassigned')
            return dict(file=file, sentence_id=owner['index'] if owner else '', sentence_label=owner['label'] if owner else '', **{prefix+'_id': e['index'], prefix+'_label': e['label'], prefix+'_start_s': e['start'], prefix+'_end_s': e['end'], prefix+'_duration_ms': (e['end']-e['start'])*1000})
        pr = [row(e, 'pause') for e in ps]
        br = [dict(row(e, 'boundary'), boundary_kind=e['kind']) for e in bs]
        for p in pr:
            p['matched_boundary_count'] = 0
        for b in br:
            b['matched_pause_count'] = 0
        for p in pr:
            for b in br:
                if p['sentence_id'] == '' or p['sentence_id'] != b['sentence_id']:
                    continue
                gap = max(0.0, b['boundary_start_s'] - p['pause_end_s'], p['pause_start_s'] - b['boundary_end_s'])
                if gap <= args.tolerance_ms / 1000 + 1e-9:
                    p['matched_boundary_count'] += 1; b['matched_pause_count'] += 1
                    tables['boundary_pause_matches'].append(dict(file=file, sentence_id=p['sentence_id'], sentence_label=p['sentence_label'], pause_id=p['pause_id'], boundary_id=b['boundary_id'], pause_label=p['pause_label'], boundary_label=b['boundary_label'], interval_gap_ms=gap*1000, overlap_ms=max(0.0, min(p['pause_end_s'],b['boundary_end_s'])-max(p['pause_start_s'],b['boundary_start_s']))*1000, pause_start_minus_boundary_start_ms=(p['pause_start_s']-b['boundary_start_s'])*1000, pause_start_minus_boundary_end_ms=(p['pause_start_s']-b['boundary_end_s'])*1000, tolerance_ms=args.tolerance_ms))
        tables['pauses'].extend(pr); tables['expected_boundaries'].extend(br)
        for s in ss:
            pp = [p for p in pr if p['sentence_id'] == s['index']]
            bb = [b for b in br if b['sentence_id'] == s['index']]
            intervals = [dict(start=p['pause_start_s'], end=p['pause_end_s']) for p in pp]
            total = union_duration(intervals)
            raw_total = sum(p['pause_duration_ms'] for p in pp)
            if raw_total - total*1000 > 1e-6:
                log(file, 'warning', f'Sentence {s["index"]}: overlapping pauses; total uses interval union')
            duration = s['end'] - s['start']
            tables['sentences'].append(dict(file=file, sentence_id=s['index'], sentence_label=s['label'], sentence_start_s=s['start'], sentence_end_s=s['end'], sentence_duration_ms=duration*1000, pause_count=len(pp), total_pause_duration_ms=total*1000, mean_pause_duration_ms=raw_total/len(pp) if pp else '', pause_proportion=total/duration if duration else '', boundary_count=len(bb), boundaries_with_pause=sum(b['matched_pause_count']>0 for b in bb), pauses_with_boundary=sum(p['matched_boundary_count']>0 for p in pp)))
        if not ss:
            log(file, 'warning', 'No labeled sentences in tier 9')
        log(file, 'ok', f'{len(ss)} sentences; {len(ps)} pauses; {len(bs)} expected boundaries')
    # Explicit schemas retain usable headers even if a table is empty.
    schemas = {
        'sentences': 'file sentence_id sentence_label sentence_start_s sentence_end_s sentence_duration_ms pause_count total_pause_duration_ms mean_pause_duration_ms pause_proportion boundary_count boundaries_with_pause pauses_with_boundary',
        'pauses': 'file sentence_id sentence_label pause_id pause_label pause_start_s pause_end_s pause_duration_ms matched_boundary_count',
        'expected_boundaries': 'file sentence_id sentence_label boundary_id boundary_label boundary_start_s boundary_end_s boundary_duration_ms boundary_kind matched_pause_count',
        'boundary_pause_matches': 'file sentence_id sentence_label pause_id boundary_id pause_label boundary_label interval_gap_ms overlap_ms pause_start_minus_boundary_start_ms pause_start_minus_boundary_end_ms tolerance_ms',
        'extraction_log': 'file status detail',
    }
    for name, rows in tables.items():
        with (output / (name+'.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=schemas[name].split())
            writer.writeheader(); writer.writerows(rows)
    print(f'Processed {len(files)-failed}/{len(files)} files. Results: {output}')
    print('Boundary matches describe temporal proximity, not appropriateness. Check extraction_log.csv.')
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
