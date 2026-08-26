#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser(description='Replace a prefix in a CSV column')
parser.add_argument('--input', '-i', required=True, help='Input CSV file')
parser.add_argument('--output', '-o', required=True, help='Output CSV file')
parser.add_argument('--old', required=True, help='Old prefix to remove/replace')
parser.add_argument('--new', required=True, help='New prefix to insert (use empty string to remove)')
parser.add_argument('--col', type=int, default=1, help='0-based index of the column to modify (default 1)')
parser.add_argument('--delimiter', default=',', help='CSV delimiter (default ",")')
parser.add_argument('--preview', type=int, default=0, help='Print N sample changes and exit without writing')
args = parser.parse_args()

old = args.old.rstrip('/')
new = args.new.rstrip('/')
col = args.col

def replace_prefix(value, old_prefix, new_prefix):
    if not value:
        return value
    # normalize both to avoid double slashes
    if value.startswith(old_prefix + '/'):
        rest = value[len(old_prefix) + 1:]
        return (new_prefix + '/' + rest) if new_prefix else rest
    if value == old_prefix:
        return new_prefix
    return value

rows = []
changes = []
with open(args.input, newline='') as inf:
    reader = csv.reader(inf, delimiter=args.delimiter)
    for i, row in enumerate(reader):
        if len(row) > col:
            orig = row[col]
            newval = replace_prefix(orig, old, new)
            if newval != orig:
                changes.append((i, orig, newval))
                row[col] = newval
        rows.append(row)

if args.preview:
    print(f"Total rows: {len(rows)}, Changes: {len(changes)}\n")
    for idx, o, n in changes[:args.preview]:
        print(f"Row {idx}:\n  OLD: {o}\n  NEW: {n}\n")
    if args.preview > 0:
        print('Preview only; no file written.')
    raise SystemExit(0)

with open(args.output, 'w', newline='') as outf:
    writer = csv.writer(outf, delimiter=args.delimiter)
    writer.writerows(rows)

print(f"Wrote {args.output}. Total rows: {len(rows)}, changed: {len(changes)}")
