"""
Plot a Manhattan-style p-value figure for features from a WORC-style
StatisticalTestFeatures CSV (Ttest, Welch, Wilcoxon, Mann-Whitney, Chi2).

Adapts:
- WORC.plotting.plot_pvalues_features.manhattan_importance
- WORC.featureprocessing.StatisticalTestFeatures (CSV -> plot pipeline)

Points with Mann-Whitney p < threshold (default 0.05) are annotated with
their feature label.
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_GROUP_MAPPING = {
    0: 'Histogram',
    1: 'Shape',
    2: 'Orientation',
    3: 'GLCM',
    4: 'GLRLM',
    5: 'GLSZM',
    6: 'GLDM',
    7: 'NGTDM',
    8: 'Gabor',
    9: 'Semantic',
    10: 'DICOM',
    11: 'LoG',
    12: 'Vessel',
    13: 'LBP',
    14: 'Phase',
    15: 'Texture',
}


def assign_feature_group(name):
    n = name.lower()
    if 'hf_' in n:
        return 0
    if 'sf_' in n:
        return 1
    if 'of_' in n:
        return 2
    if 'glcm_' in n or 'glcmms_' in n:
        return 3
    if 'glrlm_' in n:
        return 4
    if 'glszm_' in n:
        return 5
    if 'gldm_' in n:
        return 6
    if 'ngtdm_' in n:
        return 7
    if 'gabor_' in n:
        return 8
    if 'semf_' in n:
        return 9
    if 'df_' in n:
        return 10
    if 'logf_' in n:
        return 11
    if 'vf_' in n:
        return 12
    if 'lbp_' in n:
        return 13
    if 'phasef_' in n:
        return 14
    if 'tf_' in n:
        return 15
    raise KeyError(f"Cannot find any known feature group in key {name}.")


def shorten_feature_name(name):
    replacements = [
        'CalcFeatures_', 'featureconverter_', 'PREDICT_', 'PyRadiomics_',
        'Pyradiomics_', 'predict_', 'pyradiomics_', '_predict', '_pyradiomics',
        'original_', 'train_', 'test_', '1_0_',
        'hf_', 'sf_', 'of_',
        'GLCM_', 'GLCMMS_', 'GLRLM_', 'GLSZM_', 'GLDM_', 'NGTDM_',
        'Gabor_', 'semf_', 'df_', 'logf_', 'vf_', 'Frangi_', 'LBP_',
        'phasef_', 'tf_',
        '_CT_0', '_MR_0', '_MRI_0', 'CT_0', 'MR_0', 'MRI_0',
    ]
    out = name
    for r in replacements:
        out = out.replace(r, '')
    return out.strip('_')


def manhattan_importance(values, labels, feature_labels,
                         output_png=None, mapping=None,
                         threshold_annotated=0.05,
                         title=None, top_n=10):
    """Manhattan-style scatter of -log(p) values, grouped by feature class.

    Adapted from WORC.plotting.plot_pvalues_features.manhattan_importance.
    Annotates the ``top_n`` features with the smallest p-values that also
    satisfy p < threshold_annotated. If ``top_n`` is None, all points below
    the threshold are annotated.
    """
    f = plt.figure(figsize=(20, 10))

    positions = np.arange(len(values))
    values = np.asarray(values, dtype=float)
    unique_labels = sorted(set(list(labels)))
    n_labels = len(unique_labels)
    colormap = ['#7dcfe2', '#4b78b5', 'darkgrey', 'dimgray'] * max(n_labels, 1)

    for lnum, i in enumerate(unique_labels):
        for pnum in range(len(positions)):
            if labels[pnum] == i:
                positions[pnum] += lnum
        plot_positions = [p for p, l in zip(positions, labels) if l == i]
        plot_values = [v for v, l in zip(values, labels) if l == i]
        plt.scatter(plot_positions, plot_values, c=colormap[lnum])

    label_previous = labels[0]
    pos_previous = positions[0]
    color_end = []
    vlines = []
    p = positions[0]
    for i, p in zip(labels, positions):
        if i != label_previous:
            color_end.append((pos_previous + p) / 2.0)
            label_previous = i
            pos_previous = p
            vlines.append(p - 1)
    color_end.append((pos_previous + p) / 2.0)

    ymax = np.max(values)
    for i in range(0, 100):
        if 10 ** (-i) < ymax:
            ymaxlim = i - 1
            break

    ymin = np.min(values)
    pos_vals = values[values > 0]
    yposmin = max(np.min(pos_vals), np.finfo(values.dtype).eps) if pos_vals.size else np.finfo(values.dtype).eps
    for i in range(0, 100):
        if 10 ** (-i) < (ymin if ymin > 0 else yposmin):
            yminlim = i
            break

    plt.gca().invert_yaxis()
    if ymin > 0:
        plt.yscale('log')
        plt.ylim((10 ** -ymaxlim, 10 ** -yminlim))
    else:
        plt.yscale('symlog', linthresh=10 ** -yminlim)
        plt.ylim((10 ** -ymaxlim, 0.0))
    plt.xlim((0, max(positions)))

    plt.yticks([10 ** -i for i in range(ymaxlim, yminlim + 1)],
               [f'10-{i}' for i in range(ymaxlim, yminlim + 1)])
    if mapping is None:
        plt.xticks(color_end, np.arange(len(color_end)) + 1, size=16,
                   rotation=45, ha='right')
    else:
        xticks = [mapping[i] for i in unique_labels]
        plt.xticks(color_end, xticks, size=10, rotation=45, ha='right')

    plt.vlines(vlines, 10 ** -ymaxlim, 10 ** -yminlim,
               linestyles='dotted', linewidth=0.3)

    if threshold_annotated > 10 ** -yminlim:
        y_value_annotated = threshold_annotated
        plt.hlines(threshold_annotated, 0, max(positions),
                   linestyles='dashed', linewidth=1, color='magenta')
    else:
        y_value_annotated = 10 ** -yminlim
        plt.hlines(10 ** -yminlim, 0, max(positions),
                   linestyles='dashed', linewidth=1, color='magenta')

    plt.annotate(f'p={round(threshold_annotated, 5)}',
                 (1, y_value_annotated),
                 xytext=(1, y_value_annotated * 0.95),
                 size=8, color='magenta')

    if 0.05 > 10 ** -yminlim and not np.isclose(threshold_annotated, 0.05):
        plt.hlines(0.05, 0, max(positions),
                   linestyles='dashed', linewidth=1, color='magenta')
        plt.annotate('p=0.05', (1, 0.05),
                     xytext=(1, 0.05 * 0.95), size=8, color='magenta')

    plt.xlabel("Feature groups", size=12)
    plt.ylabel("P-value Mann-Whitney U", size=12)
    if title:
        plt.title(title, size=14)

    offset = np.clip(len(values) / 200, 0.1, 100)
    candidates = [(p, v, t) for p, v, t in zip(positions, values, feature_labels)
                  if v < threshold_annotated]
    candidates.sort(key=lambda x: x[1])
    if top_n is not None:
        candidates = candidates[:top_n]
    annotated_pos = [p for p, _, _ in candidates]
    annotated_values = [v for _, v, _ in candidates]
    annotated_labels = [t for _, _, t in candidates]

    y_offset = -0.1
    for x, y, text in zip(annotated_pos, annotated_values, annotated_labels):
        plt.annotate(text, (x, y),
                     xytext=(x + offset, y * (1 - y_offset)), size=6)
        y_offset = -y_offset

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')

    if output_png is not None:
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved as {output_png}!")

    return f


def load_pvalues_csv(csv_path):
    """Load a WORC StatisticalTestFeatures CSV.

    The file has a 1-row title (label name), then a sub-header
    ['Label','Ttest','Welch','Wilcoxon','Mann-Whitney','Chi2',''].
    Returns (label_name, DataFrame).
    """
    with open(csv_path, 'r') as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if any(c.strip() for c in r)]

    label_name = rows[0][0] if rows else ''
    header = rows[1]
    data_rows = rows[2:]

    df = pd.DataFrame(data_rows, columns=header)
    if '' in df.columns:
        df = df.drop(columns=[''])

    for col in ['Ttest', 'Welch', 'Wilcoxon', 'Mann-Whitney', 'Chi2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return label_name, df


def plot_pvalues_from_csv(csv_path, output_png=None, threshold=0.05,
                          bonferroni=False, top_n=10):
    """Load a CSV and produce a Manhattan plot for the Mann-Whitney p-values.

    Annotates the ``top_n`` features (smallest p) below the threshold (after
    optional Bonferroni correction). Pass ``top_n=None`` to annotate all.
    """
    label_name, df = load_pvalues_csv(csv_path)

    raw_objects = df['Label'].tolist()
    raw_pvalues = df['Mann-Whitney'].tolist()

    objects, p_values = [], []
    for o, p in zip(raw_objects, raw_pvalues):
        if not (p is None or (isinstance(p, float) and np.isnan(p))):
            objects.append(o)
            p_values.append(float(p))

    if not p_values:
        raise ValueError(f"No valid Mann-Whitney p-values in {csv_path}")

    if bonferroni:
        threshold = threshold / len(p_values)

    labels = [assign_feature_group(o) for o in objects]
    objects = [shorten_feature_name(o) for o in objects]

    sort_idx = np.argsort(np.asarray(labels))
    p_values = [p_values[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]
    objects = [objects[i] for i in sort_idx]

    if output_png is None:
        base = os.path.splitext(csv_path)[0]
        output_png = f"{base}_mwu_manhattan.png"

    return manhattan_importance(values=p_values,
                                labels=labels,
                                feature_labels=objects,
                                output_png=output_png,
                                mapping=FEATURE_GROUP_MAPPING,
                                threshold_annotated=threshold,
                                title=label_name,
                                top_n=top_n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='Path to StatisticalTestFeatures CSV file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output PNG path (default: <csv>_mwu_manhattan.png)')
    parser.add_argument('-t', '--threshold', type=float, default=0.05,
                        help='Annotation/significance threshold (default 0.05)')
    parser.add_argument('--bonferroni', action='store_true',
                        help='Apply Bonferroni correction to the threshold')
    parser.add_argument('-n', '--top-n', type=int, default=10,
                        help='Annotate only the top N features (smallest p). '
                             'Use 0 to annotate all below threshold.')
    args = parser.parse_args()

    top_n = None if args.top_n == 0 else args.top_n
    plot_pvalues_from_csv(args.csv, output_png=args.output,
                          threshold=args.threshold,
                          bonferroni=args.bonferroni,
                          top_n=top_n)


if __name__ == '__main__':
    main()
