"""Chart training time per model, in the slide deck's visual language.

Consumes the JSON written by benchmark_epoch_time.py.

    python plot_epoch_time.py benchmark_epoch_time.json --out epoch_time.png
    python plot_epoch_time.py ... --mode per-epoch --out epoch_time_flat.png

--mode cumulative (default) plots elapsed training time against epoch number:
the lines fan out, so the cost difference is something you can see rather than
read. --mode per-epoch plots the constant per-epoch cost as flat lines.

Colours, ink and typeface come from deck/figures.py so the figure drops onto a
parchment slide unchanged; the background is transparent for the same reason.

The deck palette is low-chroma by design and CANNOT separate four series by hue
(worst pair dE 7.6 to normal vision, 2.0 under deuteranopia — validated). So
hue is not the identity channel here: every line carries its own dash pattern,
its own marker, and a direct label at its end. The one accent colour is spent
on the paper's own model.
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- deck/figures.py, verbatim -------------------------------------------
INK = '#3B2816'          # the deck's body colour
RULE = '#6B5138'         # box outlines
FAINT = '#8A7A63'        # captions, group frames
WARM = '#8A5A1A'         # the deck's accent
GREEN = '#4A6B3A'
CREAM = '#EDE4D2'
FONT = 'Georgia'

# name -> (display label, colour, linewidth, dash, marker, z)
STYLE = {
    'integral':   ('Integral Transformer (ours)', WARM,  2.8, (0, ()),       'o', 6),
    'baseline':   ('Attention baseline',          INK,   1.7, (0, (7, 3)),   's', 5),
    'meanpool':   ('Mean pool',                   FAINT, 1.7, (0, (2, 2.5)), '^', 4),
    'truncation': ('Truncation (first 512 tokens)', GREEN, 1.9, (0, ()),     'D', 5),
}
DEFAULT = ('model', RULE, 1.6, (0, (4, 3)), 'v', 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('json_file', nargs='?', default='benchmark_epoch_time.json')
    p.add_argument('--out', default='epoch_time.png')
    p.add_argument('--mode', default='bar',
                   choices=['bar', 'cumulative', 'per-epoch'])
    p.add_argument('--epochs', type=int, default=None,
                   help='epochs to plot (default: Config.num_epochs = 10)')
    p.add_argument('--opaque', action='store_true',
                   help='paint the deck cream behind the figure instead of transparency')
    p.add_argument('--dpi', type=int, default=240)
    return p.parse_args()


def stagger(ax, entries, min_gap_px=44):
    """Push end-labels apart vertically without moving them off their line.

    Three of the four models land within 1% of each other, so their true label
    positions collide; nudging them apart is the only way the values stay
    readable, and the leader line keeps each label tied to its own curve.
    """
    entries = sorted(entries, key=lambda e: e[1], reverse=True)
    ys = [ax.transData.transform((0, e[1]))[1] for e in entries]
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < min_gap_px:
            ys[i] = ys[i - 1] - min_gap_px
    inv = ax.transData.inverted()
    return [(e, inv.transform((0, y))[1]) for e, y in zip(entries, ys)]


def render_bar(data, models, order, ref, n_epochs, args):
    """One panel, one bar per model, minutes per epoch. The accent marks the
    paper's model; the x labels carry identity, so colour is emphasis only."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.set_facecolor('none')
    fig.subplots_adjust(left=0.105, right=0.975, top=0.695, bottom=0.185)

    order = sorted(models.items(), key=lambda kv: kv[1]['epoch_s'], reverse=True)
    # wrap the parenthetical onto its own line so the tick labels stay upright
    names = [STYLE.get(n, DEFAULT)[0].replace(' (', chr(10) + '(', 1)
             for n, _ in order]
    vals = [r['epoch_s'] / 60 for _, r in order]
    top = max(vals)

    for i, ((name, r), v) in enumerate(zip(order, vals)):
        ours = name == 'integral'
        ax.bar(i, v, width=0.54, color=WARM if ours else RULE,
               edgecolor='none', zorder=3, alpha=1.0 if ours else 0.55)
        ax.text(i, v + top * 0.030, f'{v:,.1f} min', ha='center', va='bottom',
                color=INK, fontsize=12.5,
                fontweight='bold' if ours else 'normal')

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, fontsize=10.5, linespacing=1.4)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(0, top * 1.20)
    ax.set_ylabel('Minutes per epoch', color=INK, fontsize=11.5, labelpad=9)
    ax.yaxis.grid(True, color=FAINT, linewidth=0.6, alpha=0.32, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(FAINT)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=FAINT, labelsize=10, length=0)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)
    for lbl in ax.get_yticklabels():
        lbl.set_color(RULE)
    return fig, ax


def main():
    args = parse_args()
    with open(args.json_file, encoding='utf-8') as f:
        data = json.load(f)

    models = data['models']
    n_epochs = args.epochs or 10
    order = sorted(models.items(), key=lambda kv: kv[1]['epoch_s'], reverse=True)
    ref = models.get('baseline')

    # --- table view, so the figure is never the only way to read the numbers
    print(f"\n{'model':<32}{'min/epoch':>11}{f'min @ {n_epochs} ep':>15}{'vs baseline':>13}")
    csv_path = os.path.splitext(args.out)[0] + '.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'params', 'min_per_epoch', f'min_at_{n_epochs}_epochs',
                    'epoch_s', 'epoch_s_min', 'epoch_s_max', 'repeats',
                    'pct_vs_baseline'])
        for name, r in order:
            per = r['epoch_s'] / 60
            pct = ((r['epoch_s'] / ref['epoch_s'] - 1) * 100) if ref else 0.0
            print(f"{STYLE.get(name, DEFAULT)[0]:<32}{per:>11.2f}"
                  f"{per * n_epochs:>15.1f}{pct:>12.2f}%")
            w.writerow([STYLE.get(name, DEFAULT)[0], r['params'], round(per, 3),
                        round(per * n_epochs, 2), round(r['epoch_s'], 2),
                        round(r.get('epoch_s_min', 0), 2),
                        round(r.get('epoch_s_max', 0), 2),
                        r.get('repeats', 1), round(pct, 3)])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [FONT, 'Times New Roman', 'DejaVu Serif'],
    })
    if args.mode == 'bar':
        fig, ax = render_bar(data, models, order, ref, n_epochs, args)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.5))
        ax.set_facecolor('none')          # the parchment shows through, not white
        fig.subplots_adjust(left=0.085, right=0.685, top=0.755, bottom=0.185)

        cumulative = args.mode == 'cumulative'
        xs = list(range(0, n_epochs + 1)) if cumulative else [1, n_epochs]
        ends = []
        for name, r in order:
            label, colour, lw, dash, marker, z = STYLE.get(name, DEFAULT)
            per = r['epoch_s'] / 60
            ys = [per * x for x in xs] if cumulative else [per, per]
            ax.plot(xs, ys, color=colour, linewidth=lw, linestyle=dash, zorder=z,
                    marker=marker if cumulative else None, markersize=4.6,
                    markevery=list(range(1, len(xs))), markeredgewidth=0,
                    solid_capstyle='round', dash_capstyle='round')
            ends.append(((name, label, colour), ys[-1], per))

        top = max(e[1] for e in ends)
        ax.set_xlim(0 if cumulative else 0.6, n_epochs + (0.15 if cumulative else 0.4))
        ax.set_ylim(0, top * 1.08)
        ax.set_xlabel('Epoch', color=INK, fontsize=11.5, labelpad=7)
        ax.set_ylabel('Elapsed training time (minutes)' if cumulative
                      else 'Minutes per epoch', color=INK, fontsize=11.5, labelpad=9)
        ax.set_xticks(list(range(0, n_epochs + 1)) if cumulative
                      else list(range(1, n_epochs + 1)))

        # chrome: hairline horizontal grid only, no box, deck ink throughout
        ax.yaxis.grid(True, color=FAINT, linewidth=0.6, alpha=0.35, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(FAINT)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(colors=FAINT, labelsize=10, length=0)
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lbl.set_color(RULE)

        # direct end-labels: identity never rests on hue alone
        fig.canvas.draw()
        x_end = ax.get_xlim()[1]
        x_lab = x_end + n_epochs * 0.05
        for ((name, label, colour), y_true, per), y_lab in stagger(ax, ends):
            ax.plot([n_epochs, x_lab * 0.99], [y_true, y_lab], color=colour,
                    linewidth=0.8, alpha=0.5, clip_on=False, zorder=3)
            ax.text(x_lab, y_lab, label, va='center', ha='left', color=colour,
                    fontsize=10.2, clip_on=False,
                    fontweight='bold' if name == 'integral' else 'normal')
            ax.text(x_lab, y_lab - top * 0.05,
                    f'{y_true:,.0f} min' + (f'   {per:.1f} per epoch' if cumulative else ''),
                    va='center', ha='left', color=FAINT, fontsize=9.2, clip_on=False)

    # ---- headline: the one sentence the chart exists to make
    fig.text(0.085, 0.965, 'Training time: our model against the baselines',
             ha='left', va='top', color=INK, fontsize=16)
    integ = models.get('integral')
    if ref and integ:
        gap = (integ['epoch_s'] - ref['epoch_s']) / 60 * n_epochs
        pct = (integ['epoch_s'] / ref['epoch_s'] - 1) * 100
        ratio = order[0][1]['epoch_s'] / order[-1][1]['epoch_s']
        extra = (integ['params'] - ref['params']) / 1e6
        if args.mode == 'bar':
            msg = (f'Reading the whole judgment costs the same however you combine '
                   f'the chunks — the Integral\nTransformer runs {pct:+.1f}% against '
                   f'the attention baseline, for {extra:.1f}M more parameters.\n'
                   f'Truncation is {ratio:.0f}× cheaper only because it reads the '
                   f'first 512 tokens and stops.')
        else:
            msg = (f'Reading the whole judgment costs the same however you combine the '
                   f'chunks. Over {n_epochs} epochs the Integral Transformer takes\n'
                   f'{gap:+.0f} min ({pct:+.1f}%) against the attention baseline, for '
                   f'{extra:.1f}M more parameters. '
                   f'Truncation is {ratio:.0f}× cheaper only because it reads\n'
                   f'the first 512 tokens and stops.')
        fig.text(0.085, 0.905, msg, ha='left', va='top', color=RULE,
                 fontsize=10.4, linespacing=1.55)

    foot = [f"{data['device']} · batch {data['batch_size']} · "
            f"{'FP16' if data.get('fp16') else 'FP32'} · "
            f"{data['train_batches_per_epoch']} train + "
            f"{data['val_batches_per_epoch']} val batches per epoch",
            data['measurement']]
    for k, line in enumerate(foot):
        fig.text(0.085, 0.040 - 0.024 * k, line, ha='left', va='bottom',
                 color=FAINT, fontsize=8.0)

    fig.savefig(args.out, dpi=args.dpi, transparent=not args.opaque,
                facecolor=CREAM if args.opaque else 'none')
    print(f"\nWrote {args.out} and {csv_path}")


if __name__ == '__main__':
    main()
