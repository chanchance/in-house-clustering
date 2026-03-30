"""
Generate matplotlib PNG slides (1920x1080) for clustering methods 1-5.
Output: output/slides/slide_{N:02d}_{name}.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgba
from scipy.spatial import Voronoi

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "/Users/jongchan/Desktop/claude/in-house-clustering/output/slides"
os.makedirs(OUT_DIR, exist_ok=True)

BG = '#1E1E2E'
TEXT_WHITE = 'white'
TEXT_GRAY  = '#CCCCCC'

FIG_W, FIG_H = 19.2, 10.8
DPI = 100


def new_fig():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def add_title(fig, title, accent):
    fig.text(0.5, 0.955, title, ha='center', va='top',
             fontsize=22, fontweight='bold', color=accent,
             fontfamily='DejaVu Sans')


def add_bullets(fig, lines, accent, x=0.72, y=0.82, dy=0.055):
    for i, line in enumerate(lines):
        fig.text(x, y - i * dy, line, ha='left', va='top',
                 fontsize=11, color=TEXT_GRAY, fontfamily='DejaVu Sans')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 1 — Decision Tree Clustering
# ══════════════════════════════════════════════════════════════════════════════
def slide_01():
    accent = '#4FC3F7'
    fig = new_fig()
    add_title(fig, 'Method 1 — Decision Tree Clustering', accent)

    ax = fig.add_axes([0.03, 0.06, 0.65, 0.83])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # colour palette for 16 leaves
    leaf_colors = plt.cm.tab20(np.linspace(0, 1, 16))

    # Tree layout: depth 0..4
    # Each node: (x_center, y_center, label, is_leaf, leaf_idx)
    depth_y = {0: 9.2, 1: 7.5, 2: 5.8, 3: 4.1, 4: 2.0}

    # Build tree positions breadth-first
    nodes = {}  # node_id -> (x, y)
    labels = {}
    is_leaf = {}
    leaf_idx = {}

    # root
    nodes[0] = (8, depth_y[0])
    labels[0] = 'feat_1\n≤ 0.71'
    is_leaf[0] = False

    # depth 1
    for i, (pid, side) in enumerate([(0, 0), (0, 1)]):
        nid = i + 1
        px = nodes[pid][0]
        x = px - 4 + side * 8
        nodes[nid] = (x, depth_y[1])
        labels[nid] = f'feat_2\n≤ {0.35 + side*0.3:.2f}'
        is_leaf[nid] = False

    # depth 2
    d2_parents = [1, 1, 2, 2]
    d2_sides   = [0, 1, 0, 1]
    for i, (pid, side) in enumerate(zip(d2_parents, d2_sides)):
        nid = i + 3
        px = nodes[pid][0]
        x = px - 2 + side * 4
        nodes[nid] = (x, depth_y[2])
        labels[nid] = f'feat_3\n≤ {0.42 + (i%3)*0.15:.2f}'
        is_leaf[nid] = False

    # depth 3
    d3_parents = [3, 3, 4, 4, 5, 5, 6, 6]
    d3_sides   = [0, 1, 0, 1, 0, 1, 0, 1]
    for i, (pid, side) in enumerate(zip(d3_parents, d3_sides)):
        nid = i + 7
        px = nodes[pid][0]
        x = px - 1 + side * 2
        nodes[nid] = (x, depth_y[3])
        labels[nid] = f'feat_4\n≤ {0.28 + i*0.06:.2f}'
        is_leaf[nid] = False

    # depth 4 — leaves
    d4_parents = list(range(7, 15))
    li = 0
    for i, (pid, side) in enumerate([(p, s) for p in d4_parents for s in [0, 1]]):
        nid = 15 + i
        px = nodes[pid][0]
        x = px - 0.5 + side * 1
        nodes[nid] = (x, depth_y[4])
        labels[nid] = f'C{li}'
        is_leaf[nid] = True
        leaf_idx[nid] = li
        li += 1

    # draw edges
    edge_labels = ['≤', '>']
    drawn_edges = set()
    for nid in range(1, 15 + 16):
        if nid == 0:
            continue
        # find parent
        # nodes 1,2 -> parent 0
        # nodes 3-6 -> parents 1,2
        # nodes 7-14 -> parents 3-6
        # nodes 15-30 -> parents 7-14
        if 1 <= nid <= 2:
            pid = 0
        elif 3 <= nid <= 6:
            pid = (nid - 3) // 2 + 1
        elif 7 <= nid <= 14:
            pid = (nid - 7) // 2 + 3
        elif 15 <= nid <= 30:
            pid = (nid - 15) // 2 + 7
        else:
            continue

        if (pid, nid) in drawn_edges:
            continue
        drawn_edges.add((pid, nid))

        px, py = nodes[pid]
        cx, cy = nodes[nid]
        side = (nid - 1) % 2  # 0=left(≤), 1=right(>)
        col = accent if side == 0 else '#FF8A80'
        ax.annotate('', xy=(cx, cy + 0.3), xytext=(px, py - 0.3),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
        mid_x = (px + cx) / 2 + (0.15 if side else -0.15)
        mid_y = (py + cy) / 2
        ax.text(mid_x, mid_y, edge_labels[side], fontsize=7,
                color=col, ha='center', va='center')

    # draw nodes
    for nid, (x, y) in nodes.items():
        if is_leaf.get(nid):
            li = leaf_idx[nid]
            c = leaf_colors[li]
            rect = FancyBboxPatch((x - 0.45, y - 0.28), 0.9, 0.56,
                                  boxstyle='round,pad=0.05',
                                  facecolor=c, edgecolor='white', linewidth=0.8,
                                  transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x, y, labels[nid], fontsize=7.5, color='white',
                    ha='center', va='center', fontweight='bold')
        else:
            rect = FancyBboxPatch((x - 0.55, y - 0.32), 1.1, 0.64,
                                  boxstyle='round,pad=0.05',
                                  facecolor='#2A2A3E', edgecolor=accent, linewidth=1.2,
                                  transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x, y, labels[nid], fontsize=7.5, color=TEXT_WHITE,
                    ha='center', va='center')

    bullets = [
        '• Interpretable splits on features',
        '• Each leaf = one cluster label',
        '• Fast inference (O(depth))',
        '• Handles mixed feature types',
        '• Prone to over-splitting',
        '• Depth controls granularity',
        '• Leaves C0–C15 = 16 clusters',
    ]
    add_bullets(fig, bullets, accent)

    # bullet box background
    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')

    path = os.path.join(OUT_DIR, 'slide_01_decision_tree_clustering.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 2 — K-Means (MiniBatch)
# ══════════════════════════════════════════════════════════════════════════════
def slide_02():
    accent = '#A5D6A7'
    rng = np.random.default_rng(42)

    fig = new_fig()
    add_title(fig, 'Method 2 — K-Means (MiniBatch)', accent)

    ax = fig.add_axes([0.04, 0.07, 0.62, 0.82])
    ax.set_facecolor('#12121C')
    ax.tick_params(colors=TEXT_GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    n_clusters = 6
    cluster_colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#26C6DA']

    centers = np.array([
        [1.5, 2.5], [4.5, 4.0], [2.5, 7.0],
        [7.0, 2.0], [8.5, 6.5], [6.0, 8.5],
    ])
    all_pts = []
    all_labels = []
    for k, (cx, cy) in enumerate(centers):
        n = 100
        pts = rng.normal([cx, cy], 0.6, (n, 2))
        all_pts.append(pts)
        all_labels.extend([k] * n)
    all_pts = np.vstack(all_pts)

    # scatter
    for k in range(n_clusters):
        mask = np.array(all_labels) == k
        ax.scatter(all_pts[mask, 0], all_pts[mask, 1],
                   color=cluster_colors[k], s=18, alpha=0.75, zorder=3)

    # Voronoi
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(centers)
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue
        v0, v1 = vor.vertices[ridge_vertices[0]], vor.vertices[ridge_vertices[1]]
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color='#FFFFFF', lw=1.0,
                alpha=0.35, zorder=2, linestyle='--')

    # centroids
    for k, (cx, cy) in enumerate(centers):
        ax.scatter(cx, cy, marker='*', s=350, color='white',
                   edgecolors=cluster_colors[k], linewidths=1.5, zorder=5)
        ax.text(cx + 0.15, cy + 0.25, f'C{k}', color=cluster_colors[k],
                fontsize=9, fontweight='bold', zorder=6)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel('Feature 1', color=TEXT_GRAY, fontsize=9)
    ax.set_ylabel('Feature 2', color=TEXT_GRAY, fontsize=9)

    # MiniBatch badge
    badge = FancyBboxPatch((6.8, 9.0), 2.8, 0.8,
                           boxstyle='round,pad=0.1',
                           facecolor=accent, edgecolor='white', linewidth=1,
                           transform=ax.transData, zorder=7)
    ax.add_patch(badge)
    ax.text(8.2, 9.4, 'MiniBatch', color='#1E1E2E', fontsize=10,
            fontweight='bold', ha='center', va='center', zorder=8)

    # legend
    handles = [mpatches.Patch(color=cluster_colors[k], label=f'Cluster {k}')
               for k in range(n_clusters)]
    ax.legend(handles=handles, loc='lower left', fontsize=8,
              facecolor='#252535', edgecolor=accent, labelcolor=TEXT_GRAY)

    bullets = [
        '• Mini-batches speed up fit',
        '• 6 Gaussian blobs shown',
        '• ★ = cluster centroids',
        '• Voronoi = decision boundary',
        '• Euclidean distance metric',
        '• Scalable to millions of pts',
        '• Hard assignment (1 label/pt)',
    ]
    add_bullets(fig, bullets, accent)

    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')

    path = os.path.join(OUT_DIR, 'slide_02_kmeans_minibatch.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 3 — Autoencoder + K-Means
# ══════════════════════════════════════════════════════════════════════════════
def slide_03():
    accent = '#CE93D8'
    fig = new_fig()
    add_title(fig, 'Method 3 — Autoencoder + K-Means', accent)

    ax = fig.add_axes([0.02, 0.08, 0.66, 0.80])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # pipeline blocks
    blocks = [
        (0.25, 1.5, 1.2, 1.0, 'Input\n126D',          '#37474F', TEXT_WHITE, False),
        (1.85, 1.2, 1.3, 1.6, 'Encoder\n128→64\n→32→16', '#4A148C', accent,   True),
        (3.45, 1.5, 1.2, 1.0, 'Latent\n16D',           '#6A1B9A', accent,     False),
        (4.95, 1.2, 1.5, 1.6, 'K-Means\nClustering',  '#1B5E20', '#A5D6A7',  True),
        (6.75, 1.5, 1.5, 1.0, 'Cluster\nLabels',       '#1A237E', '#90CAF9',  False),
    ]

    block_centers = []
    for (bx, by, bw, bh, label, fc, tc, has_detail) in blocks:
        rect = FancyBboxPatch((bx, by), bw, bh,
                              boxstyle='round,pad=0.08',
                              facecolor=fc, edgecolor=accent, linewidth=1.5,
                              transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        cx = bx + bw / 2
        cy = by + bh / 2
        block_centers.append((cx, cy, bx + bw))
        ax.text(cx, cy, label, color=tc, fontsize=9.5, fontweight='bold',
                ha='center', va='center', zorder=4)

        if has_detail and label.startswith('Encoder'):
            # sub-labels
            layer_info = ['BN+LReLU', 'BN+LReLU', 'BN+LReLU']
            for li, info in enumerate(layer_info):
                yi = by + bh * (0.82 - li * 0.25)
                ax.text(cx, yi, info, color='#B39DDB', fontsize=6.5,
                        ha='center', va='center', zorder=5,
                        style='italic')

        if has_detail and label.startswith('K-Means'):
            ax.text(cx, by + 0.18, 'k clusters', color='#A5D6A7',
                    fontsize=7, ha='center', va='bottom', zorder=5, style='italic')

    # arrows between blocks
    arrow_props = dict(arrowstyle='->', color=accent, lw=2.0,
                       connectionstyle='arc3,rad=0')
    for i in range(len(block_centers) - 1):
        x0 = block_centers[i][2]
        x1 = blocks[i + 1][0]
        y0 = block_centers[i][1]
        y1 = block_centers[i + 1][1]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrow_props, zorder=5)

    # dimension annotations on arrows
    dim_labels = ['126D', '16D', '16D', 'labels']
    xs_mid = [(block_centers[i][2] + blocks[i+1][0]) / 2
              for i in range(len(block_centers)-1)]
    for i, (xm, dl) in enumerate(zip(xs_mid, dim_labels)):
        ax.text(xm, block_centers[i][1] + 0.28, dl,
                color='#FFD54F', fontsize=8, ha='center', va='bottom',
                fontweight='bold', zorder=6)

    # title for diagram
    ax.text(5.0, 3.65, 'End-to-End Pipeline', color=TEXT_GRAY,
            fontsize=11, ha='center', va='top', style='italic')

    bullets = [
        '• 126D → 16D latent space',
        '• Encoder: BatchNorm + LeakyReLU',
        '• Dimensionality reduction first',
        '• K-Means on latent vectors',
        '• Unsupervised representation',
        '• GPU-accelerated training',
        '• Captures nonlinear structure',
    ]
    add_bullets(fig, bullets, accent)

    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')

    path = os.path.join(OUT_DIR, 'slide_03_autoencoder_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 4 — GMM (Gaussian Mixture Model)
# ══════════════════════════════════════════════════════════════════════════════
def slide_04():
    accent = '#FFCC80'
    rng = np.random.default_rng(7)
    fig = new_fig()
    add_title(fig, 'Method 4 — GMM  (soft-assignment, full covariance)', accent)

    ax = fig.add_axes([0.04, 0.07, 0.62, 0.82])
    ax.set_facecolor('#12121C')
    ax.tick_params(colors=TEXT_GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    cluster_colors = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC']
    centers = np.array([[2, 3], [6, 2], [4, 7], [8, 7], [2, 8]])
    covs = [
        np.array([[0.6, 0.3], [0.3, 0.4]]),
        np.array([[0.8, -0.4], [-0.4, 0.5]]),
        np.array([[0.5, 0.2], [0.2, 0.7]]),
        np.array([[0.9, 0.1], [0.1, 0.4]]),
        np.array([[0.4, -0.2], [-0.2, 0.6]]),
    ]

    from matplotlib.patches import Ellipse

    for k, (mu, cov, col) in enumerate(zip(centers, covs, cluster_colors)):
        # scatter
        pts = rng.multivariate_normal(mu, cov * 0.4, 80)
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=14, alpha=0.6, zorder=3)

        # 1σ and 2σ ellipses
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        for n_std, alpha_e, lw in [(1, 0.7, 1.8), (2, 0.35, 1.2)]:
            w, h = 2 * n_std * np.sqrt(vals)
            ell = Ellipse(xy=mu, width=w, height=h, angle=angle,
                          edgecolor=col, facecolor=col,
                          alpha=alpha_e * 0.25 if n_std == 2 else 0,
                          linewidth=lw, linestyle='-' if n_std == 1 else '--',
                          zorder=4)
            ax.add_patch(ell)
            if n_std == 2:
                ell2 = Ellipse(xy=mu, width=w, height=h, angle=angle,
                               edgecolor=col, facecolor='none',
                               linewidth=lw, linestyle='--', zorder=4)
                ax.add_patch(ell2)

        # center marker
        ax.scatter(*mu, marker='+', s=200, color='white', linewidths=2, zorder=6)
        ax.text(mu[0] + 0.15, mu[1] + 0.2, f'μ{k}', color=col,
                fontsize=9, fontweight='bold', zorder=7)

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_xlabel('Feature 1', color=TEXT_GRAY, fontsize=9)
    ax.set_ylabel('Feature 2', color=TEXT_GRAY, fontsize=9)

    # 1σ / 2σ legend
    from matplotlib.lines import Line2D
    leg_handles = [
        Line2D([0], [0], color='white', lw=1.8, label='1σ ellipse'),
        Line2D([0], [0], color='white', lw=1.2, linestyle='--', label='2σ ellipse'),
    ]
    ax.legend(handles=leg_handles, loc='lower right', fontsize=8,
              facecolor='#252535', edgecolor=accent, labelcolor=TEXT_GRAY)

    # subsample note
    note_bg = FancyBboxPatch((0.2, 9.4), 4.5, 0.7,
                             boxstyle='round,pad=0.1',
                             facecolor='#2A2A3E', edgecolor=accent, linewidth=1,
                             transform=ax.transData, zorder=7)
    ax.add_patch(note_bg)
    ax.text(2.45, 9.75, '50K subsample fit  →  full predict',
            color=accent, fontsize=9, ha='center', va='center',
            fontweight='bold', zorder=8)

    bullets = [
        '• Probabilistic soft-assignment',
        '• Full covariance matrices',
        '• 1σ and 2σ confidence ellipses',
        '• EM algorithm optimization',
        '• Handles elongated clusters',
        '• 50K subsample for scalability',
        '• BIC/AIC for k selection',
    ]
    add_bullets(fig, bullets, accent)

    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')

    path = os.path.join(OUT_DIR, 'slide_04_gmm.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 5 — Bisecting K-Means
# ══════════════════════════════════════════════════════════════════════════════
def slide_05():
    accent = '#80DEEA'
    fig = new_fig()
    add_title(fig, 'Method 5 — Bisecting K-Means', accent)

    ax = fig.add_axes([0.03, 0.07, 0.64, 0.82])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Levels: 0→1 cluster, 1→2, 2→3(split worst), 3→4 clusters shown as boxes
    # Box colors by final cluster identity
    level_colors = [
        ['#546E7A'],                                      # level 0: 1 cluster
        ['#EF5350', '#42A5F5'],                           # level 1: 2 clusters
        ['#EF5350', '#66BB6A', '#42A5F5'],                # level 2: 3 (split C1 blue)
        ['#EF5350', '#66BB6A', '#FFA726', '#AB47BC'],     # level 3: 4 clusters
    ]

    level_y = [8.0, 6.0, 4.0, 2.0]
    box_w, box_h = 1.6, 0.9

    # For each level, evenly space boxes
    node_positions = {}  # (level, idx) -> (x_center, y_center)

    for lv, cols in enumerate(level_colors):
        n = len(cols)
        total_w = n * box_w + (n - 1) * 0.5
        x_start = (14 - total_w) / 2
        for i, col in enumerate(cols):
            cx = x_start + i * (box_w + 0.5) + box_w / 2
            cy = level_y[lv]
            node_positions[(lv, i)] = (cx, cy)
            rect = FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h,
                                  boxstyle='round,pad=0.07',
                                  facecolor=col, edgecolor='white', linewidth=1.4,
                                  transform=ax.transData, zorder=3)
            ax.add_patch(rect)
            label = f'Cluster {i}'
            ax.text(cx, cy, label, color='white', fontsize=8.5,
                    fontweight='bold', ha='center', va='center', zorder=4)

    # Level labels on left
    level_labels = ['Level 0\n(k=1)', 'Level 1\n(k=2)', 'Level 2\n(k=3)', 'Level 3\n(k=4)']
    for lv, lbl in enumerate(level_labels):
        ax.text(0.5, level_y[lv], lbl, color=TEXT_GRAY, fontsize=8,
                ha='center', va='center')

    # Draw split arrows
    # Level 0 -> Level 1: cluster 0 splits into 0,1
    splits = [
        (0, 0, [(1, 0), (1, 1)], False),         # L0C0 -> L1C0, L1C1
        (1, 1, [(2, 1), (2, 2)], True),           # L1C1 -> L2C1, L2C2 (highest SSE)
        (2, 2, [(3, 2), (3, 3)], True),           # L2C2 -> L3C2, L3C3
    ]

    # pass-through (unchanged) connections
    passthrough = [
        (1, 0, 2, 0),  # L1C0 -> L2C0
        (2, 0, 3, 0),  # L2C0 -> L3C0
        (2, 1, 3, 1),  # L2C1 -> L3C1
    ]

    for (slv, si, targets, is_high_sse) in splits:
        sx, sy = node_positions[(slv, si)]
        col = '#FF5252' if is_high_sse else accent
        for (tlv, ti) in targets:
            tx, ty = node_positions[(tlv, ti)]
            ax.annotate('', xy=(tx, ty + box_h/2 + 0.05),
                        xytext=(sx, sy - box_h/2 - 0.05),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                        connectionstyle='arc3,rad=0.0'),
                        zorder=5)

    for (slv, si, tlv, ti) in passthrough:
        sx, sy = node_positions[(slv, si)]
        tx, ty = node_positions[(tlv, ti)]
        ax.annotate('', xy=(tx, ty + box_h/2 + 0.05),
                    xytext=(sx, sy - box_h/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.2,
                                    connectionstyle='arc3,rad=0.0', linestyle='dashed'),
                    zorder=5)

    # Annotation: "Split highest SSE cluster"
    ann_x, ann_y = node_positions[(1, 1)]
    ax.annotate('Split highest\nSSE cluster',
                xy=(ann_x + box_w/2 + 0.1, ann_y),
                xytext=(ann_x + box_w/2 + 1.8, ann_y + 0.5),
                color='#FF5252', fontsize=8.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF5252', lw=1.4),
                zorder=6)

    # "Unchanged" label
    pt_x, pt_y = node_positions[(1, 0)]
    ax.annotate('Unchanged', xy=(pt_x - box_w/2 - 0.05, pt_y),
                xytext=(pt_x - box_w/2 - 2.0, pt_y - 0.3),
                color='#78909C', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.0),
                zorder=6)

    bullets = [
        '• Divisive (top-down) approach',
        '• Bisects cluster with max SSE',
        '• Reaches target k iteratively',
        '• More balanced than vanilla k-means',
        '• O(k · n) per iteration',
        '• Red arrows = bisection step',
        '• Grey arrows = cluster preserved',
    ]
    add_bullets(fig, bullets, accent)

    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')

    path = os.path.join(OUT_DIR, 'slide_05_bisecting_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating slides...')
    slide_01()
    slide_02()
    slide_03()
    slide_04()
    slide_05()
    print('Done. All 5 slides saved to', OUT_DIR)
