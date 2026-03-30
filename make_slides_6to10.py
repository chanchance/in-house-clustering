"""
Generate matplotlib PNG slides (1920x1080) for clustering methods 6-10.
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
from matplotlib.lines import Line2D

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


def add_key_char_box(fig, accent):
    bbox_ax = fig.add_axes([0.69, 0.05, 0.30, 0.84])
    bbox_ax.set_facecolor('#252535')
    bbox_ax.axis('off')
    fig.text(0.715, 0.875, 'Key Characteristics', fontsize=13,
             color=accent, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 6 — Agglomerative Ward Clustering
# ══════════════════════════════════════════════════════════════════════════════
def slide_06():
    accent = '#EF9A9A'
    rng = np.random.default_rng(42)
    fig = new_fig()
    add_title(fig, 'Method 6 — Agglomerative Ward Clustering', accent)

    ax = fig.add_axes([0.03, 0.07, 0.64, 0.82])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ── Dendrogram-style bottom-up merging ────────────────────────────────────
    # 8 leaf points at the bottom, progressively merged upward in 3 levels
    leaf_colors = ['#EF5350', '#EF5350',
                   '#42A5F5', '#42A5F5',
                   '#66BB6A', '#66BB6A',
                   '#FFA726', '#FFA726']
    cluster_colors_l1 = ['#EF5350', '#42A5F5', '#66BB6A', '#FFA726']
    cluster_colors_l2 = ['#CE93D8', '#FFD54F']
    root_color = accent

    # x positions for 8 leaves, evenly spread
    leaf_xs = np.linspace(1.5, 12.5, 8)
    leaf_y  = 1.2

    # Level 1 merges: pairs (0,1), (2,3), (4,5), (6,7) → 4 nodes
    l1_xs = [(leaf_xs[i] + leaf_xs[i+1]) / 2 for i in range(0, 8, 2)]
    l1_y  = 3.5

    # Level 2 merges: pairs (0,1), (2,3) → 2 nodes
    l2_xs = [(l1_xs[i] + l1_xs[i+1]) / 2 for i in range(0, 4, 2)]
    l2_y  = 6.0

    # Root merge
    root_x = (l2_xs[0] + l2_xs[1]) / 2
    root_y = 8.3

    def draw_merge(ax, x0, y0, x1, y1, x_parent, y_parent, color, lw=2.0):
        # horizontal bar at parent level, two vertical drops
        ax.plot([x0, x_parent], [y_parent, y_parent], color=color, lw=lw, zorder=3)
        ax.plot([x1, x_parent], [y_parent, y_parent], color=color, lw=lw, zorder=3)
        ax.plot([x0, x0], [y0 + 0.25, y_parent], color=color, lw=lw, zorder=3)
        ax.plot([x1, x1], [y0 + 0.25, y_parent], color=color, lw=lw, zorder=3)

    # Draw leaf nodes
    for i, (lx, lc) in enumerate(zip(leaf_xs, leaf_colors)):
        circ = plt.Circle((lx, leaf_y), 0.28, color=lc, zorder=4)
        ax.add_patch(circ)
        ax.text(lx, leaf_y, f'p{i}', color='white', fontsize=7,
                ha='center', va='center', fontweight='bold', zorder=5)

    # Level 1 merges
    for i in range(4):
        x0, x1 = leaf_xs[i*2], leaf_xs[i*2+1]
        xp = l1_xs[i]
        draw_merge(ax, x0, leaf_y, x1, leaf_y, xp, l1_y, cluster_colors_l1[i], lw=2.2)
        # merged node
        rect = FancyBboxPatch((xp - 0.55, l1_y - 0.28), 1.1, 0.56,
                              boxstyle='round,pad=0.06',
                              facecolor=cluster_colors_l1[i], edgecolor='white', linewidth=1.2,
                              transform=ax.transData, zorder=4)
        ax.add_patch(rect)
        ax.text(xp, l1_y, f'M{i}', color='white', fontsize=8.5,
                ha='center', va='center', fontweight='bold', zorder=5)

    # Level 2 merges
    for i in range(2):
        x0, x1 = l1_xs[i*2], l1_xs[i*2+1]
        xp = l2_xs[i]
        draw_merge(ax, x0, l1_y, x1, l1_y, xp, l2_y, cluster_colors_l2[i], lw=2.5)
        rect = FancyBboxPatch((xp - 0.65, l2_y - 0.32), 1.3, 0.64,
                              boxstyle='round,pad=0.07',
                              facecolor=cluster_colors_l2[i], edgecolor='white', linewidth=1.5,
                              transform=ax.transData, zorder=4)
        ax.add_patch(rect)
        ax.text(xp, l2_y, f'G{i}', color='#1E1E2E', fontsize=9,
                ha='center', va='center', fontweight='bold', zorder=5)

    # Root merge
    draw_merge(ax, l2_xs[0], l2_y, l2_xs[1], l2_y, root_x, root_y, root_color, lw=3.0)
    root_rect = FancyBboxPatch((root_x - 0.8, root_y - 0.38), 1.6, 0.76,
                               boxstyle='round,pad=0.08',
                               facecolor=accent, edgecolor='white', linewidth=2.0,
                               transform=ax.transData, zorder=4)
    ax.add_patch(root_rect)
    ax.text(root_x, root_y, 'ROOT', color='#1E1E2E', fontsize=11,
            ha='center', va='center', fontweight='bold', zorder=5)

    # Level labels
    for y, lbl in [(leaf_y, 'N points'), (l1_y, 'Merge level 1'), (l2_y, 'Merge level 2'), (root_y, 'Root')]:
        ax.text(0.2, y, lbl, color=TEXT_GRAY, fontsize=8, ha='left', va='center', style='italic')

    # "30K subsample + KNN assign" annotation banner
    banner = FancyBboxPatch((1.2, 9.2), 10.5, 0.65,
                            boxstyle='round,pad=0.08',
                            facecolor='#2A2A3E', edgecolor=accent, linewidth=1.5,
                            transform=ax.transData, zorder=6)
    ax.add_patch(banner)
    ax.text(6.45, 9.525, '30K subsample fit  →  full KNN assign',
            color=accent, fontsize=10, ha='center', va='center',
            fontweight='bold', zorder=7)

    bullets = [
        '• Bottom-up (agglomerative) merge',
        '• Ward linkage: minimize SSE',
        '• 3 merge levels shown',
        '• Dendrogram guides cut point',
        '• 30K subsample for scalability',
        '• KNN assigns remaining points',
        '• No centroid assumption needed',
    ]
    add_bullets(fig, bullets, accent)
    add_key_char_box(fig, accent)

    path = os.path.join(OUT_DIR, 'slide_06_agglomerative_ward.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 7 — HDBSCAN
# ══════════════════════════════════════════════════════════════════════════════
def slide_07():
    accent = '#B0BEC5'
    rng = np.random.default_rng(17)
    fig = new_fig()
    add_title(fig, 'Method 7 — HDBSCAN', accent)

    ax = fig.add_axes([0.04, 0.07, 0.62, 0.82])
    ax.set_facecolor('#12121C')
    ax.tick_params(colors=TEXT_GRAY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    cluster_specs = [
        ([2.5, 3.0], [[0.5, 0.1], [0.1, 0.4]], '#42A5F5', 70, 'A'),
        ([7.0, 2.5], [[0.4, -0.15], [-0.15, 0.5]], '#66BB6A', 80, 'B'),
        ([5.0, 7.5], [[0.6, 0.2], [0.2, 0.35]], '#FFA726', 60, 'C'),
    ]

    # Dense cluster points
    for (mu, cov, col, n, lbl) in cluster_specs:
        pts = rng.multivariate_normal(mu, np.array(cov) * 0.3, n)
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=20, alpha=0.8, zorder=3)

    # Noise points
    noise_x = rng.uniform(0.5, 9.5, 22)
    noise_y = rng.uniform(0.5, 9.5, 22)
    ax.scatter(noise_x, noise_y, marker='x', color='#78909C', s=45,
               linewidths=1.5, zorder=4, label='Noise')

    # Mutual reachability distance rings around dense regions
    from matplotlib.patches import Ellipse
    mrd_specs = [
        ([2.5, 3.0], 1.4, 1.1, '#42A5F5'),
        ([7.0, 2.5], 1.2, 1.5, '#66BB6A'),
        ([5.0, 7.5], 1.6, 1.0, '#FFA726'),
    ]
    for (mu, rw, rh, col) in mrd_specs:
        ell = Ellipse(xy=mu, width=rw*2, height=rh*2,
                      edgecolor=col, facecolor=col, alpha=0.07,
                      linewidth=2.0, linestyle='--', zorder=2)
        ax.add_patch(ell)
        ell2 = Ellipse(xy=mu, width=rw*2, height=rh*2,
                       edgecolor=col, facecolor='none',
                       linewidth=2.0, linestyle='--', zorder=2)
        ax.add_patch(ell2)

    # MRD label
    ax.text(2.5, 3.0 + 1.1 + 0.15, 'mutual reach.\ndistance', color='#42A5F5',
            fontsize=7.5, ha='center', va='bottom', style='italic', zorder=5)

    # "noise → KNN reassign" dotted arrows from noise points to nearest cluster
    reassign_pairs = [
        ((1.0, 6.5), (2.5, 3.0)),
        ((8.8, 8.0), (5.0, 7.5)),
        ((0.8, 1.5), (2.5, 3.0)),
    ]
    for (nx, ny), (cx, cy) in reassign_pairs:
        ax.annotate('', xy=(cx, cy), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.2,
                                    linestyle='dotted',
                                    connectionstyle='arc3,rad=0.2'),
                    zorder=5)
    ax.text(0.3, 5.2, 'noise →\nKNN\nreassign', color='#78909C',
            fontsize=8, ha='left', va='center', style='italic', zorder=6)

    # Cluster labels
    for (mu, _, _, col, lbl) in cluster_specs:
        ax.text(mu[0], mu[1], lbl, color='white', fontsize=12,
                ha='center', va='center', fontweight='bold', zorder=6)

    # "Auto cluster count" badge
    badge = FancyBboxPatch((6.5, 8.5), 2.8, 0.8,
                           boxstyle='round,pad=0.1',
                           facecolor=accent, edgecolor='white', linewidth=1,
                           transform=ax.transData, zorder=7)
    ax.add_patch(badge)
    ax.text(7.9, 8.9, 'Auto cluster\ncount', color='#1E1E2E', fontsize=9,
            fontweight='bold', ha='center', va='center', zorder=8)

    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel('Feature 1', color=TEXT_GRAY, fontsize=9)
    ax.set_ylabel('Feature 2', color=TEXT_GRAY, fontsize=9)

    handles = [
        mpatches.Patch(color='#42A5F5', label='Cluster A'),
        mpatches.Patch(color='#66BB6A', label='Cluster B'),
        mpatches.Patch(color='#FFA726', label='Cluster C'),
        Line2D([0], [0], marker='x', color='#78909C', linestyle='none',
               markersize=8, label='Noise', markeredgewidth=1.5),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=8,
              facecolor='#252535', edgecolor=accent, labelcolor=TEXT_GRAY)

    bullets = [
        '• Density-based: no k needed',
        '• Noise points marked ✕',
        '• Mutual reachability distance',
        '• Hierarchical density tree',
        '• Robust to cluster shape',
        '• Noise → KNN reassignment',
        '• Auto cluster count selection',
    ]
    add_bullets(fig, bullets, accent)
    add_key_char_box(fig, accent)

    path = os.path.join(OUT_DIR, 'slide_07_hdbscan.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 8 — Spectral Clustering
# ══════════════════════════════════════════════════════════════════════════════
def slide_08():
    accent = '#F48FB1'
    rng = np.random.default_rng(7)
    fig = new_fig()
    add_title(fig, 'Method 8 — Spectral Clustering', accent)

    # 3-panel layout inside left 67% of figure
    panel_axes = []
    panel_rects = [
        [0.03, 0.10, 0.19, 0.78],   # Panel 1: similarity graph
        [0.25, 0.10, 0.19, 0.78],   # Panel 2: Laplacian eigenvectors
        [0.47, 0.10, 0.19, 0.78],   # Panel 3: K-Means in spectral space
    ]
    panel_titles = [
        'Similarity\nGraph',
        'Laplacian\nEigenvectors',
        'K-Means in\nSpectral Space',
    ]

    for rect, ptitle in zip(panel_rects, panel_titles):
        pax = fig.add_axes(rect)
        pax.set_facecolor('#12121C')
        for spine in pax.spines.values():
            spine.set_edgecolor('#444466')
        pax.set_title(ptitle, color=accent, fontsize=10, fontweight='bold', pad=6)
        panel_axes.append(pax)

    # ── Panel 1: Similarity graph ──────────────────────────────────────────
    pax = panel_axes[0]
    pax.set_xlim(-1.5, 1.5)
    pax.set_ylim(-1.5, 1.5)
    pax.axis('off')

    node_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    node_colors = ['#EF5350', '#EF5350', '#EF5350',
                   '#42A5F5', '#42A5F5',
                   '#66BB6A', '#66BB6A', '#66BB6A']
    node_xy = np.column_stack([np.cos(node_angles), np.sin(node_angles)])

    # edges within clusters (strong) and between (weak)
    cluster_assign = [0, 0, 0, 1, 1, 2, 2, 2]
    for i in range(8):
        for j in range(i+1, 8):
            same = cluster_assign[i] == cluster_assign[j]
            col = '#AAAAAA' if same else '#333355'
            lw = 1.5 if same else 0.5
            alpha = 0.7 if same else 0.3
            pax.plot([node_xy[i, 0], node_xy[j, 0]],
                     [node_xy[i, 1], node_xy[j, 1]],
                     color=col, lw=lw, alpha=alpha, zorder=2)

    for i, (xy, nc) in enumerate(zip(node_xy, node_colors)):
        circ = plt.Circle(xy, 0.13, color=nc, zorder=4)
        pax.add_patch(circ)

    pax.text(0, -1.45, '8K subsample', color=accent, fontsize=7.5,
             ha='center', va='bottom', style='italic')

    # ── Panel 2: Laplacian eigenvectors heatmap ───────────────────────────
    pax = panel_axes[1]
    pax.tick_params(colors=TEXT_GRAY, labelsize=7)
    for spine in pax.spines.values():
        spine.set_edgecolor('#444466')

    # Synthetic eigenvector matrix: 8 nodes x 3 eigenvectors, sorted by cluster
    ev_data = np.zeros((8, 3))
    ev_data[0:3, 0] = 0.57 + rng.normal(0, 0.03, 3)
    ev_data[3:5, 1] = 0.71 + rng.normal(0, 0.03, 2)
    ev_data[5:8, 2] = 0.58 + rng.normal(0, 0.03, 3)
    ev_data += rng.normal(0, 0.04, ev_data.shape)

    im = pax.imshow(ev_data, aspect='auto', cmap='RdBu_r',
                    vmin=-0.9, vmax=0.9, interpolation='nearest')
    pax.set_xticks([0, 1, 2])
    pax.set_xticklabels(['e₁', 'e₂', 'e₃'], color=TEXT_GRAY, fontsize=8)
    pax.set_yticks(range(8))
    pax.set_yticklabels([f'n{i}' for i in range(8)], color=TEXT_GRAY, fontsize=7)
    fig.colorbar(im, ax=pax, fraction=0.046, pad=0.04,
                 label='eigenvector value').ax.yaxis.label.set_color(TEXT_GRAY)

    # ── Panel 3: K-Means in spectral space ───────────────────────────────
    pax = panel_axes[2]
    pax.set_facecolor('#12121C')
    pax.tick_params(colors=TEXT_GRAY, labelsize=7)
    for spine in pax.spines.values():
        spine.set_edgecolor('#444466')

    centers_sp = np.array([[0.57, 0.05], [0.05, 0.71], [0.05, 0.05]])
    sp_colors = ['#EF5350', '#42A5F5', '#66BB6A']
    for k, (ctr, col) in enumerate(zip(centers_sp, sp_colors)):
        pts = rng.normal(ctr, 0.06, (25, 2))
        pts = np.clip(pts, -0.1, 0.85)
        pax.scatter(pts[:, 0], pts[:, 1], color=col, s=16, alpha=0.75, zorder=3)
        pax.scatter(*ctr, marker='*', s=200, color='white',
                    edgecolors=col, linewidths=1.2, zorder=5)

    pax.set_xlabel('e₁', color=TEXT_GRAY, fontsize=8)
    pax.set_ylabel('e₂', color=TEXT_GRAY, fontsize=8)

    # Arrows between panels
    arrow_kw = dict(arrowstyle='->', color=accent, lw=2.5,
                    transform=fig.transFigure, clip_on=False)
    for x0, x1 in [(0.222, 0.248), (0.442, 0.468)]:
        fig.add_artist(FancyArrowPatch((x0, 0.495), (x1, 0.495), **arrow_kw))

    # "8K subsample + KNN assign" note under panels
    fig.text(0.33, 0.055, '8K subsample fit  →  full KNN assign',
             color=accent, fontsize=10, ha='center', va='bottom',
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#2A2A3E',
                       edgecolor=accent, linewidth=1.5))

    bullets = [
        '• Graph-based: captures manifold',
        '• Similarity matrix → Laplacian',
        '• Top-k eigenvectors as features',
        '• K-Means on spectral embedding',
        '• Detects non-convex clusters',
        '• 8K subsample for scalability',
        '• KNN assigns remaining points',
    ]
    add_bullets(fig, bullets, accent)
    add_key_char_box(fig, accent)

    path = os.path.join(OUT_DIR, 'slide_08_spectral_clustering.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 9 — IsolationForest + K-Means
# ══════════════════════════════════════════════════════════════════════════════
def slide_09():
    accent = '#FFAB91'
    rng = np.random.default_rng(99)
    fig = new_fig()
    add_title(fig, 'Method 9 — IsolationForest + K-Means', accent)

    ax = fig.add_axes([0.03, 0.08, 0.64, 0.80])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # ── Stage 1 box ──────────────────────────────────────────────────────────
    s1_rect = FancyBboxPatch((0.3, 0.5), 5.8, 8.0,
                             boxstyle='round,pad=0.12',
                             facecolor='#1A1A2A', edgecolor=accent, linewidth=2.0,
                             transform=ax.transData, zorder=2)
    ax.add_patch(s1_rect)
    ax.text(3.2, 8.25, 'Stage 1 — IsolationForest', color=accent,
            fontsize=11, fontweight='bold', ha='center', va='center', zorder=3)

    # Scatter with anomaly score coloring
    n_inlier = 120
    n_outlier = 18
    inlier_centers = np.array([[2.0, 3.5], [3.5, 5.5], [4.5, 2.5]])
    inlier_pts = np.vstack([rng.normal(c, 0.35, (40, 2)) for c in inlier_centers])
    outlier_pts = rng.uniform([0.5, 0.8], [5.8, 7.5], (n_outlier, 2))

    # score: inliers ~ 0.6-0.95 (normal), outliers ~ 0.0-0.3 (anomalous)
    inlier_scores = rng.uniform(0.55, 0.95, n_inlier)
    outlier_scores = rng.uniform(0.02, 0.28, n_outlier)

    cmap = plt.cm.RdYlGn
    sc = ax.scatter(inlier_pts[:, 0] * 0.7 + 0.55,
                    inlier_pts[:, 1] * 0.7 + 0.6,
                    c=inlier_scores, cmap=cmap, vmin=0, vmax=1,
                    s=22, alpha=0.85, zorder=4)
    ax.scatter(outlier_pts[:, 0] * 0.7 + 0.55,
               outlier_pts[:, 1] * 0.7 + 0.6,
               c=outlier_scores, cmap=cmap, vmin=0, vmax=1,
               s=22, alpha=0.85, zorder=4)

    # Mark outliers with red X
    for op in outlier_pts:
        ox = op[0] * 0.7 + 0.55
        oy = op[1] * 0.7 + 0.6
        ax.plot(ox, oy, 'x', color='#EF5350', markersize=9,
                markeredgewidth=2.0, zorder=5)

    ax.text(1.4, 0.75, 'anomaly score heatmap', color=TEXT_GRAY,
            fontsize=7.5, ha='left', va='bottom', style='italic', zorder=5)
    ax.text(1.4, 0.52, '✕ = outlier (isolated early)', color='#EF5350',
            fontsize=7.5, ha='left', va='bottom', zorder=5)

    # ── Stage 2 box ──────────────────────────────────────────────────────────
    s2_rect = FancyBboxPatch((8.0, 0.5), 5.7, 8.0,
                             boxstyle='round,pad=0.12',
                             facecolor='#1A1A2A', edgecolor='#A5D6A7', linewidth=2.0,
                             transform=ax.transData, zorder=2)
    ax.add_patch(s2_rect)
    ax.text(10.85, 8.25, 'Stage 2 — K-Means on Inliers', color='#A5D6A7',
            fontsize=11, fontweight='bold', ha='center', va='center', zorder=3)

    # Inlier clusters in stage 2
    cl_centers2 = np.array([[9.5, 4.0], [11.0, 6.5], [12.5, 3.0]])
    cl_colors2 = ['#42A5F5', '#FFA726', '#AB47BC']
    for k, (cc, col) in enumerate(zip(cl_centers2, cl_colors2)):
        pts2 = rng.normal(cc, 0.5, (40, 2))
        ax.scatter(pts2[:, 0], pts2[:, 1], color=col, s=20, alpha=0.8, zorder=4)
        ax.scatter(*cc, marker='*', s=250, color='white',
                   edgecolors=col, linewidths=1.5, zorder=5)
        ax.text(cc[0] + 0.15, cc[1] + 0.35, f'C{k}', color=col,
                fontsize=9, fontweight='bold', zorder=6)

    # Outlier X points in stage 2 space
    for op in outlier_pts[:8]:
        ox2 = op[0] * 0.55 + 8.2
        oy2 = op[1] * 0.55 + 0.8
        ax.plot(ox2, oy2, 'x', color='#EF5350', markersize=8,
                markeredgewidth=1.8, zorder=5)
    ax.text(9.0, 0.75, '✕ outliers excluded', color='#EF5350',
            fontsize=7.5, ha='left', va='bottom', zorder=5)

    # ── Arrow between stages ─────────────────────────────────────────────────
    ax.annotate('', xy=(8.0, 4.5), xytext=(6.1, 4.5),
                arrowprops=dict(arrowstyle='->', color=accent, lw=3.0,
                                connectionstyle='arc3,rad=0'),
                zorder=6)
    ax.text(7.05, 4.85, 'inliers\nonly', color=accent, fontsize=9,
            ha='center', va='bottom', fontweight='bold', zorder=7)

    bullets = [
        '• Stage 1: isolate anomalies',
        '• Anomaly score heatmap scatter',
        '• Outliers ✕ excluded from fit',
        '• Stage 2: K-Means on inliers',
        '• Cleaner cluster boundaries',
        '• Robust to contamination',
        '• Outliers labeled separately',
    ]
    add_bullets(fig, bullets, accent)
    add_key_char_box(fig, accent)

    path = os.path.join(OUT_DIR, 'slide_09_isolation_forest_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Slide 10 — VAE + K-Means
# ══════════════════════════════════════════════════════════════════════════════
def slide_10():
    accent = '#C5E1A5'
    rng = np.random.default_rng(55)
    fig = new_fig()
    add_title(fig, 'Method 10 — VAE + K-Means', accent)

    ax = fig.add_axes([0.02, 0.06, 0.66, 0.84])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    def draw_block(ax, bx, by, bw, bh, label, fc, ec, tc, fontsize=9.5):
        rect = FancyBboxPatch((bx, by), bw, bh,
                              boxstyle='round,pad=0.08',
                              facecolor=fc, edgecolor=ec, linewidth=1.8,
                              transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        ax.text(bx + bw/2, by + bh/2, label, color=tc, fontsize=fontsize,
                fontweight='bold', ha='center', va='center', zorder=4)
        return bx + bw/2, by + bh/2

    def draw_arrow(ax, x0, y0, x1, y1, col, label=None, rad=0.0, lw=2.0, style='->'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=5)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my + 0.22, label, color=col, fontsize=7.5,
                    ha='center', va='bottom', zorder=6)

    # ── VAE pipeline row (y=5.5 to 7.5) ──────────────────────────────────────
    vae_y = 5.8
    vae_h = 1.5

    # Input block
    draw_block(ax, 0.2, vae_y, 1.5, vae_h, 'Input\n126D', '#37474F', accent, TEXT_WHITE)
    # Encoder block
    draw_block(ax, 2.1, vae_y, 2.2, vae_h, 'Encoder\nFC layers', '#4A148C', accent, TEXT_WHITE)
    # μ and logσ² blocks stacked
    draw_block(ax, 4.7, vae_y + 0.6, 1.4, 0.65, 'μ', '#1565C0', '#90CAF9', TEXT_WHITE, fontsize=11)
    draw_block(ax, 4.7, vae_y,       1.4, 0.55, 'log σ²', '#1565C0', '#90CAF9', TEXT_WHITE, fontsize=9)
    # Reparameterize block
    draw_block(ax, 6.5, vae_y, 1.8, vae_h, 'Reparam.\nz = μ + σε', '#006064', '#80DEEA', TEXT_WHITE)
    # Decoder block
    draw_block(ax, 8.7, vae_y, 2.2, vae_h, 'Decoder\nFC layers', '#4A148C', accent, TEXT_WHITE)
    # Reconstruction block
    draw_block(ax, 11.3, vae_y, 2.4, vae_h, 'Reconstruction\n126D', '#37474F', accent, TEXT_WHITE)

    # Arrows in VAE pipeline
    draw_arrow(ax, 1.7, vae_y+vae_h/2, 2.1, vae_y+vae_h/2, accent, '126D')
    draw_arrow(ax, 4.3, vae_y+vae_h/2, 4.7, vae_y+0.875, accent)
    draw_arrow(ax, 4.3, vae_y+vae_h/2, 4.7, vae_y+0.275, accent)
    draw_arrow(ax, 6.1, vae_y+0.875,   6.5, vae_y+vae_h/2, '#90CAF9')
    draw_arrow(ax, 6.1, vae_y+0.275,   6.5, vae_y+vae_h/2, '#90CAF9')
    draw_arrow(ax, 8.3, vae_y+vae_h/2, 8.7, vae_y+vae_h/2, accent, 'z')
    draw_arrow(ax, 10.9, vae_y+vae_h/2, 11.3, vae_y+vae_h/2, accent)

    # Loss labels
    # KL-divergence loss bracket
    ax.annotate('', xy=(4.7, vae_y - 0.25), xytext=(8.3, vae_y - 0.25),
                arrowprops=dict(arrowstyle='<->', color='#FFD54F', lw=1.5),
                zorder=5)
    ax.text(6.5, vae_y - 0.45, 'KL-Divergence Loss  DKL(q(z|x) ‖ p(z))',
            color='#FFD54F', fontsize=8, ha='center', va='top', zorder=6)

    ax.annotate('', xy=(0.2, vae_y - 0.6), xytext=(13.7, vae_y - 0.6),
                arrowprops=dict(arrowstyle='<->', color='#EF9A9A', lw=1.5),
                zorder=5)
    ax.text(7.0, vae_y - 0.8, 'Reconstruction Loss  ‖x − x̂‖²',
            color='#EF9A9A', fontsize=8, ha='center', va='top', zorder=6)

    # VAE label banner
    vae_banner = FancyBboxPatch((0.0, 8.65), 13.9, 0.65,
                                boxstyle='round,pad=0.08',
                                facecolor='#2A2A3E', edgecolor=accent, linewidth=1.5,
                                transform=ax.transData, zorder=2)
    ax.add_patch(vae_banner)
    ax.text(6.95, 8.975, 'Variational Autoencoder  (β-VAE regularization)',
            color=accent, fontsize=10.5, fontweight='bold', ha='center', va='center', zorder=3)

    # β-VAE badge
    badge = FancyBboxPatch((11.8, 8.68), 1.9, 0.58,
                           boxstyle='round,pad=0.07',
                           facecolor=accent, edgecolor='white', linewidth=1,
                           transform=ax.transData, zorder=6)
    ax.add_patch(badge)
    ax.text(12.75, 8.97, 'β-VAE', color='#1E1E2E', fontsize=10,
            fontweight='bold', ha='center', va='center', zorder=7)

    # ── K-Means row ───────────────────────────────────────────────────────────
    km_y = 2.6
    km_h = 1.5

    # Arrow from z (reparameterize center) down to K-Means
    z_cx = 6.5 + 1.8/2   # center of reparam block
    draw_arrow(ax, z_cx, vae_y, z_cx, km_y + km_h,
               '#80DEEA', 'z latent', lw=2.2)

    draw_block(ax, 4.8, km_y, 3.0, km_h, 'K-Means\nClustering', '#1B5E20', '#A5D6A7', TEXT_WHITE)

    draw_arrow(ax, 4.8 + 3.0, km_y + km_h/2, 10.5, km_y + km_h/2, '#A5D6A7')
    draw_block(ax, 10.5, km_y, 3.0, km_h, 'Cluster\nLabels', '#1A237E', '#90CAF9', TEXT_WHITE)

    # Scatter mini-visualization inside K-Means box
    cl_xy = np.array([[5.5, 3.35], [6.4, 2.9], [7.2, 3.5]])
    cl_c  = ['#EF5350', '#FFA726', '#66BB6A']
    for cxy, cc in zip(cl_xy, cl_c):
        pts_m = rng.normal(cxy, 0.13, (12, 2))
        ax.scatter(pts_m[:, 0], pts_m[:, 1], color=cc, s=10, alpha=0.75, zorder=5)
        ax.scatter(*cxy, marker='*', s=120, color='white',
                   edgecolors=cc, linewidths=1.0, zorder=6)

    bullets = [
        '• VAE learns latent distribution',
        '• μ, logσ² → reparameterize z',
        '• KL-divergence regularization',
        '• Reconstruction loss guides repr.',
        '• β-VAE controls disentanglement',
        '• K-Means on z latent space',
        '• Captures probabilistic structure',
    ]
    add_bullets(fig, bullets, accent)
    add_key_char_box(fig, accent)

    path = os.path.join(OUT_DIR, 'slide_10_vae_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating slides 6-10...')
    slide_06()
    slide_07()
    slide_08()
    slide_09()
    slide_10()
    print('Done. All 5 slides saved to', OUT_DIR)
