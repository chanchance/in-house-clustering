"""
Executive-style white-background clustering method slides (1920x1080).
10 methods, one slide each. Image-focused, clean for senior leadership.
Output: output/slides/exec_slide_{N:02d}_{name}.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi

OUT_DIR = "/Users/jongchan/Desktop/claude/in-house-clustering/output/slides"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_W, FIG_H = 19.2, 10.8
DPI = 100

# ── palette ──────────────────────────────────────────────────────────────────
ACCENT = {
    1:  '#1565C0',   # blue
    2:  '#2E7D32',   # green
    3:  '#6A1B9A',   # purple
    4:  '#E65100',   # orange
    5:  '#00838F',   # teal
    6:  '#C62828',   # red
    7:  '#4527A0',   # deep-purple
    8:  '#1B5E20',   # dark-green
    9:  '#BF360C',   # deep-orange
    10: '#0277BD',   # light-blue
}
CLUSTER_COLORS = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA',
                  '#00ACC1','#F4511E','#6D4C41','#546E7A','#039BE5',
                  '#7CB342','#FFB300','#D81B60','#00897B','#3949AB','#00BCD4']


def new_fig(accent_color, method_num, title):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor('white')

    # top header bar
    header = fig.add_axes([0.0, 0.935, 1.0, 0.065])
    header.set_facecolor(accent_color)
    header.axis('off')
    header.text(0.015, 0.5, f'Method {method_num:02d}', transform=header.transAxes,
                ha='left', va='center', fontsize=15, color='white',
                fontweight='bold', alpha=0.85)
    header.text(0.5, 0.5, title, transform=header.transAxes,
                ha='center', va='center', fontsize=20, color='white',
                fontweight='bold')
    header.text(0.985, 0.5, 'In-House Layout Feature Clustering',
                transform=header.transAxes, ha='right', va='center',
                fontsize=11, color='white', alpha=0.7)
    return fig


def sidebar(fig, accent, bullets):
    """Right sidebar: key points."""
    sb = fig.add_axes([0.705, 0.04, 0.285, 0.875])
    sb.set_facecolor('#F8F9FA')
    for spine in sb.spines.values():
        spine.set_visible(False)
    sb.set_xticks([]); sb.set_yticks([])

    sb.text(0.5, 0.96, 'Algorithm Summary', transform=sb.transAxes,
            ha='center', va='top', fontsize=13, fontweight='bold', color=accent)

    dy = 0.88
    for b in bullets:
        if b.startswith('##'):
            sb.text(0.07, dy, b[2:].strip(), transform=sb.transAxes,
                    ha='left', va='top', fontsize=10.5, color=accent,
                    fontweight='bold')
            dy -= 0.065
        else:
            sb.text(0.07, dy, b, transform=sb.transAxes,
                    ha='left', va='top', fontsize=9.8, color='#333333',
                    linespacing=1.4)
            dy -= 0.072
    return sb


def ax_style(ax, bg='#F7F9FC'):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')
    ax.tick_params(colors='#555555', labelsize=9)
    ax.xaxis.label.set_color('#555555')
    ax.yaxis.label.set_color('#555555')


# ═════════════════════════════════════════════════════════════════════════════
# 01 — Decision Tree Clustering
# ═════════════════════════════════════════════════════════════════════════════
def slide_01():
    ac = ACCENT[1]
    fig = new_fig(ac, 1, 'Decision Tree Clustering')

    ax = fig.add_axes([0.01, 0.04, 0.685, 0.88])
    ax.set_facecolor('white')
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis('off')

    leaf_colors = plt.cm.tab20(np.linspace(0, 1, 16))
    depth_y = {0: 9.2, 1: 7.5, 2: 5.8, 3: 4.1, 4: 2.0}

    nodes = {0: (8, depth_y[0])}
    labels = {0: 'feat_1\n≤ 0.71'}
    is_leaf = {0: False}
    leaf_idx = {}

    for i, side in enumerate([0, 1]):
        nid = i + 1
        nodes[nid] = (nodes[0][0] - 4 + side * 8, depth_y[1])
        labels[nid] = f'feat_2\n≤ {0.35 + side*0.3:.2f}'
        is_leaf[nid] = False

    for i, (pid, side) in enumerate([(1,0),(1,1),(2,0),(2,1)]):
        nid = i + 3
        px = nodes[pid][0]
        nodes[nid] = (px - 2 + side*4, depth_y[2])
        labels[nid] = f'feat_3\n≤ {0.42+(i%3)*0.15:.2f}'
        is_leaf[nid] = False

    for i, (pid, side) in enumerate([(p,s) for p in range(3,7) for s in [0,1]]):
        nid = i + 7
        px = nodes[pid][0]
        nodes[nid] = (px - 1 + side*2, depth_y[3])
        labels[nid] = f'feat_4\n≤ {0.28+i*0.06:.2f}'
        is_leaf[nid] = False

    li = 0
    for i, (pid, side) in enumerate([(p,s) for p in range(7,15) for s in [0,1]]):
        nid = 15 + i
        px = nodes[pid][0]
        nodes[nid] = (px - 0.5 + side, depth_y[4])
        labels[nid] = f'C{li}'
        is_leaf[nid] = True
        leaf_idx[nid] = li
        li += 1

    for nid in range(1, 31):
        if   1 <= nid <= 2:  pid = 0
        elif 3 <= nid <= 6:  pid = (nid-3)//2 + 1
        elif 7 <= nid <= 14: pid = (nid-7)//2 + 3
        elif 15<= nid <= 30: pid = (nid-15)//2 + 7
        else: continue
        px, py = nodes[pid]; cx, cy = nodes[nid]
        side = (nid-1) % 2
        col = ac if side == 0 else '#E53935'
        ax.annotate('', xy=(cx, cy+0.3), xytext=(px, py-0.3),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
        mx = (px+cx)/2 + (0.15 if side else -0.15)
        ax.text(mx, (py+cy)/2, '≤' if side==0 else '>', fontsize=7, color=col, ha='center')

    for nid, (x, y) in nodes.items():
        if is_leaf.get(nid):
            c = leaf_colors[leaf_idx[nid]]
            rect = FancyBboxPatch((x-0.45, y-0.28), 0.9, 0.56,
                                  boxstyle='round,pad=0.05',
                                  facecolor=c, edgecolor='#FFFFFF', linewidth=0.8,
                                  transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x, y, labels[nid], fontsize=7, color='white',
                    ha='center', va='center', fontweight='bold')
        else:
            rect = FancyBboxPatch((x-0.55, y-0.32), 1.1, 0.64,
                                  boxstyle='round,pad=0.05',
                                  facecolor='#EEF2FF', edgecolor=ac, linewidth=1.4,
                                  transform=ax.transData)
            ax.add_patch(rect)
            ax.text(x, y, labels[nid], fontsize=7.5, color='#1A1A2E',
                    ha='center', va='center')

    # legend label
    ax.text(8, 0.5, 'Leaf nodes C0–C15 = 16 clusters',
            ha='center', fontsize=10, color='#555', style='italic')

    sidebar(fig, ac, [
        '## Core Idea',
        'Recursive binary splits on features\nassign each leaf as a cluster.',
        '',
        '## Key Steps',
        '① Grow decision tree on features',
        '② Each terminal leaf → cluster label',
        '③ Depth controls granularity',
        '',
        '## Properties',
        'Interpretable split rules',
        'Fast O(depth) inference',
        'Handles mixed feature types',
        'Prone to over-splitting at depth >5',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_01_decision_tree.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 02 — K-Means MiniBatch
# ═════════════════════════════════════════════════════════════════════════════
def slide_02():
    ac = ACCENT[2]
    fig = new_fig(ac, 2, 'K-Means  (MiniBatch)')
    rng = np.random.default_rng(42)

    ax = fig.add_axes([0.03, 0.06, 0.655, 0.86])
    ax_style(ax)

    n_clusters = 6
    cols = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1']
    centers = np.array([[1.5,2.5],[4.5,4.0],[2.5,7.0],[7.0,2.0],[8.5,6.5],[6.0,8.5]])

    all_pts, all_lbl = [], []
    for k, (cx, cy) in enumerate(centers):
        pts = rng.normal([cx,cy], 0.6, (100,2))
        all_pts.append(pts); all_lbl.extend([k]*100)
    all_pts = np.vstack(all_pts)

    for k in range(n_clusters):
        mask = np.array(all_lbl) == k
        ax.scatter(all_pts[mask,0], all_pts[mask,1],
                   color=cols[k], s=22, alpha=0.7, zorder=3)

    vor = Voronoi(centers)
    for rv in vor.ridge_vertices:
        if -1 in rv: continue
        v0, v1 = vor.vertices[rv[0]], vor.vertices[rv[1]]
        ax.plot([v0[0],v1[0]], [v0[1],v1[1]], color='#999', lw=1.2,
                alpha=0.5, linestyle='--', zorder=2)

    for k, (cx, cy) in enumerate(centers):
        ax.scatter(cx, cy, marker='*', s=400, color='white',
                   edgecolors=cols[k], linewidths=2.0, zorder=6)
        ax.text(cx+0.18, cy+0.3, f'C{k}', color=cols[k], fontsize=10,
                fontweight='bold', zorder=7)

    ax.set_xlim(-0.5, 10.5); ax.set_ylim(0, 11)
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)

    handles = [mpatches.Patch(color=cols[k], label=f'Cluster {k}') for k in range(n_clusters)]
    ax.legend(handles=handles, loc='upper left', fontsize=9,
              facecolor='white', edgecolor='#CCC')

    badge = FancyBboxPatch((7.5, 10.0), 2.5, 0.7, boxstyle='round,pad=0.1',
                           facecolor=ac, edgecolor='white', linewidth=1,
                           transform=ax.transData, zorder=8)
    ax.add_patch(badge)
    ax.text(8.75, 10.35, 'MiniBatch ⚡', color='white', fontsize=10,
            fontweight='bold', ha='center', va='center', zorder=9)

    sidebar(fig, ac, [
        '## Core Idea',
        'Partition data into k spherical\nclusters by minimizing inertia.',
        '',
        '## Algorithm (MiniBatch)',
        '① Init k centroids (k-means++)',
        '② Sample mini-batch each iteration',
        '③ Update centroids incrementally',
        '④ Repeat until convergence',
        '',
        '## Properties',
        'Euclidean distance metric',
        'Hard assignment (1 label/point)',
        'Voronoi = decision boundary',
        '★ = final cluster centroids',
        'Scales to millions of points',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_02_kmeans_minibatch.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 03 — Autoencoder + K-Means
# ═════════════════════════════════════════════════════════════════════════════
def slide_03():
    ac = ACCENT[3]
    fig = new_fig(ac, 3, 'Autoencoder + K-Means')

    ax = fig.add_axes([0.01, 0.08, 0.685, 0.82])
    ax.set_facecolor('white'); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    blocks = [
        (0.2,  1.8, 1.3, 1.4, 'Input\n126D',             '#ECEFF1', '#37474F', False),
        (1.9,  1.5, 1.5, 2.0, 'Encoder\n128→64\n→32→16', '#EDE7F6', ac,       True),
        (3.75, 1.8, 1.3, 1.4, 'Latent\n16D',             '#D1C4E9', ac,       False),
        (5.35, 1.5, 1.6, 2.0, 'K-Means\nk clusters',     '#E8F5E9', '#2E7D32',True),
        (7.25, 1.8, 1.5, 1.4, 'Cluster\nLabels',         '#E3F2FD', '#1565C0',False),
    ]

    centers = []
    for (bx,by,bw,bh,label,fc,tc,_) in blocks:
        rect = FancyBboxPatch((bx,by), bw, bh, boxstyle='round,pad=0.1',
                              facecolor=fc, edgecolor=ac, linewidth=2.0,
                              transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        cx = bx+bw/2; cy = by+bh/2
        centers.append((cx, cy, bx+bw))
        ax.text(cx, cy, label, color=tc, fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=4)

    dim_labels = ['126D', '16D', '16D', 'labels']
    for i in range(len(centers)-1):
        x0 = centers[i][2]; y0 = centers[i][1]
        x1 = blocks[i+1][0]; y1 = centers[i+1][1]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.2), zorder=5)
        mx = (x0+x1)/2
        ax.text(mx, y0+0.3, dim_labels[i], color=ac, fontsize=9,
                ha='center', fontweight='bold')

    # layer details under Encoder
    for i, lyr in enumerate(['BN+LeakyReLU']*3):
        ax.text(centers[1][0], 1.55+(i*0.18), lyr, color='#7E57C2', fontsize=7.5,
                ha='center', style='italic')

    # t-SNE visualization placeholder
    rng = np.random.default_rng(7)
    ax2 = fig.add_axes([0.05, 0.10, 0.30, 0.38])
    ax_style(ax2, '#FAFAFA')
    ax2.set_title('Latent Space (t-SNE)', fontsize=9, color='#444', pad=4)
    for k in range(6):
        c = rng.normal([np.cos(k*np.pi/3)*3, np.sin(k*np.pi/3)*3], 0.7, (60,2))
        ax2.scatter(c[:,0], c[:,1], color=CLUSTER_COLORS[k], s=12, alpha=0.7)
    ax2.set_xticks([]); ax2.set_yticks([])

    sidebar(fig, ac, [
        '## Core Idea',
        'Compress 126D feature vectors\nto 16D latent space, then cluster.',
        '',
        '## Architecture',
        'Input → 128 → 64 → 32 → 16D',
        'BatchNorm + LeakyReLU layers',
        'Unsupervised representation',
        '',
        '## Training',
        '① Pre-train AE on reconstruction loss',
        '② Extract 16D latent vectors',
        '③ Apply K-Means on latent space',
        '',
        '## Properties',
        'Captures nonlinear structure',
        'GPU-accelerated training',
        '126D → 16D (87% reduction)',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_03_autoencoder_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 04 — GMM
# ═════════════════════════════════════════════════════════════════════════════
def slide_04():
    ac = ACCENT[4]
    fig = new_fig(ac, 4, 'Gaussian Mixture Model (GMM)')
    rng = np.random.default_rng(7)

    ax = fig.add_axes([0.03, 0.06, 0.655, 0.86])
    ax_style(ax)

    cols = ['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA']
    centers = np.array([[2,3],[6,2],[4,7],[8,7],[2,8]])
    covs = [
        np.array([[0.6,0.3],[0.3,0.4]]),
        np.array([[0.8,-0.4],[-0.4,0.5]]),
        np.array([[0.5,0.2],[0.2,0.7]]),
        np.array([[0.9,0.1],[0.1,0.4]]),
        np.array([[0.4,-0.2],[-0.2,0.6]]),
    ]

    for k, (mu, cov, col) in enumerate(zip(centers, covs, cols)):
        pts = rng.multivariate_normal(mu, cov*0.4, 80)
        ax.scatter(pts[:,0], pts[:,1], color=col, s=18, alpha=0.65, zorder=3)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:,order]
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        for n_std, lw, ls in [(1,2.0,'-'),(2,1.2,'--')]:
            w, h = 2*n_std*np.sqrt(vals)
            ell = Ellipse(xy=mu, width=w, height=h, angle=angle,
                          edgecolor=col, facecolor=col if n_std==1 else 'none',
                          alpha=0.15 if n_std==1 else 1, linewidth=lw, linestyle=ls, zorder=4)
            ax.add_patch(ell)
        ax.scatter(*mu, marker='+', s=250, color='#333', linewidths=2.0, zorder=6)
        ax.text(mu[0]+0.15, mu[1]+0.25, f'μ{k}\nΣ{k}', color=col, fontsize=8.5,
                fontweight='bold', zorder=7)

    ax.set_xlim(0,10.5); ax.set_ylim(0.5,10.5)
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)

    leg = [Line2D([0],[0], color='#555', lw=2.0, label='1σ ellipse'),
           Line2D([0],[0], color='#555', lw=1.2, ls='--', label='2σ ellipse')]
    ax.legend(handles=leg, fontsize=9, facecolor='white', edgecolor='#CCC')

    note = FancyBboxPatch((0.3,9.4), 5.5, 0.7, boxstyle='round,pad=0.1',
                          facecolor='#FFF3E0', edgecolor=ac, linewidth=1.5,
                          transform=ax.transData, zorder=8)
    ax.add_patch(note)
    ax.text(3.05, 9.75, '50K subsample fit  →  full dataset predict',
            color=ac, fontsize=9.5, ha='center', va='center', fontweight='bold', zorder=9)

    sidebar(fig, ac, [
        '## Core Idea',
        'Model data as mixture of K\nGaussians with full covariance.',
        '',
        '## Algorithm (EM)',
        '① Init means, covariances, weights',
        '② E-step: compute responsibilities',
        '③ M-step: update parameters',
        '④ Iterate until log-likelihood converges',
        '',
        '## Properties',
        'Soft (probabilistic) assignment',
        'Handles elongated/tilted clusters',
        '50K subsample for scalability',
        'BIC/AIC for optimal k selection',
        '+ = cluster mean (μ)',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_04_gmm.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 05 — Bisecting K-Means
# ═════════════════════════════════════════════════════════════════════════════
def slide_05():
    ac = ACCENT[5]
    fig = new_fig(ac, 5, 'Bisecting K-Means')

    ax = fig.add_axes([0.03, 0.06, 0.655, 0.86])
    ax.set_facecolor('white'); ax.set_xlim(0,14); ax.set_ylim(0,10); ax.axis('off')

    level_colors = [
        ['#B0BEC5'],
        ['#EF5350','#1E88E5'],
        ['#EF5350','#43A047','#1E88E5'],
        ['#EF5350','#43A047','#FB8C00','#8E24AA'],
    ]
    level_y = [8.8, 6.5, 4.2, 1.8]
    bw, bh = 1.8, 0.9
    node_pos = {}

    for lv, cols in enumerate(level_colors):
        n = len(cols)
        total = n*bw + (n-1)*0.6
        x0 = (14-total)/2
        for i, col in enumerate(cols):
            cx = x0 + i*(bw+0.6) + bw/2
            cy = level_y[lv]
            node_pos[(lv,i)] = (cx, cy)
            rect = FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
                                  boxstyle='round,pad=0.08',
                                  facecolor=col, edgecolor='white', linewidth=2,
                                  transform=ax.transData, zorder=3)
            ax.add_patch(rect)
            ax.text(cx, cy, f'Cluster {i}', color='white', fontsize=9,
                    fontweight='bold', ha='center', va='center', zorder=4)

    for lv, lbl in enumerate(['k=1','k=2','k=3','k=4']):
        ax.text(0.4, level_y[lv], lbl, color='#555', fontsize=9,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='#F0F0F0', ec='#CCC'))

    # split arrows
    splits = [(0,0,[(1,0),(1,1)],False),(1,1,[(2,1),(2,2)],True),(2,2,[(3,2),(3,3)],True)]
    passthrough = [(1,0,2,0),(2,0,3,0),(2,1,3,1)]

    for (slv,si,targets,high_sse) in splits:
        sx,sy = node_pos[(slv,si)]
        col = '#E53935' if high_sse else '#555'
        for (tlv,ti) in targets:
            tx,ty = node_pos[(tlv,ti)]
            ax.annotate('', xy=(tx, ty+bh/2+0.05), xytext=(sx, sy-bh/2-0.05),
                        arrowprops=dict(arrowstyle='->', color=col, lw=2.0), zorder=5)

    for (slv,si,tlv,ti) in passthrough:
        sx,sy = node_pos[(slv,si)]; tx,ty = node_pos[(tlv,ti)]
        ax.annotate('', xy=(tx, ty+bh/2+0.05), xytext=(sx, sy-bh/2-0.05),
                    arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.4,
                                    linestyle='dashed'), zorder=5)

    # annotation
    ann_x,ann_y = node_pos[(1,1)]
    ax.annotate('Bisect: highest SSE cluster',
                xy=(ann_x+bw/2+0.1, ann_y), xytext=(ann_x+bw/2+2.5, ann_y+0.8),
                color='#E53935', fontsize=9.5, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5), zorder=6)

    sidebar(fig, ac, [
        '## Core Idea',
        'Top-down divisive approach:\nrepeatedly bisect the cluster\nwith the highest SSE.',
        '',
        '## Algorithm',
        '① Start with all data in 1 cluster',
        '② Select cluster with max SSE',
        '③ Apply 2-means to split it',
        '④ Repeat until k clusters reached',
        '',
        '## Properties',
        'More balanced than vanilla K-Means',
        'O(k · n) per iteration',
        'Red arrows = bisection step',
        'Gray arrows = cluster unchanged',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_05_bisecting_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 06 — Agglomerative Ward
# ═════════════════════════════════════════════════════════════════════════════
def slide_06():
    ac = ACCENT[6]
    fig = new_fig(ac, 6, 'Agglomerative Hierarchical (Ward)')

    ax = fig.add_axes([0.03, 0.06, 0.655, 0.86])
    ax_style(ax, '#FAFAFA')

    # Build simple dendrogram manually
    from scipy.cluster.hierarchy import dendrogram, linkage
    rng = np.random.default_rng(42)
    data = np.vstack([
        rng.normal([1,1], 0.3, (10,2)),
        rng.normal([3,1], 0.3, (10,2)),
        rng.normal([2,3], 0.3, (10,2)),
        rng.normal([4,3], 0.3, (10,2)),
    ])
    Z = linkage(data, method='ward')

    leaf_cols = (
        ['#E53935']*10 + ['#1E88E5']*10 + ['#43A047']*10 + ['#FB8C00']*10
    )

    def color_func(leaf):
        return leaf_cols[leaf]

    dn = dendrogram(Z, ax=ax, color_threshold=Z[-3,2],
                    above_threshold_color='#9E9E9E',
                    leaf_rotation=0, leaf_font_size=0,
                    no_labels=True)

    # color the leaves
    for i, (xi, yi) in enumerate(zip(dn['icoord'], dn['dcoord'])):
        pass  # already colored by dendrogram

    # horizontal cut line
    cut_h = Z[-3, 2] * 1.02
    ax.axhline(cut_h, color=ac, lw=2.5, ls='--', zorder=5)
    ax.text(ax.get_xlim()[1]*0.02, cut_h*1.03, f'Cut → 4 clusters',
            color=ac, fontsize=10, fontweight='bold')

    ax.set_xlabel('Sample index', fontsize=10)
    ax.set_ylabel('Ward linkage distance', fontsize=10)
    ax.set_title('Ward Dendrogram  (40 samples, 4 true clusters)',
                 fontsize=11, color='#444', pad=6)

    # cluster scatter inset
    ax2 = fig.add_axes([0.38, 0.12, 0.30, 0.38])
    ax_style(ax2)
    ax2.set_title('Cluster Assignments', fontsize=9, color='#444', pad=4)
    cluster_labels = [0]*10 + [1]*10 + [2]*10 + [3]*10
    for k in range(4):
        mask = np.array(cluster_labels) == k
        ax2.scatter(data[mask,0], data[mask,1],
                    color=CLUSTER_COLORS[k], s=30, alpha=0.8, label=f'C{k}')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xticks([]); ax2.set_yticks([])

    sidebar(fig, ac, [
        '## Core Idea',
        'Bottom-up merging: start with\nn clusters, merge closest pair\nat each step.',
        '',
        '## Algorithm',
        '① Each point = its own cluster',
        '② Merge pair with min Ward distance',
        '   ΔV = merge variance increase',
        '③ Repeat until k clusters remain',
        '',
        '## Properties',
        'Deterministic (no random init)',
        'Produces full dendrogram',
        'Cut at threshold → flat clusters',
        'O(n² log n) complexity',
        'Best for compact, equal clusters',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_06_agglomerative_ward.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 07 — HDBSCAN
# ═════════════════════════════════════════════════════════════════════════════
def slide_07():
    ac = ACCENT[7]
    fig = new_fig(ac, 7, 'HDBSCAN  (Hierarchical Density-Based)')
    rng = np.random.default_rng(0)

    ax = fig.add_axes([0.03, 0.06, 0.655, 0.86])
    ax_style(ax)

    # clusters of different shapes/densities
    c0 = rng.normal([2,2], 0.4, (80,2))
    c1 = rng.normal([6,2], 0.6, (80,2))
    # elongated
    t = np.linspace(0, 2*np.pi, 100)
    c2 = np.column_stack([3+2.5*np.cos(t), 6+1.2*np.sin(t)]) + rng.normal(0,0.25,(100,2))
    # small dense
    c3 = rng.normal([7.5,6.5], 0.25, (40,2))
    # noise
    noise = rng.uniform([0,0],[10,9], (25,2))

    cols_pts = ['#E53935','#1E88E5','#43A047','#FB8C00']
    for pts, col, lbl in zip([c0,c1,c2,c3], cols_pts, range(4)):
        ax.scatter(pts[:,0], pts[:,1], color=col, s=20, alpha=0.75, zorder=3,
                   label=f'Cluster {lbl}')
    ax.scatter(noise[:,0], noise[:,1], color='#AAAAAA', s=18, marker='x',
               alpha=0.6, zorder=2, label='Noise (−1)')

    # min_samples circle illustration
    sample_pt = np.array([2.0, 2.0])
    circle = plt.Circle(sample_pt, 0.6, color=ac, fill=False, lw=1.8,
                        linestyle='--', zorder=6)
    ax.add_patch(circle)
    ax.text(sample_pt[0]+0.65, sample_pt[1]+0.1, 'core\npoint',
            color=ac, fontsize=8, fontweight='bold')

    ax.set_xlim(-0.3, 10.5); ax.set_ylim(-0.3, 9.5)
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.legend(fontsize=9, loc='upper right', facecolor='white', edgecolor='#CCC')
    ax.set_title('Arbitrary-shape clusters + noise detection', fontsize=10,
                 color='#444', pad=6)

    sidebar(fig, ac, [
        '## Core Idea',
        'Density-based clustering:\nfinds clusters of arbitrary\nshape, labels noise as −1.',
        '',
        '## Algorithm',
        '① Build mutual reachability graph',
        '② Compute min spanning tree',
        '③ Condense cluster hierarchy',
        '④ Extract stable clusters',
        '',
        '## Properties',
        'No need to specify k',
        'Handles varying densities',
        'Robust to noise & outliers',
        'Elongated and curved shapes OK',
        'X = noise/outlier points',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_07_hdbscan.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 08 — Spectral Clustering
# ═════════════════════════════════════════════════════════════════════════════
def slide_08():
    ac = ACCENT[8]
    fig = new_fig(ac, 8, 'Spectral Clustering')
    rng = np.random.default_rng(3)

    # left: graph visualization
    ax = fig.add_axes([0.03, 0.06, 0.32, 0.86])
    ax_style(ax)
    ax.set_title('Similarity Graph', fontsize=10, color='#444', pad=6)

    n_per = 8
    angles0 = np.linspace(0, 2*np.pi, n_per, endpoint=False)
    angles1 = np.linspace(0, 2*np.pi, n_per, endpoint=False)
    c0_pts = np.column_stack([1.5*np.cos(angles0), 1.5*np.sin(angles0)])
    c1_pts = np.column_stack([4.0*np.cos(angles1)+6, 4.0*np.sin(angles1)+5])

    cols_g = ['#E53935']*n_per + ['#1E88E5']*n_per
    all_g = np.vstack([c0_pts, c1_pts])

    for i in range(n_per):
        for j in range(i+1, n_per):
            ax.plot([c0_pts[i,0],c0_pts[j,0]],[c0_pts[i,1],c0_pts[j,1]],
                    color='#E53935', lw=0.8, alpha=0.35, zorder=2)
            ax.plot([c1_pts[i,0],c1_pts[j,0]],[c1_pts[i,1],c1_pts[j,1]],
                    color='#1E88E5', lw=0.8, alpha=0.35, zorder=2)

    for pts, col in zip([c0_pts, c1_pts], ['#E53935','#1E88E5']):
        ax.scatter(pts[:,0], pts[:,1], color=col, s=60, zorder=4, edgecolors='white', lw=1)

    ax.set_xlim(-3,12); ax.set_ylim(-5,10)
    ax.set_xticks([]); ax.set_yticks([])

    # middle: Laplacian eigenvectors
    ax2 = fig.add_axes([0.36, 0.06, 0.17, 0.86])
    ax_style(ax2, '#FAFAFA')
    ax2.set_title('Laplacian\nEigenvectors', fontsize=10, color='#444', pad=6)

    n = n_per * 2
    eig_vals = np.zeros(n)
    eig_vals[:n_per] = rng.normal(0.1, 0.05, n_per)
    eig_vals[n_per:] = rng.normal(0.9, 0.05, n_per)
    y_pos = np.arange(n)
    colors_bar = ['#E53935']*n_per + ['#1E88E5']*n_per
    ax2.barh(y_pos, eig_vals, color=colors_bar, alpha=0.85, height=0.7)
    ax2.axvline(0.5, color=ac, lw=2, ls='--')
    ax2.set_xlabel('v₂ value', fontsize=9)
    ax2.set_yticks([]); ax2.set_xlim(-0.1, 1.1)

    # right: K-Means in embedding space
    ax3 = fig.add_axes([0.54, 0.06, 0.145, 0.86])
    ax_style(ax3)
    ax3.set_title('Embedded\nK-Means', fontsize=10, color='#444', pad=6)
    emb = np.zeros((n, 2))
    emb[:n_per] = rng.normal([0.1,0.1], 0.05, (n_per,2))
    emb[n_per:] = rng.normal([0.9,0.9], 0.05, (n_per,2))
    for k, col in zip([0,1],['#E53935','#1E88E5']):
        sl = slice(k*n_per, (k+1)*n_per)
        ax3.scatter(emb[sl,0], emb[sl,1], color=col, s=50,
                    edgecolors='white', lw=1, zorder=4)
    ax3.set_xticks([]); ax3.set_yticks([])

    # step labels between panels
    for x, lbl in [(0.342, '①\n→'), (0.525, '②\n→')]:
        fig.text(x, 0.49, lbl, ha='center', va='center',
                 fontsize=14, color='#555', fontweight='bold')

    sidebar(fig, ac, [
        '## Core Idea',
        'Use graph Laplacian eigenvectors\nas low-dim embedding, then\napply K-Means.',
        '',
        '## Algorithm',
        '① Build affinity matrix W (RBF kernel)',
        '② Compute normalized Laplacian L',
        '③ Extract top-k eigenvectors',
        '④ K-Means on eigenvector rows',
        '',
        '## Properties',
        'Finds non-convex clusters',
        'Works on graph/manifold data',
        'Kernel controls similarity',
        'O(n³) — use on subsample',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_08_spectral.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 09 — Isolation Forest + K-Means
# ═════════════════════════════════════════════════════════════════════════════
def slide_09():
    ac = ACCENT[9]
    fig = new_fig(ac, 9, 'Isolation Forest + K-Means')
    rng = np.random.default_rng(5)

    # Pipeline diagram
    ax = fig.add_axes([0.01, 0.55, 0.685, 0.36])
    ax.set_facecolor('white'); ax.set_xlim(0,10); ax.set_ylim(0,2); ax.axis('off')

    pipe = [
        (0.2,  0.4, 1.5, 1.2, 'Raw\nData',           '#ECEFF1', '#37474F'),
        (2.1,  0.4, 1.8, 1.2, 'Isolation\nForest',   '#FFCCBC', ac),
        (4.2,  0.4, 1.6, 1.2, 'Filter\nOutliers',    '#FFF9C4', '#F57F17'),
        (6.1,  0.4, 1.6, 1.2, 'K-Means\nClustering', '#C8E6C9', '#2E7D32'),
        (8.0,  0.4, 1.7, 1.2, 'Labels\n+ Outliers',  '#E3F2FD', '#1565C0'),
    ]
    ctrs = []
    for (bx,by,bw,bh,lbl,fc,tc) in pipe:
        rect = FancyBboxPatch((bx,by), bw, bh, boxstyle='round,pad=0.08',
                              facecolor=fc, edgecolor='#BBBBBB', linewidth=1.5,
                              transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        cx=bx+bw/2; cy=by+bh/2; ctrs.append((cx,cy,bx+bw))
        ax.text(cx, cy, lbl, color=tc, fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=4)

    for i in range(len(ctrs)-1):
        ax.annotate('', xy=(pipe[i+1][0], ctrs[i][1]),
                    xytext=(ctrs[i][2], ctrs[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.0), zorder=5)

    ax.text(5.0, 1.85, 'Two-Stage Pipeline', ha='center', fontsize=11,
            color='#444', fontweight='bold')

    # scatter: outliers vs inliers
    ax2 = fig.add_axes([0.03, 0.06, 0.32, 0.45])
    ax_style(ax2)
    ax2.set_title('Step 1: Outlier Detection', fontsize=10, color='#444', pad=5)
    inliers = rng.normal([5,5], 1.2, (200,2))
    outliers = rng.uniform([0,0],[10,10], (20,2))
    ax2.scatter(inliers[:,0], inliers[:,1], color='#90CAF9', s=18,
                alpha=0.7, label='Inliers', zorder=3)
    ax2.scatter(outliers[:,0], outliers[:,1], color='#EF5350', s=50,
                marker='x', lw=2, label='Outliers', zorder=4)
    ax2.set_xlabel('Feature 1', fontsize=9)
    ax2.set_ylabel('Feature 2', fontsize=9)
    ax2.legend(fontsize=9, facecolor='white')

    # scatter: after clustering
    ax3 = fig.add_axes([0.37, 0.06, 0.32, 0.45])
    ax_style(ax3)
    ax3.set_title('Step 2: K-Means on Inliers', fontsize=10, color='#444', pad=5)
    centers_km = np.array([[4,4],[6,4],[5,7]])
    for k, (cx,cy) in enumerate(centers_km):
        pts = rng.normal([cx,cy], 0.8, (65,2))
        ax3.scatter(pts[:,0], pts[:,1], color=CLUSTER_COLORS[k], s=18,
                    alpha=0.7, zorder=3)
        ax3.scatter(cx, cy, marker='*', s=300, color='white',
                    edgecolors=CLUSTER_COLORS[k], lw=2, zorder=5)
    ax3.scatter(outliers[:,0], outliers[:,1], color='#EF5350', s=50,
                marker='x', lw=2, alpha=0.5, label='Outlier(−1)', zorder=4)
    ax3.set_xlabel('Feature 1', fontsize=9)
    ax3.legend(fontsize=8, facecolor='white')

    sidebar(fig, ac, [
        '## Core Idea',
        'Two-stage: detect & remove\nanomalies first, then cluster\nclean data with K-Means.',
        '',
        '## Algorithm',
        '① Isolation Forest on full dataset',
        '   anomaly score via iTree depth',
        '② Filter out flagged outliers',
        '③ K-Means on clean inliers',
        '④ Outliers labeled as −1',
        '',
        '## Properties',
        'Robust to noise and anomalies',
        'Isolation score = path length',
        '★ = cluster centroids',
        'X = detected outliers (−1)',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_09_isolation_forest_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
# 10 — VAE + K-Means
# ═════════════════════════════════════════════════════════════════════════════
def slide_10():
    ac = ACCENT[10]
    fig = new_fig(ac, 10, 'Variational Autoencoder + K-Means  (VAE)')
    rng = np.random.default_rng(9)

    # Pipeline
    ax = fig.add_axes([0.01, 0.08, 0.685, 0.82])
    ax.set_facecolor('white'); ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')

    blocks = [
        (0.15, 2.3, 1.3, 1.4, 'Input\n126D',             '#ECEFF1','#37474F'),
        (1.85, 1.9, 1.6, 2.2, 'Encoder\nq(z|x)',         '#E3F2FD', ac),
        (3.8,  2.0, 1.5, 1.0, 'μ(x)',                    '#BBDEFB', ac),
        (3.8,  3.3, 1.5, 1.0, 'σ(x)',                    '#BBDEFB', ac),
        (5.6,  2.3, 1.4, 1.4, 'z ~ N(μ,σ²)',             '#D1C4E9','#6A1B9A'),
        (7.3,  1.9, 1.6, 2.2, 'Decoder\np(x|z)',         '#E8F5E9','#2E7D32'),
        (9.2,  2.3, 1.3, 1.4, 'Recon\nx̂',               '#C8E6C9','#1B5E20'),
        (5.6,  0.2, 1.4, 1.1, 'K-Means\nLabels',         '#FFF9C4','#E65100'),
    ]

    node_info = {}
    for (bx,by,bw,bh,lbl,fc,tc) in blocks:
        rect = FancyBboxPatch((bx,by), bw, bh, boxstyle='round,pad=0.08',
                              facecolor=fc, edgecolor='#BBBBBB', linewidth=1.5,
                              transform=ax.transData, zorder=3)
        ax.add_patch(rect)
        cx=bx+bw/2; cy=by+bh/2
        node_info[lbl.split('\n')[0]] = (cx,cy,bx+bw,by,bh)
        ax.text(cx, cy, lbl, color=tc, fontsize=9.5, fontweight='bold',
                ha='center', va='center', zorder=4)

    # arrows
    arrows = [
        ('Input',   'Encoder\nq(z|x)', False),
    ]
    # Input -> Encoder
    ax.annotate('', xy=(1.85, 3.0), xytext=(1.45, 3.0),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.0), zorder=5)
    # Encoder -> mu
    ax.annotate('', xy=(3.8, 2.5), xytext=(3.45, 2.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8), zorder=5)
    # Encoder -> sigma
    ax.annotate('', xy=(3.8, 3.8), xytext=(3.45, 3.8),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8), zorder=5)
    # mu,sigma -> z
    ax.annotate('', xy=(5.6, 3.0), xytext=(5.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.8), zorder=5)
    ax.annotate('', xy=(5.6, 3.0), xytext=(5.3, 3.8),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.8), zorder=5)
    # z -> decoder
    ax.annotate('', xy=(7.3, 3.0), xytext=(7.0, 3.0),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.0), zorder=5)
    # decoder -> recon
    ax.annotate('', xy=(9.2, 3.0), xytext=(8.9, 3.0),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.0), zorder=5)
    # z -> k-means
    ax.annotate('', xy=(6.3, 1.3), xytext=(6.3, 2.3),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=2.2), zorder=5)

    # reparameterization note
    ax.text(6.3, 3.95, 'reparameterization\ntrick: z = μ + ε·σ',
            ha='center', fontsize=8.5, color='#6A1B9A', style='italic')

    # loss annotations
    ax.text(5.0, 5.5, 'Loss = Reconstruction (MSE) + KL Divergence  (β-VAE)',
            ha='center', fontsize=10.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F3E5F5', ec='#9C27B0', lw=1.5))

    sidebar(fig, ac, [
        '## Core Idea',
        'Learn a structured probabilistic\nlatent space, then cluster\nthe sampled embeddings.',
        '',
        '## vs Autoencoder',
        'AE: deterministic latent z',
        'VAE: z ~ N(μ(x), σ²(x))',
        'Smoother, more separable space',
        '',
        '## Algorithm',
        '① Train VAE with ELBO loss',
        '② Sample z for each sample',
        '③ K-Means on z vectors',
        '',
        '## Properties',
        'Structured latent manifold',
        'Generative model (sample x̂)',
        'KL term regularizes z space',
        'β controls clustering quality',
    ])

    path = os.path.join(OUT_DIR, 'exec_slide_10_vae_kmeans.png')
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {path}')


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating executive slides (white background)...')
    slide_01(); slide_02(); slide_03(); slide_04(); slide_05()
    slide_06(); slide_07(); slide_08(); slide_09(); slide_10()
    print(f'\nDone. All 10 slides saved to {OUT_DIR}')
