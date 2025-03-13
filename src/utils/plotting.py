import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
import random


def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure_old(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2, marker='x', linewidths=0.1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2, marker='x', linewidths=0.1)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))

        ###########
        # add random indices
        indices = list(range(len(mkpts0)))
        random.shuffle(indices)
        ###########

        ###########
        use_uniform_color = True
        if use_uniform_color:
            # c = color[i]
            fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                                 (fkpts0[i, 1], fkpts1[i, 1]),
                                                 transform=fig.transFigure, c='b', linewidth=0.3)
                         ###########
                         for i in indices[:20]]
            ###########
            # for i in range(len(mkpts0))]

            # axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c='cyan', s=0.1)
            # axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c='cyan', s=0.1)
            # use red cross to highlight
            axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c='red', s=1, marker='x', linewidths=0.1)
            axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c='red', s=1, marker='x', linewidths=0.1)
        else:
            fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                                 (fkpts0[i, 1], fkpts1[i, 1]),
                                                 transform=fig.transFigure, c=color[i], linewidth=0.1)
                         for i in indices[:10]]
            # for i in range(len(mkpts0))]

            axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=0.1)
            axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=0.1)
        ###########

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None, save_individual=True):
    """
    Visualize matching keypoints between two images, with an option to save them as separate images.

    :param img0: First image (numpy array).
    :param img1: Second image (numpy array).
    :param mkpts0: Matched keypoints in the first image (N, 2).
    :param mkpts1: Matched keypoints in the second image (N, 2).
    :param color: Color for matches (can be uniform or per-match).
    :param kpts0: Optional keypoints for image 0 (N, 2).
    :param kpts1: Optional keypoints for image 1 (N, 2).
    :param text: List of annotation text.
    :param dpi: Figure resolution.
    :param path: Path to save the combined figure.
    :param save_individual: Whether to save the individual images separately.
    :return: None or Matplotlib figure (if `path` is None).
    """
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} vs mkpts1: {mkpts1.shape[0]}'

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')

    for i in range(2):  # Remove axis ticks and frames
        axes[i].axis("off")

    plt.tight_layout(pad=1)

    if kpts0 is not None and kpts1 is not None:
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2, marker='x', linewidths=0.1)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2, marker='x', linewidths=0.1)

    # Draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))

        indices = list(range(len(mkpts0)))
        random.shuffle(indices)

        use_uniform_color = True
        if use_uniform_color:
            fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                                 (fkpts0[i, 1], fkpts1[i, 1]),
                                                 transform=fig.transFigure, c='b', linewidth=0.3)
                         for i in indices[:20]]
            axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c='red', s=1, marker='x', linewidths=0.1)
            axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c='red', s=1, marker='x', linewidths=0.1)
        else:
            fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                                 (fkpts0[i, 1], fkpts1[i, 1]),
                                                 transform=fig.transFigure, c=color[i], linewidth=0.1)
                         for i in indices[:10]]
            axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=0.1)
            axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=0.1)

    # Add annotation text
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # Save or return figure
    # if path:
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Save individual images if required
    if save_individual:
        fig_individual_1, ax0 = plt.subplots(figsize=(10, 6), dpi=dpi)
        ax0.imshow(img0, cmap='gray')
        ax0.scatter(mkpts0[:, 0], mkpts0[:, 1], c='red', s=1, marker='x', linewidths=0.1)
        ax0.axis("off")
        plt.savefig(path.replace('.jpg', '_left.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close(fig_individual_1)

        fig_individual_2, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)
        ax1.imshow(img1, cmap='gray')
        ax1.scatter(mkpts1[:, 0], mkpts1[:, 1], c='red', s=1, marker='x', linewidths=0.1)
        ax1.axis("off")
        plt.savefig(path.replace('.jpg', '_right.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close(fig_individual_2)

    a = 0


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)

    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)

    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]

    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure


def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.

    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
            milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2 - x * 2, x * 2, np.zeros_like(x), np.ones_like(x) * alpha], -1), 0, 1)