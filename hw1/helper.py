"""Visualization helpers for HW1 (given - do not modify).

    from helper import show_examples, animate_attack

  show_examples(model, x_orig, x_pert, y, n=6, normalized=True, title='')
      -> figure: original | perturbation (amplified) | perturbed image, with predictions
  animate_attack(model, x_orig, trajectory, y, normalized=True, path='attack_process.gif', fps=4, title='')
      -> GIF of the attack process on ONE image, iteration by iteration

Both functions are documented in the notebook, and in their docstrings (`help(show_examples)`).
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image as _GifImage, display as _display

__all__ = ['show_examples', 'animate_attack']

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']   # CIFAR-10 class names
DEFAULT_MEAN, DEFAULT_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)       # the normalisation of `test_loader` in the notebook


def _to_pixel_fn(normalized, mean, std):
    """Returns a function that maps images to [0, 1] pixel space (undoing the normalisation if `normalized`)."""
    m, s = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)
    return (lambda x: (x * s + m).clamp(0, 1)) if normalized else (lambda x: x.clamp(0, 1))


def show_examples(model, x_orig, x_pert, y, n=6, normalized=True, title='', mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Shows `n` examples as rows:  original image | perturbation (amplified) | perturbed image,
    with the true label, the model's prediction and its confidence (green = correct, red = wrong).
      model:          the classifier; it must accept images in the same space as x_orig / x_pert (it is switched to eval mode)
      x_orig, x_pert: tensors [B, 3, 32, 32] - original and perturbed images (x_pert[i] is the perturbed x_orig[i])
      y:              true labels [B]
      n:              number of examples (rows)
      normalized:     True  -> images are in the normalised space of `test_loader` (as returned by the dataloader)
                      False -> images are in [0, 1] pixel space
      title:          optional title of the figure
      mean, std:      the normalisation constants of `test_loader` (only needed if you changed them)
    Examples that the model classified correctly before the perturbation are shown first."""
    model.eval()
    dev = next(model.parameters()).device
    to_pix = _to_pixel_fn(normalized, mean, std)
    m = min(len(x_orig), 200)
    with torch.no_grad():
        p_orig = F.softmax(model(x_orig[:m].to(dev)), dim=1).cpu()
        p_pert = F.softmax(model(x_pert[:m].to(dev)), dim=1).cpu()
    was_ok = p_orig.argmax(1) == y[:m].cpu()
    idx = torch.cat([was_ok.nonzero().flatten(), (~was_ok).nonzero().flatten()])[:n].tolist()

    fig, axs = plt.subplots(len(idx), 3, figsize=(8.5, 2.9 * len(idx)), squeeze=False)
    for r, i in enumerate(idx):
        a, b = to_pix(x_orig[i:i + 1].cpu())[0], to_pix(x_pert[i:i + 1].cpu())[0]
        d = b - a
        gain = 0.5 / max(d.abs().max().item(), 1e-8)                       # amplify the perturbation so that it is visible
        l0, l2, linf = int((d.abs() > 1e-6).sum()), d.norm().item(), d.abs().max().item()
        label = int(y[i])
        panels = [(a, p_orig[i], 'original'),
                  (d * gain + 0.5, None, f'perturbation (amplified x{gain:.0f})\nL0={l0}   L2={l2:.2f}   Linf={linf * 255:.1f}/255'),
                  (b, p_pert[i], 'perturbed')]
        for c, (img, p, name) in enumerate(panels):
            axs[r, c].imshow(img.permute(1, 2, 0).numpy()); axs[r, c].axis('off')
            if p is None:
                axs[r, c].set_title(name, fontsize=8)
            else:
                pred = int(p.argmax())
                axs[r, c].set_title(f'{name}\ntrue: {CLASSES[label]} | pred: {CLASSES[pred]} ({p[pred]:.0%})',
                                    fontsize=8, color='green' if pred == label else 'red')
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.show()


def animate_attack(model, x_orig, trajectory, y, normalized=True, path='attack_process.gif', fps=4, title='',
                   mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Creates (saves and shows) a GIF of the attack process on ONE image: how the perturbation grows, how the image
    and the model's prediction change, and how the confidence of the true class collapses - iteration by iteration.
      model:       the classifier; it must accept images in the same space as x_orig / trajectory (it is switched to eval mode)
      x_orig:      the original image [3, 32, 32]
      trajectory:  the perturbed image after every iteration, a tensor [T, 3, 32, 32]
                   (first entry = the starting point of the attack, then one entry per iteration) - you record it in your attack
      y:           the true label of the image (int)
      normalized:  True -> images are in the normalised space of `test_loader`;  False -> in [0, 1] pixel space
      path:        where the GIF is saved
      fps:         frames (= attack iterations) per second
      title:       optional title of the figure
      mean, std:   the normalisation constants of `test_loader` (only needed if you changed them)"""
    model.eval()
    dev = next(model.parameters()).device
    to_pix = _to_pixel_fn(normalized, mean, std)
    x0 = x_orig.detach().reshape(1, 3, 32, 32).cpu()
    traj = trajectory.detach().reshape(-1, 3, 32, 32).cpu()
    with torch.no_grad():
        p0 = F.softmax(model(x0.to(dev)), dim=1).cpu()[0]
        probs = F.softmax(model(traj.to(dev)), dim=1).cpu()                 # [T, 10] class probabilities at every iteration
    a, frames = to_pix(x0)[0], to_pix(traj)
    deltas = frames - a
    gain = 0.5 / max(deltas.abs().max().item(), 1e-8)                        # one fixed amplification for all frames
    label, T = int(y), len(frames)
    true_conf = probs[:, label].numpy()

    fig = plt.figure(figsize=(9, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    ax_o, ax_d, ax_p = (fig.add_subplot(gs[0, j]) for j in range(3))
    ax_b, ax_l = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    pred0 = int(p0.argmax())
    ax_o.imshow(a.permute(1, 2, 0).numpy()); ax_o.axis('off')
    ax_o.set_title(f'original\ntrue: {CLASSES[label]} | pred: {CLASSES[pred0]} ({p0[pred0]:.0%})', fontsize=9, color='green' if pred0 == label else 'red')
    im_d = ax_d.imshow((deltas[0] * gain + 0.5).permute(1, 2, 0).numpy(), vmin=0, vmax=1); ax_d.axis('off')
    im_p = ax_p.imshow(frames[0].permute(1, 2, 0).numpy()); ax_p.axis('off')
    bars = ax_b.bar(range(10), probs[0].numpy())
    ax_b.set_ylim(0, 1.05); ax_b.set_xticks(range(10)); ax_b.set_xticklabels(CLASSES, rotation=90, fontsize=7); ax_b.set_title('class probabilities', fontsize=9)
    ax_l.plot(range(T), true_conf, color='lightgray')
    line, = ax_l.plot([], [], color='tab:blue'); dot, = ax_l.plot([], [], 'o', color='tab:red')
    ax_l.set_xlim(0, max(T - 1, 1)); ax_l.set_ylim(0, 1.05); ax_l.set_xlabel('iteration'); ax_l.set_ylabel(f'P({CLASSES[label]})  (true class)')
    ax_l.axhline(0.5, color='gray', ls=':', lw=0.8); ax_l.set_title('confidence in the true class', fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(t):
        d = deltas[t]
        l0, l2, linf = int((d.abs() > 1e-6).sum()), d.norm().item(), d.abs().max().item()
        im_d.set_data((d * gain + 0.5).permute(1, 2, 0).numpy())
        ax_d.set_title(f'perturbation (amplified x{gain:.0f})\nL0={l0}  L2={l2:.2f}  Linf={linf * 255:.1f}/255', fontsize=8)
        im_p.set_data(frames[t].permute(1, 2, 0).numpy())
        pred = int(probs[t].argmax())
        ax_p.set_title(f'iteration {t}\npred: {CLASSES[pred]} ({probs[t, pred]:.0%})', fontsize=9, color='green' if pred == label else 'red')
        for k, (b, h) in enumerate(zip(bars, probs[t].numpy())):
            b.set_height(h); b.set_color('tab:red' if (k == pred and pred != label) else ('tab:green' if k == label else 'lightgray'))
        line.set_data(range(t + 1), true_conf[:t + 1]); dot.set_data([t], [true_conf[t]])
        return []

    order = list(range(T)) + [T - 1] * 4                                     # pause on the last frame
    FuncAnimation(fig, update, frames=order, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    with open(path, 'rb') as f:
        _display(_GifImage(data=f.read(), format='gif'))
    print(f'saved: {path}')
