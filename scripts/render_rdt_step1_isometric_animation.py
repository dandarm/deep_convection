#!/usr/bin/env python3
"""Animate the six-panel isometric RDT step-1 concept painting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

STEP_COPY = (
    (
        "Campo nuvoloso",
        "La scena atmosferica conserva forma, scala e relazione tra le torri.",
    ),
    (
        "Superficie termica",
        "IR 10.8 diventa una topografia: i nuclei più freddi formano depressioni.",
    ),
    (
        "Minimi regionali",
        "I minimi freddi e sufficientemente separati diventano seed delle celle.",
    ),
    (
        "Crescita dei bacini",
        "Ogni seed cresce verso temperature più calde seguendo contorni annidati.",
    ),
    (
        "Saddle e separazione",
        "Quando due bacini si incontrano, il passo termico determina il bordo adattivo.",
    ),
    (
        "Celle finali",
        "Restano i contorni delle torri significative; gli oggetti piccoli vengono filtrati.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storyboard",
        type=Path,
        default=ROOT
        / "figures"
        / "videomae_pretraining"
        / "rdt_step1_isometric_storyboard.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "figures"
        / "videomae_pretraining"
        / "rdt_step1_isometric_algorithm.mp4",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hold-seconds", type=float, default=1.8)
    parser.add_argument("--transition-seconds", type=float, default=0.65)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def split_storyboard(path: Path) -> list[np.ndarray]:
    image = plt.imread(path)
    height, width = image.shape[:2]
    if width % 3 or height % 2:
        raise ValueError(f"Expected a 3x2 divisible storyboard; received {width}x{height}.")
    panel_width = width // 3
    panel_height = height // 2
    panels = []
    for row in range(2):
        for column in range(3):
            panel = image[
                row * panel_height : (row + 1) * panel_height,
                column * panel_width : (column + 1) * panel_width,
            ]
            panels.append(np.asarray(panel[..., :3], dtype=np.float32))
    return panels


def blended_panel(
    panels: list[np.ndarray], current: int, following: int, alpha: float
) -> np.ndarray:
    if current == following or alpha <= 0:
        return panels[current]
    return panels[current] * (1.0 - alpha) + panels[following] * alpha


def render(args: argparse.Namespace) -> None:
    panels = split_storyboard(args.storyboard)
    if len(panels) != len(STEP_COPY):
        raise RuntimeError("Storyboard and copy must contain the same number of steps.")
    hold_frames = max(1, round(args.hold_seconds * args.fps))
    transition_frames = max(1, round(args.transition_seconds * args.fps))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi, facecolor="#071321")
    image_axis = fig.add_axes([0.0, 0.0, 0.5625, 1.0])
    image_axis.set_axis_off()
    image_artist = image_axis.imshow(panels[0], interpolation="lanczos")

    copy_axis = fig.add_axes([0.60, 0.10, 0.35, 0.80], facecolor="none")
    copy_axis.set_axis_off()
    copy_axis.text(
        0.0,
        0.96,
        "RDT-CW-INSPIRED · STEP 1",
        transform=copy_axis.transAxes,
        color="#6dd5fa",
        fontsize=12,
        fontweight="semibold",
        va="top",
    )
    number_artist = copy_axis.text(
        0.0,
        0.78,
        "01",
        transform=copy_axis.transAxes,
        color="#ffffff",
        fontsize=54,
        fontweight="bold",
        va="top",
    )
    title_artist = copy_axis.text(
        0.0,
        0.53,
        STEP_COPY[0][0],
        transform=copy_axis.transAxes,
        color="#ffffff",
        fontsize=25,
        fontweight="semibold",
        va="top",
    )
    description_artist = copy_axis.text(
        0.0,
        0.42,
        STEP_COPY[0][1],
        transform=copy_axis.transAxes,
        color="#cbd5e1",
        fontsize=15,
        va="top",
        wrap=True,
        linespacing=1.45,
    )
    status_artist = copy_axis.text(
        0.0,
        0.10,
        "",
        transform=copy_axis.transAxes,
        color="#94a3b8",
        fontsize=11,
        va="bottom",
    )
    dots = []
    for index in range(len(STEP_COPY)):
        dot = copy_axis.scatter(
            0.035 + index * 0.075,
            0.02,
            s=55,
            color="#334155",
            transform=copy_axis.transAxes,
            clip_on=False,
        )
        dots.append(dot)

    fig.text(
        0.98,
        0.028,
        "Illustrazione algoritmica · non derivata da immagini IR",
        ha="right",
        va="bottom",
        color="#64748b",
        fontsize=9,
    )

    def update_copy(step: int, transition: float = 0.0) -> None:
        visible_step = min(step + (1 if transition >= 0.5 else 0), len(STEP_COPY) - 1)
        number_artist.set_text(f"{visible_step + 1:02d}")
        title_artist.set_text(STEP_COPY[visible_step][0])
        description_artist.set_text(STEP_COPY[visible_step][1])
        status_artist.set_text(f"Procedura {visible_step + 1} di {len(STEP_COPY)}")
        for index, dot in enumerate(dots):
            dot.set_color("#22d3ee" if index <= visible_step else "#334155")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=5500,
        metadata={
            "title": "RDT step 1 isometric algorithm animation",
            "artist": "deep_convection",
            "comment": "Generated concept art; not derived from satellite pixels",
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    with writer.saving(fig, str(args.output), dpi=args.dpi):
        for step in range(len(panels)):
            update_copy(step)
            image_artist.set_data(panels[step])
            for _ in range(hold_frames):
                writer.grab_frame()

            if step == len(panels) - 1:
                continue
            for transition_index in range(1, transition_frames + 1):
                phase = transition_index / transition_frames
                eased = phase * phase * (3.0 - 2.0 * phase)
                image_artist.set_data(blended_panel(panels, step, step + 1, eased))
                update_copy(step, eased)
                writer.grab_frame()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    render(args)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
