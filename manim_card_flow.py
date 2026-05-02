from __future__ import annotations

import numpy as np
from manim import *

from pokemon_yard_model import make_initial_distribution, simulate_exchanges, gini, group_shares


class CardFlowScene(Scene):
    def construct(self):
        initial = make_initial_distribution(seed=7)
        result = simulate_exchanges(initial, steps=180, seed=11)
        history = result.history
        order = result.initial_order
        rank = {kid: pos for pos, kid in enumerate(order)}
        max_cards = max(1, int(history.max()))

        title = Text("100 детей: больше карточек -> больше возможностей обмена", font_size=30)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.8)

        baseline_y = -2.55
        chart_width = 12.0
        chart_height = 4.1
        bar_gap = 0.018
        bar_width = chart_width / len(initial) - bar_gap
        x0 = -chart_width / 2

        def color_for_child(kid: int):
            pos = rank[kid]
            if pos < 20:
                return RED_C
            if pos >= 80:
                return GREEN_C
            return BLUE_C

        def bar_x(pos: int):
            return x0 + (pos + 0.5) * chart_width / len(initial)

        def make_bars(snapshot: np.ndarray):
            bars = VGroup()
            for pos, kid in enumerate(order):
                h = max(0.035, chart_height * snapshot[kid] / max_cards)
                bar = Rectangle(width=bar_width, height=h, stroke_width=0)
                bar.set_fill(color_for_child(kid), opacity=0.92)
                bar.move_to([bar_x(pos), baseline_y + h / 2, 0])
                bars.add(bar)
            return bars

        def make_stats(frame_idx: int):
            snapshot = history[frame_idx]
            shares = group_shares(history[[frame_idx]], initial)
            text = (
                f"шаг {frame_idx:03d}   "
                f"Gini {gini(snapshot):.2f}   "
                f"нижние 20%: {shares['bottom_20'][0] * 100:.1f}%   "
                f"верхние 20%: {shares['top_20'][0] * 100:.1f}%"
            )
            label = Text(text, font_size=22)
            label.next_to(title, DOWN, buff=0.28)
            return label

        axis = Line([x0, baseline_y, 0], [x0 + chart_width, baseline_y, 0], color=GRAY_B)
        left_label = Text("меньше на старте", font_size=18, color=RED_C).next_to(axis, DOWN).align_to(axis, LEFT)
        right_label = Text("больше на старте", font_size=18, color=GREEN_C).next_to(axis, DOWN).align_to(axis, RIGHT)

        bars = make_bars(history[0])
        stats = make_stats(0)
        self.play(Create(axis), FadeIn(left_label), FadeIn(right_label), FadeIn(stats), FadeIn(bars), run_time=1.0)

        frame_indices = list(range(12, len(history), 12))
        if frame_indices[-1] != len(history) - 1:
            frame_indices.append(len(history) - 1)

        previous = 0
        for frame_idx in frame_indices:
            new_bars = make_bars(history[frame_idx])
            new_stats = make_stats(frame_idx)

            window_events = [
                event
                for event in result.events
                if previous <= event[0] < frame_idx and rank[event[2]] < rank[event[1]]
            ]
            window_events = sorted(window_events, key=lambda event: event[3], reverse=True)[:10]

            particles = VGroup()
            paths = []
            for _, initiator, counterparty, transfer in window_events:
                start = np.array([bar_x(rank[counterparty]), baseline_y + 0.25, 0])
                end = np.array([bar_x(rank[initiator]), baseline_y + 3.75, 0])
                dot = Dot(start, radius=0.035 + 0.012 * min(transfer, 5), color=YELLOW)
                particles.add(dot)
                arc = ArcBetweenPoints(start, end, angle=-0.55, color=YELLOW)
                paths.append(arc)

            animations = [Transform(bars, new_bars), Transform(stats, new_stats)]
            animations.extend(MoveAlongPath(dot, path) for dot, path in zip(particles, paths))
            self.play(*animations, run_time=0.55, rate_func=linear)
            if len(particles) > 0:
                self.play(FadeOut(particles), run_time=0.12)
            previous = frame_idx

        legend = VGroup(
            Dot(color=RED_C), Text("нижние 20% старта", font_size=18),
            Dot(color=BLUE_C), Text("средние 60%", font_size=18),
            Dot(color=GREEN_C), Text("верхние 20% старта", font_size=18),
        ).arrange(RIGHT, buff=0.16)
        legend.to_edge(DOWN)
        self.play(FadeIn(legend), run_time=0.6)
        self.wait(1.5)
