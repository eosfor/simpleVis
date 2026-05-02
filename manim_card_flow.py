from __future__ import annotations

import numpy as np
from manim import *

from pokemon_yard_model import make_initial_distribution, simulate_exchanges, gini, group_shares


def group_labels(values: np.ndarray) -> np.ndarray:
    n = len(values)
    order = np.argsort(values)
    labels = np.empty(n, dtype=object)
    labels[order[: n // 5]] = "bottom_20"
    labels[order[n // 5 : 4 * n // 5]] = "middle_60"
    labels[order[4 * n // 5 :]] = "top_20"
    return labels


def group_color(label: str):
    if label == "bottom_20":
        return RED_C
    if label == "top_20":
        return GREEN_C
    return BLUE_C


class CardFlowScene(Scene):
    def construct(self):
        initial_counts = make_initial_distribution(seed=7)
        result = simulate_exchanges(initial_counts, steps=180, seed=11)
        initial = result.initial
        history = result.history
        order = result.initial_order
        rank = {kid: pos for pos, kid in enumerate(order)}
        max_value = max(1.0, float(history.max()))

        title = Text("Добровольные обмены: концентрация рыночной ценности", font_size=30)
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
                h = max(0.035, chart_height * snapshot[kid] / max_value)
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
                dot = Dot(start, radius=0.035 + 0.004 * min(transfer, 12), color=YELLOW)
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


class CircleTradeScene(Scene):
    def construct(self):
        initial_counts = make_initial_distribution(seed=7)
        result = simulate_exchanges(initial_counts, steps=180, seed=11)
        initial = result.initial
        history = result.history
        card_count_history = result.card_count_history
        order = result.initial_order
        rank = {kid: pos for pos, kid in enumerate(order)}
        n = len(initial_counts)
        max_value = max(1.0, float(history.max()))

        title = Text("Bundle-сделки: меняется рыночная ценность коллекций", font_size=30)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.8)

        circle_radius = 2.72
        center = np.array([0.0, -0.12, 0.0])
        positions = []
        for pos in range(n):
            angle = TAU * pos / n + PI / 2
            positions.append(center + circle_radius * np.array([np.cos(angle), np.sin(angle), 0.0]))

        def color_for_child(kid: int):
            pos = rank[kid]
            if pos < 20:
                return RED_C
            if pos >= 80:
                return GREEN_C
            return BLUE_C

        def radius_for_value(value: float):
            # Diameter is proportional to market value, with a tiny floor so
            # low-value collections remain visible.
            return 0.025 + 0.235 * value / max_value

        def make_nodes(snapshot: np.ndarray):
            nodes = VGroup()
            for pos, kid in enumerate(order):
                node = Circle(
                    radius=radius_for_value(float(snapshot[kid])),
                    stroke_width=1.0,
                    stroke_color=WHITE,
                )
                node.set_fill(color_for_child(kid), opacity=0.88)
                node.move_to(positions[pos])
                nodes.add(node)
            return nodes

        def make_stats(frame_idx: int):
            snapshot = history[frame_idx]
            card_snapshot = card_count_history[frame_idx]
            text = (
                f"шаг {frame_idx:03d}   "
                f"Gini(value) {gini(snapshot):.2f}   "
                f"0 карточек: {np.sum(card_snapshot == 0):02d}   "
                f"max value: {snapshot.max():.0f}"
            )
            label = Text(text, font_size=22)
            label.next_to(title, DOWN, buff=0.25)
            return label

        ring = Circle(radius=circle_radius, color=GRAY_B, stroke_width=1.5)
        ring.move_to(center)
        nodes = make_nodes(history[0])
        stats = make_stats(0)

        legend = VGroup(
            Dot(color=RED_C), Text("нижние 20% старта", font_size=17),
            Dot(color=BLUE_C), Text("средние 60%", font_size=17),
            Dot(color=GREEN_C), Text("верхние 20% старта", font_size=17),
            Dot(color=YELLOW), Text("передача карточек", font_size=17),
        ).arrange(RIGHT, buff=0.14)
        legend.to_edge(DOWN)

        self.play(Create(ring), FadeIn(nodes), FadeIn(stats), FadeIn(legend), run_time=1.0)

        frame_indices = list(range(6, len(history), 6))
        if frame_indices[-1] != len(history) - 1:
            frame_indices.append(len(history) - 1)

        previous = 0
        for frame_idx in frame_indices:
            new_nodes = make_nodes(history[frame_idx])
            new_stats = make_stats(frame_idx)
            window_events = [
                event
                for event in result.events
                if previous <= event[0] < frame_idx
            ]
            window_events = sorted(window_events, key=lambda event: event[3], reverse=True)[:14]

            particles = VGroup()
            paths = []
            for _, winner, loser, transfer in window_events:
                start = positions[rank[loser]]
                end = positions[rank[winner]]
                dot = Dot(start, radius=0.032 + 0.008 * min(transfer, 5), color=YELLOW)
                particles.add(dot)
                paths.append(ArcBetweenPoints(start, end, angle=0.35))

            animations = [Transform(nodes, new_nodes), Transform(stats, new_stats)]
            animations.extend(MoveAlongPath(dot, path) for dot, path in zip(particles, paths))
            self.play(*animations, run_time=0.38, rate_func=linear)
            if len(particles) > 0:
                self.play(FadeOut(particles), run_time=0.08)
            previous = frame_idx

        self.wait(1.2)


class GroupMigrationScene(Scene):
    def construct(self):
        initial_counts = make_initial_distribution(seed=7)
        result = simulate_exchanges(initial_counts, steps=180, seed=11)
        history = result.history
        n = len(initial_counts)
        max_value = max(1.0, float(history.max()))

        title = Text("Дети переходят между группами по рыночной ценности", font_size=30)
        title.to_edge(UP)
        self.play(FadeIn(title), run_time=0.8)

        group_order = ["bottom_20", "middle_60", "top_20"]
        group_names = {
            "bottom_20": "нижние 20%",
            "middle_60": "средние 60%",
            "top_20": "верхние 20%",
        }
        group_centers = {
            "bottom_20": np.array([-4.25, -0.18, 0.0]),
            "middle_60": np.array([0.0, -0.18, 0.0]),
            "top_20": np.array([4.25, -0.18, 0.0]),
        }
        group_ring_radii = {
            "bottom_20": 1.08,
            "middle_60": 1.58,
            "top_20": 1.08,
        }

        def radius_for_value(value: float):
            return 0.035 + 0.16 * value / max_value

        def positions_for_snapshot(snapshot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            labels = group_labels(snapshot)
            positions = np.zeros((n, 3), dtype=float)
            for group in group_order:
                members = np.flatnonzero(labels == group)
                members = members[np.argsort(-snapshot[members])]
                count = max(len(members), 1)
                for slot, child in enumerate(members):
                    angle = TAU * slot / count + PI / 2
                    if group == "middle_60":
                        ring = group_ring_radii[group] * (0.58 if slot % 3 == 0 else 1.0)
                    else:
                        ring = group_ring_radii[group] * (0.62 if slot % 2 == 0 else 1.0)
                    positions[child] = group_centers[group] + ring * np.array([np.cos(angle), np.sin(angle), 0.0])
            return labels, positions

        def make_nodes(frame_idx: int):
            snapshot = history[frame_idx]
            labels, positions = positions_for_snapshot(snapshot)
            nodes = VGroup()
            for child in range(n):
                node = Circle(
                    radius=radius_for_value(float(snapshot[child])),
                    stroke_width=1.0,
                    stroke_color=WHITE,
                )
                node.set_fill(group_color(str(labels[child])), opacity=0.9)
                node.move_to(positions[child])
                nodes.add(node)
            return nodes

        def make_stats(frame_idx: int, movers_count: int):
            snapshot = history[frame_idx]
            text = (
                f"шаг {frame_idx:03d}   "
                f"Gini {gini(snapshot):.2f}   "
                f"перешли группу: {movers_count:02d}   "
                f"max value: {snapshot.max():.0f}"
            )
            label = Text(text, font_size=21)
            label.next_to(title, DOWN, buff=0.25)
            return label

        zones = VGroup()
        zone_labels = VGroup()
        for group in group_order:
            zone = Circle(
                radius=group_ring_radii[group] + 0.28,
                stroke_color=group_color(group),
                stroke_width=2,
                fill_opacity=0.04,
            )
            zone.move_to(group_centers[group])
            zones.add(zone)
            label = Text(group_names[group], font_size=21, color=group_color(group))
            label.next_to(zone, DOWN, buff=0.25)
            zone_labels.add(label)

        nodes = make_nodes(0)
        stats = make_stats(0, 0)
        self.play(Create(zones), FadeIn(zone_labels), FadeIn(nodes), FadeIn(stats), run_time=1.0)

        frame_indices = list(range(9, len(history), 9))
        if frame_indices[-1] != len(history) - 1:
            frame_indices.append(len(history) - 1)

        previous = 0
        previous_labels = group_labels(history[0])
        for frame_idx in frame_indices:
            current_labels = group_labels(history[frame_idx])
            movers = np.flatnonzero(previous_labels != current_labels)
            movers_count = len(movers)

            new_nodes = make_nodes(frame_idx)
            new_stats = make_stats(frame_idx, movers_count)

            mover_labels = VGroup()
            _, current_positions = positions_for_snapshot(history[frame_idx])
            important_movers = sorted(
                movers,
                key=lambda child: abs(history[frame_idx, child] - history[previous, child]),
                reverse=True,
            )[:8]
            for child in important_movers:
                label = Text(f"child_{child:02d}", font_size=14, color=YELLOW)
                label.move_to(current_positions[child] + np.array([0.0, 0.22, 0.0]))
                mover_labels.add(label)

            self.play(Transform(nodes, new_nodes), Transform(stats, new_stats), run_time=0.72, rate_func=smooth)
            if len(mover_labels) > 0:
                self.play(FadeIn(mover_labels), run_time=0.12)
                self.play(FadeOut(mover_labels), run_time=0.42)

            previous = frame_idx
            previous_labels = current_labels

        self.wait(1.2)
