from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    initial: np.ndarray
    history: np.ndarray
    events: list[tuple[int, int, int, int]]
    initial_order: np.ndarray
    opportunity_history: np.ndarray


def make_initial_distribution(
    n_children: int = 100,
    mean_cards: float = 20.0,
    std_cards: float = 12.0,
    seed: int = 7,
) -> np.ndarray:
    """Normal-like random start with a hard zero floor."""
    rng = np.random.default_rng(seed)
    cards = np.rint(rng.normal(mean_cards, std_cards, n_children)).astype(int)
    return np.clip(cards, 0, None)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if np.any(values < 0):
        values = values - values.min()
    total = values.sum()
    if total == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_values) / total - (n + 1)) / n)


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    total = values.sum()
    x = np.linspace(0, 1, values.size + 1)
    if total == 0:
        return x, np.zeros(values.size + 1)
    y = np.concatenate([[0], np.cumsum(values) / total])
    return x, y


def group_indices(initial: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(initial)
    n = len(initial)
    return {
        "bottom_20": order[: n // 5],
        "middle_60": order[n // 5 : 4 * n // 5],
        "top_20": order[4 * n // 5 :],
    }


def group_shares(history: np.ndarray, initial: np.ndarray) -> dict[str, np.ndarray]:
    groups = group_indices(initial)
    totals = history.sum(axis=1)
    shares: dict[str, np.ndarray] = {}
    for name, idx in groups.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            shares[name] = np.where(totals > 0, history[:, idx].sum(axis=1) / totals, 0)
    return shares


def simulate_exchanges(
    initial: np.ndarray,
    steps: int = 180,
    card_lead_probability: float = 0.006,
    stake_fraction: float = 0.08,
    max_transfer: int = 4,
    spread_to_cards: float = 0.35,
    seed: int = 11,
) -> SimulationResult:
    """Simulate exchanges driven by unequal opportunity counts.

    The important modeling choice is that opportunities are created by the
    cards themselves. Every currently held card has the same independent chance
    to produce a trade lead. Richer children do not get a better per-deal draw,
    and participants are not selected by a weighted lottery. They simply own
    more cards, so more independent leads can appear in their portfolio, and
    their larger budget lets them take larger deals.

    A positive spread means the initiating child found a deal worth taking.
    The transfer is conserved: one child gains exactly what another loses.
    """
    rng = np.random.default_rng(seed)
    holdings = np.asarray(initial, dtype=int).copy()
    n = holdings.size
    history = [holdings.copy()]
    opportunity_history = []
    events: list[tuple[int, int, int, int]] = []

    for step in range(steps):
        # Each card independently creates a lead. This avoids directly choosing
        # children with a probability proportional to their card count.
        attempts = rng.binomial(holdings, card_lead_probability)
        opportunity_history.append(attempts.copy())
        initiators = np.repeat(np.arange(n), attempts)
        rng.shuffle(initiators)

        for initiator in initiators:
            if holdings[initiator] <= 0:
                continue

            counterparties = np.flatnonzero((holdings > 0) & (np.arange(n) != initiator))
            if counterparties.size == 0:
                break

            counterparty = int(rng.choice(counterparties))

            # Same per-opportunity distribution for everyone.
            spread = rng.normal(loc=0.0, scale=1.0)
            if spread <= 0:
                continue

            affordable_size = int(np.floor(holdings[initiator] * stake_fraction)) + 1
            max_size = min(affordable_size, int(holdings[counterparty]), max_transfer)
            if max_size <= 0:
                continue

            deal_size = int(rng.integers(1, max_size + 1))
            transfer = int(np.ceil(min(max_size, spread_to_cards * spread * deal_size)))
            transfer = max(1, min(transfer, int(holdings[counterparty])))

            holdings[initiator] += transfer
            holdings[counterparty] -= transfer
            events.append((step, int(initiator), counterparty, transfer))

        history.append(holdings.copy())

    return SimulationResult(
        initial=np.asarray(initial, dtype=int),
        history=np.asarray(history, dtype=int),
        events=events,
        initial_order=np.argsort(initial),
        opportunity_history=np.asarray(opportunity_history, dtype=int),
    )


def describe_distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    return {
        "total": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "zeros": float(np.sum(values == 0)),
        "gini": gini(values),
    }
