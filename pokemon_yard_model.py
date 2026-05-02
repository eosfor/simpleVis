from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    # Market value of each child's collection.
    initial: np.ndarray
    history: np.ndarray

    # Number of physical cards each child holds.
    initial_card_counts: np.ndarray
    card_count_history: np.ndarray

    # Subjective utility of each child's collection.
    utility_history: np.ndarray

    # Market-value flow events: (step, receiver, sender, value_delta).
    events: list[tuple[int, int, int, float]]

    # Net physical-card flow events: (step, receiver, sender, card_delta).
    card_events: list[tuple[int, int, int, int]]

    initial_order: np.ndarray
    opportunity_history: np.ndarray
    card_values: np.ndarray
    card_kinds: np.ndarray
    preference_weights: np.ndarray


def make_initial_distribution(
    n_children: int = 100,
    mean_cards: float = 20.0,
    std_cards: float = 12.0,
    seed: int = 7,
) -> np.ndarray:
    """Normal-like random start with a hard zero floor.

    This is the physical number of cards per child. The market value of those
    cards is generated inside simulate_exchanges because each individual card
    also has rarity, type, and price.
    """
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


def _make_cards(
    total_cards: int,
    n_kinds: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rarity = rng.choice(
        np.array([0, 1, 2, 3]),
        size=total_cards,
        p=np.array([0.72, 0.20, 0.07, 0.01]),
    )
    rarity_base = np.array([1.0, 3.0, 12.0, 55.0])
    kind = rng.integers(0, n_kinds, size=total_cards)
    values = rarity_base[rarity] * rng.lognormal(mean=0.0, sigma=0.28, size=total_cards)
    return values, kind


def _make_preferences(
    n_children: int,
    n_kinds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    preferences = rng.lognormal(mean=0.0, sigma=0.38, size=(n_children, n_kinds))
    favorites = rng.integers(0, n_kinds, size=n_children)
    preferences[np.arange(n_children), favorites] *= rng.uniform(1.45, 2.55, size=n_children)
    return preferences


def _collection_metrics(
    ownership: np.ndarray,
    card_values: np.ndarray,
    card_utility: np.ndarray,
    n_children: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    market = np.bincount(ownership, weights=card_values, minlength=n_children).astype(float)
    counts = np.bincount(ownership, minlength=n_children).astype(int)
    utility = np.zeros(n_children, dtype=float)
    for child in range(n_children):
        owned = np.flatnonzero(ownership == child)
        if owned.size > 0:
            utility[child] = float(card_utility[child, owned].sum())
    return market, counts, utility


def _choose_counterparty(
    child: int,
    active_children: np.ndarray,
    rng: np.random.Generator,
    local_radius: int,
    local_trade_probability: float,
) -> int | None:
    candidates = active_children[active_children != child]
    if candidates.size == 0:
        return None

    if rng.random() < local_trade_probability:
        n = active_children.max() + 1
        distance = np.abs(candidates - child)
        circular_distance = np.minimum(distance, n - distance)
        local = candidates[circular_distance <= local_radius]
        if local.size > 0:
            return int(rng.choice(local))

    return int(rng.choice(candidates))


def _find_voluntary_bundle_trade(
    owner: int,
    bidder: int,
    ownership: np.ndarray,
    card_values: np.ndarray,
    card_utility: np.ndarray,
    rng: np.random.Generator,
    max_bundle_size: int,
    search_width: int,
    min_utility_gain: float,
    min_owner_market_gain: float,
) -> tuple[int, np.ndarray] | None:
    owner_cards = np.flatnonzero(ownership == owner)
    bidder_cards = np.flatnonzero(ownership == bidder)
    if owner_cards.size == 0 or bidder_cards.size == 0:
        return None

    targets = rng.choice(
        owner_cards,
        size=min(search_width, owner_cards.size),
        replace=False,
    )
    offer_pool = rng.choice(
        bidder_cards,
        size=min(max(search_width * 2, max_bundle_size), bidder_cards.size),
        replace=False,
    )

    best: tuple[float, int, np.ndarray] | None = None
    for target in targets:
        # A lead from an owned card means another child wants it. The owner has
        # a liquid asset and looks for a bundle the bidder can offer in return.
        offer_scores = card_utility[owner, offer_pool] - card_utility[bidder, offer_pool]
        ranked_offer_pool = offer_pool[np.argsort(offer_scores)[::-1]]

        for bundle_size in range(1, min(max_bundle_size, ranked_offer_pool.size) + 1):
            offer = ranked_offer_pool[:bundle_size]
            owner_gain = card_utility[owner, offer].sum() - card_utility[owner, target]
            bidder_gain = card_utility[bidder, target] - card_utility[bidder, offer].sum()
            if owner_gain <= min_utility_gain or bidder_gain <= min_utility_gain:
                continue

            market_gain = card_values[offer].sum() - card_values[target]
            if market_gain < min_owner_market_gain:
                continue
            score = owner_gain + 0.22 * market_gain
            if best is None or score > best[0]:
                best = (float(score), int(target), offer.copy())

    if best is None:
        return None
    return best[1], best[2]


def simulate_exchanges(
    initial_card_counts: np.ndarray,
    steps: int = 180,
    card_lead_probability: float = 0.008,
    value_lead_strength: float = 0.85,
    max_bundle_size: int = 4,
    search_width: int = 10,
    min_utility_gain: float = 0.12,
    min_owner_market_gain: float = 0.0,
    n_kinds: int = 6,
    local_radius: int = 12,
    local_trade_probability: float = 0.72,
    seed: int = 11,
) -> SimulationResult:
    """Simulate voluntary trades with unequal opportunity counts.

    Each physical card independently creates trade leads. More valuable cards
    create more leads because they are more liquid: more children want them.
    A child with a better portfolio does not get a better per-deal probability;
    they see more possible deals and can assemble larger bundles. Every
    executed trade must improve subjective utility for both children, but the
    market value of their collections can still become more concentrated over
    time.
    """
    rng = np.random.default_rng(seed)
    initial_card_counts = np.asarray(initial_card_counts, dtype=int)
    n_children = initial_card_counts.size
    total_cards = int(initial_card_counts.sum())

    card_values, card_kinds = _make_cards(total_cards, n_kinds, rng)
    mean_card_value = max(float(card_values.mean()), 1e-9)
    card_lead_probabilities = card_lead_probability * (
        (1.0 - value_lead_strength)
        + value_lead_strength * np.sqrt(card_values / mean_card_value)
    )
    card_lead_probabilities = np.clip(card_lead_probabilities, 0.0, 0.35)
    preference_weights = _make_preferences(n_children, n_kinds, rng)

    idiosyncratic_fit = rng.lognormal(mean=0.0, sigma=0.16, size=(n_children, total_cards))
    card_utility = (
        np.power(card_values, 0.72)[None, :]
        * preference_weights[:, card_kinds]
        * idiosyncratic_fit
    )

    ownership = np.repeat(np.arange(n_children), initial_card_counts)
    rng.shuffle(ownership)

    initial_market, initial_counts, initial_utility = _collection_metrics(
        ownership,
        card_values,
        card_utility,
        n_children,
    )
    market_history = [initial_market.copy()]
    count_history = [initial_counts.copy()]
    utility_history = [initial_utility.copy()]
    opportunity_history = []
    events: list[tuple[int, int, int, float]] = []
    card_events: list[tuple[int, int, int, int]] = []

    for step in range(steps):
        current_counts = np.bincount(ownership, minlength=n_children).astype(int)
        lead_flags = rng.binomial(1, card_lead_probabilities).astype(int)
        attempts = np.bincount(ownership, weights=lead_flags, minlength=n_children).astype(int)
        opportunity_history.append(attempts.copy())

        initiators = np.repeat(np.arange(n_children), attempts)
        rng.shuffle(initiators)

        for initiator in initiators:
            active_children = np.flatnonzero(np.bincount(ownership, minlength=n_children) > 0)
            counterparty = _choose_counterparty(
                int(initiator),
                active_children,
                rng,
                local_radius,
                local_trade_probability,
            )
            if counterparty is None:
                continue

            current_count = int(np.sum(ownership == initiator))
            positive_mean_count = max(float(current_counts[current_counts > 0].mean()), 1.0)
            dynamic_bundle_size = min(max_bundle_size, max(1, 1 + current_count // 12))
            dynamic_search_width = max(
                4,
                int(round(search_width * np.sqrt(max(current_count, 1) / positive_mean_count))),
            )

            trade = _find_voluntary_bundle_trade(
                int(initiator),
                counterparty,
                ownership,
                card_values,
                card_utility,
                rng,
                dynamic_bundle_size,
                dynamic_search_width,
                min_utility_gain,
                min_owner_market_gain,
            )
            if trade is None:
                continue

            target, offer = trade
            market_delta = float(card_values[offer].sum() - card_values[target])
            count_delta = int(offer.size - 1)

            ownership[target] = counterparty
            ownership[offer] = initiator

            if market_delta > 0:
                events.append((step, int(initiator), counterparty, market_delta))
            elif market_delta < 0:
                events.append((step, counterparty, int(initiator), -market_delta))

            if count_delta > 0:
                card_events.append((step, int(initiator), counterparty, count_delta))
            elif count_delta < 0:
                card_events.append((step, counterparty, int(initiator), -count_delta))

        market, counts, utility = _collection_metrics(
            ownership,
            card_values,
            card_utility,
            n_children,
        )
        market_history.append(market.copy())
        count_history.append(counts.copy())
        utility_history.append(utility.copy())

    market_history_array = np.asarray(market_history, dtype=float)
    return SimulationResult(
        initial=initial_market,
        history=market_history_array,
        initial_card_counts=initial_counts,
        card_count_history=np.asarray(count_history, dtype=int),
        utility_history=np.asarray(utility_history, dtype=float),
        events=events,
        card_events=card_events,
        initial_order=np.argsort(initial_market),
        opportunity_history=np.asarray(opportunity_history, dtype=int),
        card_values=card_values,
        card_kinds=card_kinds,
        preference_weights=preference_weights,
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
