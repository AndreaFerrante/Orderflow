"""
Markov Chain models for market prediction on tick-by-tick and compressed bar data.

This module provides:
- MarkovChainPredictor: Fixed-order Markov chains for UP/DOWN/FLAT state prediction
- AdaptiveMarkovChainPredictor: Variable-order chains with automatic order selection
- MultiFeatureHMM: Hidden Markov Models for multi-dimensional regime analysis

All classes support tick-by-tick and compressed bar (Volume/Range/Time) data.
"""

from collections import defaultdict
from typing import Optional, Tuple, Dict, List, Union
from .markov_utilities import *
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MarkovChainPredictor(object):
    """
    Fixed-order Markov Chain predictor for market state transitions.

    Predicts the next market state (UP/DOWN/FLAT) given a sequence of previous states.
    Uses fixed-order transitions with Laplace smoothing for stability.

    Parameters
    ----------
    order : int, default=1
        Order of the Markov chain (how many previous states to consider).

    Examples
    --------
    >>> prices = [100, 101, 102, 101, 100]
    >>> states = threshold_prices_states(prices)
    >>> predictor = MarkovChainPredictor(order=2)
    >>> predictor.fit(states)
    >>> next_prob = predictor.predict_distribution(['UP', 'DOWN'])
    """

    def __init__(self, order: int = 1):
        if order < 1:
            raise ValueError("Order must be >= 1.")
        self.order = order
        self.transition_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def fit(self, states: List[str]) -> None:
        """
        Fit the Markov chain on historical states.

        Parameters
        ----------
        states : List[str]
            Sequence of states (e.g., ['UP', 'DOWN', 'FLAT']).

        Raises
        ------
        ValueError
            If states has <= order elements.
        """
        if len(states) <= self.order:
            raise ValueError(f"Need >order ({self.order}) states, got {len(states)}.")

        # Calculate transition frequencies
        for i in range(self.order, len(states)):
            prev_states = tuple(states[i - self.order : i])
            current_state = states[i]
            self.transition_counts[prev_states][current_state] += 1

        # Calculate probabilities
        for prev_states, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.transition_probs[prev_states] = {
                s: c / total for s, c in next_counts.items()
            }

        logger.info(f"Markov(order={self.order}) fitted on {len(states)} states. "
                   f"Found {len(self.transition_probs)} unique patterns.")

    def predict_distribution(self, recent_states: List[str]) -> Dict[str, float]:
        """
        Get probability distribution for the next state.

        Parameters
        ----------
        recent_states : List[str]
            Last states (must have length >= order).

        Returns
        -------
        Dict[str, float]
            Probability distribution over next states.
        """
        if len(recent_states) < self.order:
            raise ValueError(f"Need >= {self.order} recent states, got {len(recent_states)}.")

        prev_states = tuple(recent_states[-self.order :])

        if prev_states in self.transition_probs:
            return self.transition_probs[prev_states]

        # Fallback: try shorter patterns
        for reduced_order in range(self.order - 1, 0, -1):
            reduced_prev_states = tuple(recent_states[-reduced_order:])
            candidates = {
                k: v for k, v in self.transition_probs.items()
                if k[-reduced_order:] == reduced_prev_states
            }

            if candidates:
                aggregated_counts = defaultdict(float)
                for dist in candidates.values():
                    for st, p in dist.items():
                        aggregated_counts[st] += p

                # Normalize
                total = sum(aggregated_counts.values())
                return {st: p / total for st, p in aggregated_counts.items()}

        # Uniform prior
        return {"UP": 1 / 3, "DOWN": 1 / 3, "FLAT": 1 / 3}

    def predict_next_state(self, recent_states: List[str]) -> str:
        """
        Predict the single most likely next state.

        Parameters
        ----------
        recent_states : List[str]
            Last states.

        Returns
        -------
        str
            Most probable next state.
        """
        dist = self.predict_distribution(recent_states)
        return max(dist.items(), key=lambda x: x[1])[0]


class AdaptiveMarkovChainPredictor(object):
    """
    Adaptive variable-order Markov Chain with automatic order selection.

    Automatically selects the best Markov chain order based on validation performance.
    Uses Laplace smoothing for stability and graceful degradation when data is sparse.

    Parameters
    ----------
    max_order : int, default=5
        Maximum order to test (1 to max_order).
    smoothing_alpha : float, default=0.1
        Laplace smoothing parameter (>0 for stability).

    Attributes
    ----------
    best_order : int
        Selected order after fitting (1 to max_order).
    """

    def __init__(self, max_order: int = 5, smoothing_alpha: float = 0.1):
        if max_order < 1:
            raise ValueError("max_order must be >= 1.")
        if smoothing_alpha <= 0:
            raise ValueError("smoothing_alpha must be > 0.")

        self.max_order = max_order
        self.smoothing_alpha = smoothing_alpha
        self.best_order = None
        self.transition_probs_by_order = {}
        self.fitted = False

    def _fit_single_order(self, states: List[str], order: int) -> Dict[Tuple[str, ...], Dict[str, float]]:
        """
        Fit di una catena di Markov di ordine fissato con Laplace smoothing.

        :param states: lista di stati
        :param order: ordine della catena
        :return: dizionario delle probabilità di transizione
        """
        transition_counts = defaultdict(lambda: defaultdict(float))
        # Possibili stati
        unique_states = list(set(states))

        # Conta transizioni
        for i in range(order, len(states)):
            prev_states = tuple(states[i - order:i])
            current_state = states[i]
            transition_counts[prev_states][current_state] += 1.0

        # Calcolo probabilità con smoothing
        transition_probs = {}
        for prev_states, next_counts in transition_counts.items():
            total = sum(next_counts.values()) + self.smoothing_alpha * len(unique_states)
            transition_probs[prev_states] = {
                s: (next_counts[s] + self.smoothing_alpha) / total for s in unique_states
            }
        return transition_probs

    def fit(self, states: List[str], validation_ratio: float = 0.2) -> None:
        """
        Fit the model by testing orders 1 to max_order.

        Selects the best order using log-likelihood on validation set.

        Parameters
        ----------
        states : List[str]
            Historical states.
        validation_ratio : float, default=0.2
            Fraction of data used for validation (0 < validation_ratio < 1).

        Raises
        ------
        ValueError
            If insufficient states.
        RuntimeError
            If no valid model found.
        """
        if len(states) < 10:
            raise ValueError(f"Need >= 10 states for fitting, got {len(states)}.")
        if not (0 < validation_ratio < 1):
            raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}.")

        n = len(states)
        val_size = int(n * validation_ratio)
        train_states = states[:-val_size]
        val_states = states[-val_size:]

        unique_states = list(set(states))
        best_ll = -np.inf
        best_order = None
        best_model = None

        # Test all orders
        for order in range(1, self.max_order + 1):
            if len(train_states) <= order:
                continue

            transition_probs = self._fit_single_order(train_states, order)

            # Evaluate on validation set
            ll = 0.0
            count = 0
            for i in range(order, len(val_states)):
                prev_states = tuple(val_states[i - order : i])
                current_state = val_states[i]

                if prev_states in transition_probs:
                    p = transition_probs[prev_states].get(current_state, 0.0)
                    ll += np.log(p) if p > 0 else np.log(1e-12)
                else:
                    ll += np.log(1.0 / len(unique_states))
                count += 1

            avg_ll = ll / max(count, 1)
            if avg_ll > best_ll:
                best_ll = avg_ll
                best_order = order
                best_model = transition_probs

        if best_model is None:
            raise RuntimeError("No valid model found. Insufficient data or all orders rejected.")

        self.best_order = best_order
        self.transition_probs_by_order = best_model
        self.fitted = True

        logger.info(f"AdaptiveMarkov fitted: best_order={best_order}, "
                   f"val_ll={best_ll:.4f}, patterns={len(best_model)}")

    def predict_distribution(self, recent_states: List[str]) -> Dict[str, float]:
        """
        Get probability distribution for next state using best order.

        Parameters
        ----------
        recent_states : List[str]
            Recent states (length >= best_order preferred).

        Returns
        -------
        Dict[str, float]
            Probability distribution over next states.

        Raises
        ------
        RuntimeError
            If model not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        needed_length = self.best_order
        if len(recent_states) < needed_length:
            # Fallback to uniform
            all_states = self._all_known_states()
            return {s: 1.0 / len(all_states) for s in all_states}

        prev_states = tuple(recent_states[-needed_length:])
        if prev_states in self.transition_probs_by_order:
            return self.transition_probs_by_order[prev_states]

        # Unknown pattern - uniform prior
        all_states = self._all_known_states()
        return {s: 1.0 / len(all_states) for s in all_states}

    def predict_next_state(self, recent_states: List[str]) -> str:
        """
        Predict single most likely next state.

        Parameters
        ----------
        recent_states : List[str]
            Recent states.

        Returns
        -------
        str
            Most probable next state.
        """
        dist = self.predict_distribution(recent_states)
        return max(dist.items(), key=lambda x: x[1])[0]

    def _all_known_states(self) -> List[str]:
        """Get all states seen during training."""
        all_states = set()
        for dist in self.transition_probs_by_order.values():
            all_states.update(dist.keys())
        return list(all_states)


class MultiFeatureHMM(object):
    """
    Hidden Markov Model for multi-dimensional market data.

    Performs regime detection and smoothing using Gaussian HMM.
    Supports Viterbi decoding and posterior probability extraction.

    Parameters
    ----------
    model : hmm.GaussianHMM, optional
        Pre-initialized HMM. If None, must be set before fitting.

    Attributes
    ----------
    fitted : bool
        Whether the model has been fitted.
    """

    def __init__(self, model: Optional[hmm.GaussianHMM] = None):
        self.model = model
        self.fitted = False if model is None else True

    def fit(self, data: np.ndarray) -> None:
        """
        Fit the HMM on multi-dimensional data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If no model provided.
        """
        if self.model is None:
            raise RuntimeError("No model set. Provide model in __init__().")
        self.model.fit(data)
        self.fitted = True
        logger.info(f"HMM fitted: {self.model.n_components} states, "
                   f"log-likelihood={self.model.score(data):.2f}")

    def predict_states(self, data: np.ndarray) -> np.ndarray:
        """
        Decode hidden states using Viterbi algorithm.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Sequence of hidden states.

        Raises
        ------
        RuntimeError
            If model not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(data)

    def predict_proba_states(self, data: np.ndarray) -> np.ndarray:
        """
        Get posterior state probabilities.

        Uses forward-backward smoothing to get probability over all states
        for each observation.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_states) posterior probabilities.

        Raises
        ------
        RuntimeError
            If model not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(data)

    def score(self, data: np.ndarray) -> float:
        """
        Compute log-likelihood of data under model.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_features).

        Returns
        -------
        float
            Log-likelihood score.

        Raises
        ------
        RuntimeError
            If model not fitted.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.score(data)


# ============================================================================
# Helper Functions for Compressed Bars
# ============================================================================


def get_states_from_ohlc(df: pd.DataFrame, method: str = "close") -> List[str]:
    """
    Generate UP/DOWN/FLAT states from OHLC bar data (tick or compressed bars).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain relevant OHLC columns depending on method.
    method : {"close", "hl_range", "oc_range"}
        - "close": Compare each close to the previous close (adaptive threshold).
        - "hl_range": Classify each bar as UP/DOWN/FLAT by whether
          (High - Low) is above or below the rolling average range.
        - "oc_range": Classify each bar directly by sign of (Close - Open).

    Returns
    -------
    List[str]
        States corresponding to bar movements (length = len(df) for oc_range/hl_range,
        len(df) - 1 for close).
    """
    if method == "close":
        if "Close" not in df.columns:
            raise ValueError("'Close' column required for method='close'.")
        return adaptive_threshold_prices_states(df["Close"].tolist(), window=20)

    elif method == "hl_range":
        # hl_range: bar is UP if its range is above rolling-mean range, DOWN if below.
        # This captures expansion vs contraction in compressed bars.
        if not all(c in df.columns for c in ["High", "Low"]):
            raise ValueError("'High', 'Low' columns required for method='hl_range'.")
        ranges = (df["High"] - df["Low"]).values
        window = 20
        states: List[str] = []
        for i in range(len(ranges)):
            start = max(0, i - window)
            avg_range = np.mean(ranges[start:i]) if i > 0 else 0.0
            if ranges[i] > avg_range:
                states.append("UP")
            elif ranges[i] < avg_range:
                states.append("DOWN")
            else:
                states.append("FLAT")
        return states

    elif method == "oc_range":
        # oc_range: directly classify each bar by sign of (Close - Open).
        # No diff needed — the change IS the signal.
        if not all(c in df.columns for c in ["Open", "Close"]):
            raise ValueError("'Open', 'Close' columns required for method='oc_range'.")
        changes = (df["Close"] - df["Open"]).values
        states = []
        for chg in changes:
            if chg > 0:
                states.append("UP")
            elif chg < 0:
                states.append("DOWN")
            else:
                states.append("FLAT")
        return states

    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'close', 'hl_range', or 'oc_range'.")


def predict_bar_state(
    df: pd.DataFrame,
    predictor: Union[MarkovChainPredictor, AdaptiveMarkovChainPredictor],
    lookback: Optional[int] = None,
    method: str = "close",
) -> Tuple[str, Dict[str, float]]:
    """
    Predict next bar state for compressed bar data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC bar data.
    predictor : MarkovChainPredictor or AdaptiveMarkovChainPredictor
        Fitted predictor.
    lookback : int, optional
        How many recent bars to use. Defaults to predictor.order for
        MarkovChainPredictor, or predictor.best_order for
        AdaptiveMarkovChainPredictor.
    method : str
        State generation method (see get_states_from_ohlc).

    Returns
    -------
    Tuple[str, Dict[str, float]]
        (Predicted state, probability distribution)
    """
    if lookback is None:
        if isinstance(predictor, AdaptiveMarkovChainPredictor):
            if not predictor.fitted:
                raise RuntimeError("AdaptiveMarkovChainPredictor not fitted. Call fit() first.")
            lookback = predictor.best_order
        else:
            lookback = predictor.order

    states = get_states_from_ohlc(df, method=method)
    if len(states) < lookback:
        raise ValueError(
            f"Not enough bars to extract {lookback} recent states (got {len(states)}).")

    recent = states[-lookback:]
    pred_state = predictor.predict_next_state(recent)
    prob_dist = predictor.predict_distribution(recent)

    return pred_state, prob_dist