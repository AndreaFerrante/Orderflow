from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

class MarkovChainPredictor:
    
    """
    This class is used to create a small Markov Model for any stock price.
    One can review this class ot predict the next price movement given N previous ones.
    ultime N osservazioni.
    """
    
    def __init__(self, order: int = 1):
        
        if order < 1:
            raise ValueError("The order of the cahin must be bigger than one.")
        self.order = order
        self.transition_counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def fit(self, states: List[str]) -> None:
        
        if len(states) <= self.order:
            raise ValueError("The number of the states must be bigger than the order of the chain.")
        
        # Calculation for the transition frequencies for the last N states.
        for i in range(self.order, len(states)):
            prev_states = tuple(states[i-self.order:i])
            current_state = states[i]
            self.transition_counts[prev_states][current_state] += 1

        # Calculation for transitions probabilities
        for prev_states, next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            self.transition_probs[prev_states] = {s: c/total for s, c in next_counts.items()}

    def predict_distribution(self, recent_states: List[str]) -> Dict[str, float]:
        
        if len(recent_states) < self.order:
            raise ValueError(f"The list of all the last states should have at least {self.order} elements.")

        prev_states = tuple(recent_states[-self.order:])

        if prev_states in self.transition_probs:
            return self.transition_probs[prev_states]

        for reduced_order in range(self.order-1, 0, -1):
            
            reduced_prev_states = tuple(recent_states[-reduced_order:])
            candidates = {k: v for k, v in self.transition_probs.items() if k[-reduced_order:] == reduced_prev_states}
            
            if candidates:
                
                aggregated_counts = defaultdict(float)
                for dist in candidates.values():
                    for st, p in dist.items():
                        aggregated_counts[st] += p
                        
                # Normalize ...
                total = sum(aggregated_counts.values())
                return {st: p/total for st, p in aggregated_counts.items()}

        # If we have no information, we assume uniform distribuition among the three states ...
        return {"UP": 1/3, "DOWN": 1/3, "FLAT": 1/3}

    def predict_next_state(self, recent_states: List[str]) -> str:
        
        dist = self.predict_distribution(recent_states)
        # Select max probability state !
        return max(dist.items(), key=lambda x: x[1])[0]


def convert_prices_to_states(prices: List[float], threshold: float = 1e-8) -> List[str]:
    
    if len(prices) < 2:
        raise ValueError("Two prices min are needed to obtain the states.")

    states = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > threshold:
            states.append("UP")
        elif diff < -threshold:
            states.append("DOWN")
        else:
            states.append("FLAT")
    return states


def adaptive_threshold_states(prices: List[float], window: int = 20, base_threshold: float = 0.01) -> List[str]:
    """
    Converte una sequenza di prezzi in stati UP/DOWN/FLAT con threshold adattivo.
    Il threshold viene scalato con la volatilità (deviazione standard) dei rendimenti
    su una finestra mobile.

    :param prices: Lista di prezzi
    :param window: Finestra per calcolo volatilità
    :param base_threshold: threshold base che verrà moltiplicato per la volatilità
    :return: lista di stati ('UP', 'DOWN', 'FLAT')
    """
    if len(prices) < 2:
        raise ValueError("Servono almeno due prezzi per calcolare gli stati.")

    returns = np.diff(prices)
    states = []
    for i in range(1, len(prices)):
        # Calcolo volatilità locale
        start_idx = max(0, i - window)
        local_returns = returns[start_idx:i] if i > 0 else returns[:1]
        vol = np.std(local_returns) if len(local_returns) > 1 else 1e-8
        threshold = base_threshold * max(vol, 1e-8)
        
        diff = prices[i] - prices[i-1]
        if diff > threshold:
            states.append("UP")
        elif diff < -threshold:
            states.append("DOWN")
        else:
            states.append("FLAT")
    return states


if __name__ == "__main__":
    
    historical_prices = [100.0, 100.5, 100.7, 100.7, 100.6, 100.65, 100.9, 100.85, 100.85, 101.0]
    states = convert_prices_to_states(historical_prices, threshold=0.01)

    predictor = MarkovChainPredictor(order=2)
    predictor.fit(states)
    
    recent_states = states[-2:]
    prediction    = predictor.predict_next_state(recent_states)
    dist          = predictor.predict_distribution(recent_states)

    print("Ultimi stati:", recent_states)
    print("Predizione del prossimo stato:", prediction)
    print("Distribuzione di probabilità:", dist)
