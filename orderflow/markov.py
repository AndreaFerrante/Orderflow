from collections import defaultdict
from typing import Optional, Tuple, Dict
from .markov_utilities import *
import warnings


warnings.filterwarnings("ignore")


class MarkovChainPredictor(object):
    
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


class AdaptiveMarkovChainPredictor(object):

    """
    Modello di previsione basato su catene di Markov di ordine variabile, con:
    - Selezione automatica dell'ordine (fino a max_order)
    - Laplace smoothing dei conteggi di transizione
    - Threshold adattivo per la definizione degli stati
    - Fallback bayesiano in caso di mancanza di dati
    """

    def __init__(self, max_order: int = 5, smoothing_alpha: float = 0.1):
        """
        Inizializza il predittore Markoviano con possibilità di ordini da 1 a max_order.

        :param max_order: Ordine massimo della catena di Markov da considerare.
        :param smoothing_alpha: Parametro di smoothing Laplaciano (>0).
        """
        if max_order < 1:
            raise ValueError("max_order deve essere >= 1.")
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
        Fit del modello testando ordini da 1 a max_order e scegliendo quello con migliore
        performance su un set di validazione interno.

        :param states: Lista di stati storici.
        :param validation_ratio: Percentuale di dati da usare come validazione.
        """

        if len(states) < 10:
            raise ValueError("Troppi pochi stati per il fitting.")
        n = len(states)
        val_size = int(n * validation_ratio)
        train_states = states[:-val_size]
        val_states = states[-val_size:]

        # Un elenco degli stati unici per misurare la log-likelihood
        unique_states = list(set(states))

        best_ll = -np.inf
        best_order = None
        best_model = None

        # Proviamo tutti gli ordini
        for order in range(1, self.max_order + 1):
            if len(train_states) <= order:
                continue
            transition_probs = self._fit_single_order(train_states, order)

            # Calcoliamo la log-likelihood sul validation set
            # partendo da order, per ogni indice calcoliamo P(state|prev_states)
            ll = 0.0
            count = 0
            for i in range(order, len(val_states)):
                prev_states = tuple(val_states[i - order:i])
                current_state = val_states[i]
                if prev_states in transition_probs:
                    p = transition_probs[prev_states].get(current_state, 0.0)
                    if p > 0:
                        ll += np.log(p)
                    else:
                        # Se p=0, penalità severa
                        ll += np.log(1e-12)
                else:
                    # Nessuna informazione: uso prior uniforme
                    ll += np.log(1.0 / len(unique_states))
                count += 1

            avg_ll = ll / max(count, 1)
            if avg_ll > best_ll:
                best_ll = avg_ll
                best_order = order
                best_model = transition_probs

        if best_model is None:
            raise RuntimeError("Non è stato trovato nessun modello valido. Forse troppi pochi dati.")

        self.best_order = best_order
        self.transition_probs_by_order = best_model
        self.fitted = True

    def predict_distribution(self, recent_states: List[str]) -> Dict[str, float]:
        """
        Data una lista di stati recenti, restituisce la distribuzione di probabilità sul prossimo stato.
        Utilizza il best_order trovato e, se non trova match, usa fallback bayesiano.

        :param recent_states: Lista degli ultimi stati noti
        :return: Dizionario {stato: probabilità}
        """
        if not self.fitted:
            raise RuntimeError("Modello non fittato. Eseguire fit() prima di predict.")

        needed_length = self.best_order
        if len(recent_states) < needed_length:
            # Troppo pochi stati: fallback
            all_states = self._all_known_states()
            return {s: 1.0 / len(all_states) for s in all_states}

        prev_states = tuple(recent_states[-needed_length:])
        if prev_states in self.transition_probs_by_order:
            return self.transition_probs_by_order[prev_states]

        # Nessuna informazione su quella particolare combinazione di stati passati.
        # Fallback: uso una media o prior uniforme
        all_states = self._all_known_states()
        return {s: 1.0 / len(all_states) for s in all_states}

    def predict_next_state(self, recent_states: List[str]) -> str:
        """
        Predice lo stato più probabile.

        :param recent_states: Lista degli ultimi stati
        :return: 'UP', 'DOWN', oppure 'FLAT'
        """
        dist = self.predict_distribution(recent_states)
        return max(dist.items(), key=lambda x: x[1])[0]

    def _all_known_states(self) -> List[str]:
        """
        Ritorna la lista di tutti gli stati visti almeno una volta in training.
        """
        all_states = set()
        for dist in self.transition_probs_by_order.values():
            all_states.update(dist.keys())
        return list(all_states)


class MultiFeatureHMM(object):

    """
    Class to provide:
      - Fit
      - Decoding Viterbi and posterior probability (predict_proba)
      - Estimation of next feature vector
    """

    def __init__(self, model: Optional[hmm.GaussianHMM] = None):

        """
        Initialize our HMM.
        """

        self.model = model
        self.fitted = False if model is None else True

    def fit(self, data: np.ndarray) -> None:

        if self.model is None:
            raise RuntimeError("Mo model present !")
        self.model.fit(data)
        self.fitted = True

    def predict_states(self, data: np.ndarray) -> np.ndarray:

        """
        It returns a sequence of states (Viterbi decoding).
        """

        if not self.fitted:
            raise RuntimeError("Modello not fitted.")

        return self.model.predict(data)

    def predict_proba_states(self, data: np.ndarray) -> np.ndarray:

        """
        Returns the posterior probability matrix (filtering and smoothing).
        In hmmlearn, predict_proba() actually performs a step-by-step 'maximum a posteriori',
        but it provides the best guess of how the probability is distributed over the states
        with each observation.
        """

        if not self.fitted:
            raise RuntimeError("Modello non fittato.")

        # In hmmlearn it is called "predict_proba()" but the doc defines it as "predict_proba(X)"
        # returning the estimate of the posterior probabilities of the hidden state for each sample.

        return self.model.predict_proba(data)

    def score(self, data: np.ndarray) -> float:

        """
        It returns the log-likelihood of the model over the data.
        """

        if not self.fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.score(data)


def run_adaptive_markov_chain_predictor():

    prices = [100.0]

    for _ in range(100000):
        prices.append(prices[-1] + np.random.normal(0.01, 2.5))

    states = adaptive_threshold_prices_states(prices, window=3)
    predictor = AdaptiveMarkovChainPredictor(max_order=10, smoothing_alpha=1000)
    predictor.fit(states, validation_ratio=0.2)

    # recent_states = states[-predictor.best_order:]
    recent_states = [np.random.choice(['UP', 'DOWN', 'FLAT']) for x in range(predictor.best_order)]
    prediction = predictor.predict_next_state(recent_states)
    dist = predictor.predict_distribution(recent_states)

    print("Distribuzione stati: \n", pd.Series(states).value_counts())
    print("Stati individuati:", list(set(states)))
    print("Ordine scelto:", predictor.best_order)
    print("Ultimi stati:", recent_states)
    print("Predizione del prossimo stato:", prediction)
    print("Distribuzione di probabilità:", dist)


def run_multi_feature_hmm():

    # 1. Simuliamo dati di mercato (prezzi, volume)
    df_market = simulate_market_data(num_steps=500, seed=42)

    # 2. Calcoliamo le feature ingegnerizzate
    df_features = compute_df_features(df_market, window_volatility=20, window_slope=5)
    # Selezioniamo le colonne di input per l'HMM (multidimensionali)
    # Esempio: 'return', 'volatility', 'slope', 'log_volume'
    features_array = df_features[['return', 'volatility', 'slope', 'log_volume']].values

    # 3. Selezione automatica del numero di stati nascosti
    #    Proviamo un range da 2 a 5 stati, usando BIC
    candidate_states = [2, 3, 4, 5]
    best_hmm = select_best_hmm_model(
        data=features_array,
        n_states_range=candidate_states,
        covariance_type='full',
        criterion='bic',
        random_state=123
    )

    # 4. Creiamo il wrapper di classe e confermiamo che è fittato
    multi_hmm = MultiFeatureHMM(model=best_hmm)
    # Fit finale (anche se best_hmm è già fit, in genere)
    multi_hmm.fit(features_array)

    # 5. Prediciamo gli stati con Viterbi
    states_seq = multi_hmm.predict_states(features_array)
    print("Stati nascosti (Viterbi) per ogni osservazione:\n", states_seq)

    # 6. Calcoliamo la matrice di probabilità a posteriori (filtraggio & smoothing)
    posterior_probs = multi_hmm.predict_proba_states(features_array)
    print("\nMatrice di probabilità a posteriori (prime 10 righe):")
    print(posterior_probs[:10])

    # 7. Score del modello
    loglike = multi_hmm.score(features_array)
    print(f"\nLog-likelihood del modello sul dataset: {loglike:.2f}")


if __name__ == "__main__":

    run_adaptive_markov_chain_predictor()
    run_multi_feature_hmm()