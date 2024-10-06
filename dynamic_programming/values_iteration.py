import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for i in range(max_iter):
        new_values = np.copy(values)
        for state in range(mdp.observation_space.n):
            action_values = []
            for action in range(mdp.action_space.n):
                q_value = 0
                next_state, reward, done = mdp.P[state][action]
                q_value += reward + gamma * (0 if done else values[next_state])
                action_values.append(q_value)
            new_values[state] = max(action_values)
        if np.allclose(values, new_values):
            break
        values = new_values
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    terminal_states = {
        (0, 3): 0.0,
        (1, 3): 0.0,
    }

    for _ in range(max_iter):
        delta = 0
        old_values = values.copy()

        for row in range(4):
            for col in range(4):
                if env.grid[row, col] == "W":
                    values[row, col] = 0.0
                    continue

                # Keep terminal states at their fixed values
                if (row, col) in terminal_states:
                    values[row, col] = terminal_states[(row, col)]
                    continue

                env.current_position = (row, col)
                max_value = float("-inf")

                for action in range(env.action_space.n):
                    next_state, reward, done, _ = env.step(action, make_move=False)
                    value = reward
                    if not done:
                        next_row, next_col = next_state
                        value += gamma * old_values[next_row, next_col]
                    max_value = max(max_value, value)

                values[row, col] = max_value
                delta = max(delta, abs(values[row, col] - old_values[row, col]))

        if delta < theta:
            break

    return values
    # END SOLUTION


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    terminal_states = {
        (0, 3): 0.0,  # Positive terminal
        (1, 3): 0.0,  # Negative terminal
    }

    for _ in range(max_iter):
        delta = 0
        old_values = values.copy()

        for row in range(4):
            for col in range(4):
                if env.grid[row, col] == "W":
                    values[row, col] = 0.0
                    continue

                if (row, col) in terminal_states:
                    values[row, col] = terminal_states[(row, col)]
                    continue

                env.current_position = (row, col)
                max_value = float("-inf")

                for action in range(env.action_space.n):
                    value = 0
                    next_states = env.get_next_states(action)

                    for next_state, reward, prob, done, _ in next_states:
                        if done:
                            value += prob * reward
                        else:
                            next_row, next_col = next_state
                            value += prob * (
                                reward + gamma * old_values[next_row, next_col]
                            )

                    max_value = max(max_value, value)

                values[row, col] = max_value
                delta = max(delta, abs(values[row, col] - old_values[row, col]))

        if delta < theta:
            if gamma == 1.0:
                mask = values > 0.999999
                values[mask] = 1.0
            break

    return values
