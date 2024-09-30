# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    if n == 0:
        return 1
    if n == 1 or n == 2:
        return 0
    if n == 3:
        return 1

    # Initialize the base cases
    dp = [0] * (n + 1)
    dp[0] = 1  # 3x0 case
    dp[1] = 0  # 3x1 case
    dp[2] = 0  # 3x2 case
    dp[3] = 1  # 3x3 case

    # Fill the dp array using the recurrence relation
    for i in range(4, n + 1):
        dp[i] = dp[i - 1] + dp[i - 3]

    return dp[n]
    # END SOLUTION