# utils/metrics.py

def levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n]


def character_error_rate(reference, hypothesis):
    if len(reference) == 0:
        return 0.0

    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def word_error_rate(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)