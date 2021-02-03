def p(s1, s2, cache):
    if s2[i] == 0 or s2[j] == 0:
        return M[s2[i], s2[j]]
    if cache[s2[i], s2[j]] in not null:
        cache[s2[i], s2[j]] = max(Lij, Aij, Rij)
    return [s2[i], s2[j]]
