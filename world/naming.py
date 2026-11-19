"""Pronounceable procedural names for systems and planets."""

ONSETS = ("V", "K", "T", "Z", "M", "N", "R", "S", "L", "D", "Th", "Kr",
          "X", "Ph", "Qu", "Y", "B", "G")
VOWELS = ("a", "e", "i", "o", "u", "ae", "ia", "ou", "ei")
CODAS = ("", "", "", "n", "r", "l", "s", "th", "x")
ENDINGS = ("ra", "nis", "thos", "lia", "dor", "vex", "ara", "ion", "esh",
           "os", "une", "ami")

ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")


def _word(rng, syllables):
    parts = []
    for index in range(syllables):
        parts.append(rng.choice(ONSETS))
        parts.append(rng.choice(VOWELS))
        parts.append(rng.choice(CODAS))
    return "".join(parts).capitalize()


def planet_name(rng):
    return _word(rng, 2) + rng.choice(ENDINGS)


def star_name(rng):
    return _word(rng, 2) + rng.choice(("", "", "a", "is", "os", "um"))


def unique_name(rng, used, generator=star_name, attempt_limit=200):
    for _ in range(attempt_limit):
        candidate = generator(rng)
        if candidate not in used:
            used.add(candidate)
            return candidate
    base = generator(rng)
    index = 2
    while True:
        suffix = ROMAN[index - 1] if index <= len(ROMAN) else str(index)
        candidate = f"{base} {suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1
