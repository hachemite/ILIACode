def compter_voyelles(texte):
    """Rend le nombre de voyelles a, e, i, o, u, y de texte, en comptant
    aussi les majuscules."""
    return sum(caractere.lower() in "aeiouy" for caractere in texte)
