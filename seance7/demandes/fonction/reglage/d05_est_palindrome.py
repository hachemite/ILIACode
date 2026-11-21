def est_palindrome(texte):
    """Rend True si texte se lit pareil dans les deux sens, sans tenir
    compte des majuscules ni des espaces. 'Esope reste ici et se repose'
    donne True."""
    propre = "".join(caractere.lower() for caractere in texte if not caractere.isspace())
    return propre == propre[::-1]
