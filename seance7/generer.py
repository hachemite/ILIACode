# ============================================================
#  Séance 7 --- la brique durable, le moteur écrit
#  Trois fonctions à écrire (partie 2 du TP N7), chacune se termine
#  par une ligne raise NotImplementedError à remplacer par le vrai code.
#  Les séances suivantes relisent ces fonctions telles quelles, pour
#  montrer ce que le moteur écrit avant et après un affinage (S9), une
#  extension du vocabulaire (S10), et pour noter le code écrit.
#  Le moteur est celui de la séance 1 d'IA Générative, chargé par
#  charger_moteur de moteur.py, que le script regler.py relit.
#  La fonction scores_suivant, en bas du fichier, est déjà écrite.
# ============================================================
import torch


def choisir_token(scores, temperature=0.8, p_noyau=0.95, generateur=None):
    """Le numéro du token choisi à partir des scores que le moteur donne
    à chaque token du vocabulaire (un vecteur de |V| nombres). Avec une
    température nulle, le token de score maximal. Sinon, les scores sont
    divisés par la température, passés au softmax, la queue est retirée
    au noyau p_noyau, et le token est tiré au sort dans ce qui reste.
    generateur est le générateur de nombres aléatoires de torch, pour
    rejouer un tirage à l'identique."""
    if temperature == 0:
        return int(torch.argmax(scores).item())
    if generateur is None:
        generateur = torch.Generator(device=scores.device)
    probabilites = torch.softmax(scores / temperature, dim=-1)
    triees, indices = torch.sort(probabilites, descending=True)
    cumul = torch.cumsum(triees, dim=-1)
    conserve = cumul - triees <= p_noyau
    conserve[0] = True
    indices_gardes = indices[conserve]
    probabilites_gardees = triees[conserve]
    probabilites_gardees = probabilites_gardees / probabilites_gardees.sum()
    choix = torch.multinomial(probabilites_gardees, 1, generator=generateur)
    return int(indices_gardes[choix.item()].item())


def generer(modele, tokeniseur, demande, n_max=200, temperature=0.8,
            p_noyau=0.95, graine=0, arret=None):
    """Le texte écrit par le moteur à la suite de la demande, un token à
    la fois, jusqu'au token de fin, à n_max tokens, ou à la première
    apparition de la séquence d'arrêt arret (qui n'est pas rendue). Seul
    le texte écrit est rendu, sans la demande. La même graine donne le
    même texte."""
    numeros = tokeniseur(demande, return_tensors="pt")["input_ids"].to(modele.device)
    generateur = torch.Generator(device=modele.device).manual_seed(graine)
    ecrits = []
    for _ in range(n_max):
        scores = scores_suivant(modele, numeros)
        suivant = choisir_token(scores, temperature, p_noyau, generateur)
        if tokeniseur.eos_token_id is not None and suivant == tokeniseur.eos_token_id:
            break
        ecrits.append(suivant)
        numeros = torch.cat((numeros, torch.tensor([[suivant]], device=modele.device)), dim=1)
        texte = tokeniseur.decode(ecrits)
        if arret is not None and arret in texte:
            texte = texte.split(arret, 1)[0]
            return texte
    return tokeniseur.decode(ecrits)


def pass_at_k(verdicts):
    """La proportion des demandes dont au moins une réponse passe tous ses
    tests. verdicts contient une liste par demande, avec un booléen par
    réponse générée, True si elle a passé les tests."""
    if not verdicts:
        return 0.0
    reussies = sum(any(reponse for reponse in demandes) for demandes in verdicts)
    return reussies / len(verdicts)


# ------------------------------------------------------------
#  Déjà écrite. Les scores que le moteur donne au token suivant, à la
#  dernière position lue, un nombre par token du vocabulaire, avant le
#  softmax. Le moteur traverse ses blocs en lecture seule.
# ------------------------------------------------------------
def scores_suivant(modele, numeros):
    """Les |V| scores du token suivant après les numéros déjà lus, un
    tenseur de forme (1, n) placé sur l'appareil du moteur."""
    with torch.no_grad():
        return modele(numeros).logits[0, -1].float()
