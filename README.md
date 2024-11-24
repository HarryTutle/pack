# pack
prevision à une heure

La forme des données doit être identique pour les deux modeles, à savoir un tenseur de trois dimensions au format float64 pour ne pas perdre de l'info, et normalisées en soustrayant la moyenne et en divisant par l'ecart-type:
- Une pour les échantillons
- Une autre pour les pas de temps, soit 100 pas (pour dix heures, donc un pas toutes les six minutes).
- Une pour les variables, soit 9 (latitude, longitude, altitude, direction, force, humidite, température, pression et le mois).


Les deux modèles ont été entrainés sur un jeu d'environ 17 millions d'échantillons. Le score moyen du modèle direction est de 68% (accuracy) et celui de la force de 0.8 (erreur moyenne absolue).
