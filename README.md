# pack
prevision à une heure

La forme des données doit être identique pour les deux modèles, à savoir un tenseur de trois dimensions au format float64 pour ne pas perdre de l'info, et normalisées en soustrayant la moyenne et en divisant par l'ecart-type du jeu d'entrainement (standardscaler):
- Une pour les échantillons
- Une autre pour les pas de temps, soit 100 pas (pour dix heures, donc un pas toutes les six minutes).
- Une pour les variables, soit 9 (latitude, longitude, altitude, direction, force, humidite, température, pression et le mois de la mesure pour induire une saisonnalité).
donc un tenseur (nombre_échantillons, 100, 9)

Les deux modèles ont été entrainés sur un jeu d'environ 17 millions d'échantillons. Le score moyen du modèle direction est de 68% (accuracy) et celui de la force de 0.8 (erreur moyenne absolue). Le résultat pour la direction indique la classe qui a eu la meilleure probabilité d'occurence sur huit classes (NE, E, SE, S, SW, W, NW, N), chaque classe couvre donc 45 degrés. Concernant la force du vent, elle est en mètres/seconde comme Meteonet.

Meteonet: voilà le lien explicatif des variables de ce jeu open source : https://meteofrance.github.io/meteonet/french/donnees/stations-observation/
et ici le lien pour telecharger les csv: https://meteonet.umr-cnrm.fr/dataset/data/

le csv erreur comprend les scores des modèles sur les stations en france pour 2018.
