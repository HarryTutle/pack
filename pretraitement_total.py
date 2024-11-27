# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:32:19 2024

@author: thoma
"""




""" librairies """

import numpy as np # permet le calcul tensoriel rapide.
import pandas as pd # traitement de tableaux, données, stats.
import matplotlib.pyplot as plt # graphiques.
import datetime as dt # utiliser une variable temporelle comme index.

from tensorflow.keras import Model # deep learning.
from tensorflow.keras import optimizers, layers, Sequential
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
from tensorflow.keras import wrappers
from tensorflow.data import Dataset, TextLineDataset
from tensorflow.io import decode_csv
from tensorflow import stack, constant, float32, convert_to_tensor, float64, slice, int64
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow import reshape



from sklearn.model_selection import train_test_split # métriques, normalisation...
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.model_selection import RandomizedSearchCV




zone=[5050, 5100, 490, 495] # permet de cibler une zone geographique en important des données de meteonet.(latitude basse*100, lat haute*100, longitude basse et haute*100)
chunksize=10000 # nombre d echantillons traité par morceau importé via la fonction pd.read_csv.
mean_tenseur=[42259671.94, 4595.35, 237.56, 142.38, 187.01, 3.84, 74.49, 286.58, 1017.05, 6.52] # moyenne des variables numero station, latitude, longitude, altitude, direction, force, humidité, temperature en kelvin, pression, mois.
std_tenseur=[25851787.75, 253.68, 385.51, 179.13, 111.68, 2.90,  18.05, 7.29, 833.26, 3.45] # cette fois, ecarts-types de ces variables. Ces deux listes permettent de normaliser les données.
chunks=[]



for chunk in pd.read_csv('D:/station meteo 2024/NW2018.csv',chunksize=chunksize): # cette boucle traite par morceaux les données pour pas surcharger la ram d'un coup. Le csv est un csv direct de meteonet.
 
    chunk['lat']=chunk['lat'].apply(lambda x: int(x*100)) # on multiplie par 100 la latitude pour changer les float en int et prendre moins de memoire.
    chunk['lon']=chunk['lon'].apply(lambda x: int(x*100)) # pareil pour la longitude.
    chunk=chunk.loc[chunk.lat>zone[0] , ]
    chunk=chunk.loc[chunk.lat<zone[1] , ]
    #chunk=chunk.loc[chunk.lon>zone[2] , ]
    #chunk=chunk.loc[chunk.lon<zone[3] , ]
    chunk=chunk.loc[chunk.number_sta==62160001, ] # si on veut aussi selectionner par station.
    chunks.append(chunk)


data=pd.concat(chunks, axis=0)
print(data.iloc[:,4:].head()) # affiche les 5 premiers echantillons.

print(data['number_sta'].nunique()) #affiche le nombre de stations dans les données.
print(data['number_sta'].unique()) # affiche numeros stations.




def cap(var):  # cette fonction change la variable direction en variable categorielle.
    
    if (var>337.5) or (var<=22.5):
        var=0
    elif (var>22.5) and (var<=67.5):
        var=45
    elif (var>67.5) and (var<=112.5):
        var=90
    elif (var>112.5) and (var<=157.5):
        var=135
    elif (var>157.5) and (var<=202.5):
        var=180
    elif (var>202.5) and (var<=247.5):
        var=225
    elif (var>247.5) and (var<=292.5):
        var=270
    elif (var>292.5) and (var<=337.5):
        var=315
        
    return var


def label_direction(val): # celle-là labellise la variable direction. 
    
    if val==0:
        
        return 0
    
    elif val==45:
        
        return 1
    
    elif val==90:
        
        return 2
    
    elif val==135:
        
        return 3
    
    elif val==180:
        
        return 4
    
    elif val==225:
        
        return 5
    
    elif val==270:
        
        return 6
    
    elif val==315:
        
        return 7
    
    else:
        
        return 'nan'







    
def Meteonet_manip(data=data, fréquence=6, heures_passé=24, minutes_futur=6, corbeille=['pluie'], variable_cible=['direction']):
        
        """
        data: donnée meteonet csv.
        
        fréquence: densité des échantillons en minutes (par 6 minutes par défaut, pas le plus faible. doit être des multiples de 6).
        
        heures_passé: taille de la séquence temporelle en heures avant le moment présent.
        
        minutes_futur: prédiction dans le futur en minutes. Minimum 6 min, doit être un multiple de 6.
        
        corbeille: liste des variables inutiles.
        
        variable_cible: variables cibles sélectionnées
        
    
        
        """
        
        
        
    
       
        liste_finale_variables=[]
        liste_finale_cibles=[]
        

        indexage_heures=list(pd.date_range('2016-01-01 00:00:00', '2018-12-31 23:00:00', freq=str(fréquence)+'min')) 
        time_heures=pd.DataFrame({'temps': indexage_heures}) 
        time_heures=time_heures.set_index('temps') 
        
        indexage_jours=list(pd.date_range('2016-01-01', '2018-12-31', freq='d')) 
        time_jours=pd.DataFrame({'temps': indexage_jours})

        for station in data['number_sta'].unique():  
           print(station) # affiche les numeros des stations au fur et à mesure du formatage.
           station_data=data.loc[data['number_sta']==station]
           station_data=station_data.sort_values(['date'],ascending=True)
           station_data=station_data.set_index('date')
           station_data.index=pd.to_datetime(station_data.index)
           station_data=station_data.resample(str(fréquence)+'min').mean()
           station_data=time_heures.join(station_data, how='outer')
           station_data["dd"]=station_data["dd"].map(lambda x: cap(x))
           station_data['number_sta']=station_data['number_sta'].apply(lambda x: int(x) if np.isnan(x)==False else x)
           #station_data['lat']=station_data['lat'].apply(lambda x: int(x*100) if np.isnan(x)==False else x)
           #station_data['lon']=station_data['lon'].apply(lambda x:int(x*100) if np.isnan(x)==False else x)
           station_data['height_sta']=station_data['height_sta'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           #station_data['dd']=station_data['dd'].apply(lambda x:int(x) if np.isnan(x)==False else x)
           station_data['ff']=station_data['ff'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['hu']=station_data['hu'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['precip']=station_data['precip'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['psl']=station_data['psl'].apply(lambda x:int(x/100) if np.isnan(x)==False else x)
           station_data['td']=station_data['td'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data['t']=station_data['t'].apply(lambda x:int(round(x)) if np.isnan(x)==False else x)
           station_data.columns=['station', 'latitude', 'longitude', 'altitude', 'direction', 'force', 'pluie', 'humidité', 'point_rosée', 'température', 'pression']
           station_data['mois']=station_data.index.month
           station_data=station_data.drop(corbeille, axis=1)
           station_data=station_data.dropna(axis=1, how='all')
           station_data=station_data.dropna(axis=0, how='any')
           station_data=time_heures.join(station_data, how='outer')
           

               
           échelle_passé=dt.timedelta(hours=heures_passé)-dt.timedelta(minutes=fréquence)
           échelle_futur=dt.timedelta(minutes=minutes_futur)
           
                   
           for date in station_data.index:
                   
               début=date-échelle_passé
               fin=date+échelle_futur
               début_index=pd.date_range(start=début, end=date, freq=str(fréquence)+'min')
               début_index=pd.DataFrame(début_index).set_index(0)
               dataset=début_index.join(station_data, how='left')
                   
                   
               try: 
                       
                   cible=station_data.loc[fin, variable_cible]
                       
               except:
                       
                   cible=np.empty((1, len(variable_cible)))
                   cible=cible.fill(np.nan)
                   cible=pd.DataFrame(cible)
                   
               if (dataset.isnull().values.any()==False) and (cible.isnull().values.any()==False) and (cible.shape[0]==len(variable_cible)) and (dataset.shape[0]==(heures_passé*60)//fréquence) and (dataset.shape[1]==12-len(corbeille)):
                       
                   dataset=np.array(dataset)
                       
                   dataset=dataset.reshape(1, (heures_passé*60)//fréquence, 12-len(corbeille))
                       
                   dataset=np.flip(dataset, axis=1)
                   liste_finale_variables.append(dataset)
                   liste_finale_cibles.append(np.array(cible).reshape(1, len(variable_cible)))
                   
                       
               else:
                   
                   pass
                 
                       
        
        liste_finale_variables=np.concatenate(liste_finale_variables, axis=0)
        liste_finale_cibles=np.concatenate(liste_finale_cibles, axis=0)
        

        liste_finale_variables=(liste_finale_variables-mean_tenseur) / std_tenseur  
        
        for val in range(liste_finale_cibles.shape[0]):
            liste_finale_cibles[val, 0]=label_direction(liste_finale_cibles[val, 0])
            
        
        
        
        return liste_finale_variables, liste_finale_cibles

        
                              
          

dataset, cibles=Meteonet_manip(heures_passé=10, corbeille=['point_rosée', 'pluie'], variable_cible=['direction', 'force'], minutes_futur=60, fréquence=6) # ici via la fonction on modèle un jeu de dimension adéquate pour nos modèles.

np.save('debug_dataset.npy', dataset) # pour sauvegarder les données formatées pour les modèles, ici le dataset.
np.save('debug_cibles.npy', cibles) # là c'est les cibles.


dataset=np.load("C:/Users/thoma/Documents/meteo/station meteo 2024/meteonet/debug_dataset.npy", allow_pickle=True) # pour charger.
cibles=np.load("C:/Users/thoma/Documents/meteo/station meteo 2024/meteonet/debug_cibles.npy", allow_pickle=True)


print(dataset.dtype)
print(type(dataset))
print(cibles.dtype)
print(type(cibles))

dir_cible=cibles[:, 0]
for_cible=cibles[:, 1]

dataset=dataset[:,:,1:] # on vire la variable numero station.
dataset=dataset.astype(np.float64) # pour avoir le bon format.
dir_cible=dir_cible.astype(np.float64)
for_cible=for_cible.astype(np.float64)

print(dataset.dtype)
print(type(dataset))

dir_model=models.load_model('C:/Users/thoma/Documents/meteo/station meteo 2024/meteonet/modeles/4_direction_total_model_lstm_10h_1h_6min.h5') # import des modeles.
for_model=models.load_model('C:/Users/thoma/Documents/meteo/station meteo 2024/meteonet/modeles/4_force_total_model_lstm_10h_1h_6min.h5')

dir_prediction=dir_model.predict(dataset).argmax(axis=1) # sans argmax, donne les huit classes et leur probabilité d'occurence. Avec argmax, conserve seulement la classe la plus probable.
for_prediction=for_model.predict(dataset)

print(accuracy_score(dir_cible, dir_prediction))
print(mean_absolute_error(for_cible, for_prediction))
print(classification_report(dir_cible, dir_prediction))

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.title('comparaison données réelles et prédiction, direction')
plt.plot(dir_prediction[:200], label='prédiction', color='red')
plt.plot(dir_cible[:200], label='données réelles', color='blue')
plt.yticks([0,1,2,3,4,5,6,7],['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
plt.ylabel('direction')
plt.xlabel('temps')
plt.grid()
plt.subplot(2, 1, 2)
plt.title('comparaison données réelles et prédiction, force')
plt.plot(for_prediction[:200], label='prédiction', color='red')
plt.plot(for_cible[:200], label='données réelles', color='blue')
plt.ylabel('force en m/s')
plt.xlabel('temps')
plt.grid()
plt.legend()
plt.show()

