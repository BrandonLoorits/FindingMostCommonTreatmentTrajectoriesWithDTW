import pandas as pd

import timeit

from tslearn.metrics import cdist_dtw
import numpy as np
from numpy import savetxt

# Funktsioon loomaks algtabel, kuhu salvestatakse andmetest vajalikud tunnused
# Filtreeritakse välja patsiendid, kellel esines vähem kui 2 seisundit

def looAlgAndmestik(taisTee):
    alg_andmestik = pd.read_csv(taisTee, sep=",") 
    alg_andmestik = alg_andmestik.iloc[: , :3]
    # Kui algandmetes märgitud ravitrajektoori algus ja lõpp - enim levinud ravitrajektooride leidmise puhul alguseks ja lõpuks seisundid (vaadeldakse ainult seisundi muutusi, ei vaadelda kui kaua seisundid kestsid)
    alg_andmestik = alg_andmestik[alg_andmestik["STATE"].str.contains("START|EXIT") == False] 
    #Lisatud, et ei oleks trajektoore, millel pikkus puudub
    alg_andmestik = alg_andmestik.groupby('SUBJECT_ID').filter(lambda SUBJECT_ID: len(SUBJECT_ID) > 1) 
    patsiendid = alg_andmestik['SUBJECT_ID'].unique()
    seisundid = alg_andmestik['STATE'].unique()
    alg_andmestik.columns = ['Patsiendi_ID','Seisund','Seisundi_Algus']
    print(alg_andmestik.head(10))
    return alg_andmestik,patsiendid,seisundid

# Funktsioon seisundite sõnastiku koostamiseks, mille abil seisundid väärtustatakse arvulise väärtusega
# Kõik astmaga seotud ravimigrupid grupeeritakse ühtsesse arvuvahemikku
# Ravimid, mis ei ole seotud astma haigusega, grupeeritakse teise gruppi

def leiaSeisundid(seisundid):
    seisundite_sonastik = {'Xanthines': 0, 
    'ICS': 1, 
    'SABA': 2, 
    'Systemic glucocorticoids': 3,
    'LABA&ICS': 4,
    'LABA': 5,
    'SAMA': 6,
    'LTRA': 7,
    'SABA&SAMA': 8,
    'LAMA': 9,
    'LABA&LAMA': 10}
    viimaneNR = 11
    for val in seisundid:
        if val not in seisundite_sonastik:
            seisundite_sonastik[val] = viimaneNR
            viimaneNR += 1
    
    return seisundite_sonastik

# Funktsioon, mis loob patsiendi ID-de abil neile vastavad ravitrajektoorid
# Patsiendi ID on võtmeks teisele sõnastikule, kus võtmeks 

def looRavitrajektoorid(alg_andmestik,patsiendid,seisundid):
    ravitrajektoorid = {}

    for i in patsiendid:
        ravitrajektoorid[i] = {'seisundi_algus':[],'arv_väärtus':[]} 

    seisundite_Sonastik = leiaSeisundid(seisundid)
    for indeks,rida in alg_andmestik.iterrows():
        ID = rida[0]
        ravim = rida[1]
        kpv = rida[2]
        if ID in ravitrajektoorid:
            ravitrajektoorid.get(ID).get('arv_väärtus').append(seisundite_Sonastik.get(ravim))
            ravitrajektoorid.get(ID).get('seisundi_algus').append(kpv)
    return ravitrajektoorid 

# Funktsioon, mis loob ainult ravitrajektoorid - salvestatakse seisundite arvulised väärtused ajalises järjestuses

def looAinultTrajektoorid(ravitrajektoorid):
    ainult_trajektoorid=[]

    for x in ravitrajektoorid:
        ainult_trajektoorid.append(ravitrajektoorid.get(x).get('arv_väärtus'))
        
    ainult_trajektoorid = np.array(ainult_trajektoorid)
    return ainult_trajektoorid

def koostaAlgtabel(ainult_trajektoorid,seisundite_Sonastik,tee):
    #ainult_trajektoorid = ainult_trajektoorid[:1000]
    unikaalsed, loendus = np.unique(ainult_trajektoorid, return_counts=True)
    tulemus = np.column_stack((unikaalsed, loendus)) 
    unikaalsed = []
    loendus = []

    for el in tulemus:
        vaheTulem = []
        for unikaalne in el[0]:
            for võti, väärtus in seisundite_Sonastik.items():
                if väärtus == unikaalne:
                    vaheTulem.append(võti)
        unikaalsed.append(vaheTulem)
        loendus.append(el[1])

    algtabel = pd.DataFrame({'Trajektoorid':unikaalsed, 'Loendus':loendus})  
    algtabel = algtabel.sort_values('Loendus', ascending=False, ignore_index=True)
    algtabel.index +=1
    #print(algtabel[:10])
    #algtabel.to_csv(tee+'Algtabel.csv') 
    return algtabel


# Muuta ainult ülakomade vahel olevat müra taset
myraTase = '0'
# Sisesta kaust, kuhu soovid tulemused salvestada ja, kus asuvad valideerimise tabel ning andmete alusfail
tee = "C:/Users/Brandon Loorits/Desktop/ulikool/5.semester/loputoo/graafikud/myra"+myraTase+"/"
alusfail = "Brandon_FINAL_ALL_DATA_"+myraTase+"_NOISE.csv"
maatriks = "myra"+myraTase+".csv"

maatriksiTee = tee + maatriks
taisTee = tee+alusfail

alg_andmestik,patsiendid,seisundid =looAlgAndmestik(taisTee)
seisundite_Sonastik = leiaSeisundid(seisundid)
ravitrajektoorid = looRavitrajektoorid(alg_andmestik,patsiendid,seisundid)
ainult_trajektoorid = looAinultTrajektoorid(ravitrajektoorid)

starttime = timeit.default_timer()
print('alustas tööga!')
sklearnX = cdist_dtw(ainult_trajektoorid[:1000])
print(sklearnX)
savetxt(maatriksiTee, sklearnX, delimiter=',')
print("Lõpetas tööga:", timeit.default_timer() - starttime)