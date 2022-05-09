import pandas as pd

import numpy as np
from numpy import loadtxt

import statistics as st

import math as mt

import matplotlib.pyplot as plt

from tslearn.clustering import silhouette_score

import plotly.express as px

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.cm as cm

from sklearn.cluster import AgglomerativeClustering
from itertools import chain
from collections import Counter, OrderedDict

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

# Funktsioon, mis leiab valideerimistabeli pikkuse
# Pikkuse leidmine vajalik, et teada, mis klastrite arvust alustatakse klasterdamist
# Igale klastrile leitakse üks kõige esinduslikum ravitrajektoor, mis on enim levinud klastris

def leiaAlgtabeliPikkus(tee):
    algtabel = pd.read_csv(tee + 'Algtabel.csv')
    #print(algtabel)
    trajektoorid = algtabel['Trajektoorid']
    trajArv = len(trajektoorid)

    return trajArv


# Funktsioon hierarhilise klasterdamise puudiagrammi visualiseerimiseks
# kasutatud alusena scikit-learn ametlikul lehel näidet:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

def koosta_dendrogramm(model,klasNR):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix,
    truncate_mode='lastp',  
    p=klasNR,  
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    color_threshold=1.2,
    above_threshold_color='y')

# Funktsioon klasterdamis mudelite koostamiseks, siluetidiagrammi ja puudiagrammi visualiseerimiseks
# Klasterdamise meetodi sisendiks olev sarnasusmaatriks peab olema varasemalt välja arvutatud faili arvutaSarnasusmaatriks.py abil

def klasterdaUus(ainult_trajektoorid,klastriteArv,cdistPath,piir,tee,valPikkus):
    skoorid = []
    mudelid = []
    sklearnX = loadtxt(cdistPath, delimiter=',')
    formatted_dataset = ainult_trajektoorid[:len(sklearnX)]
    range_n_clusters = list(range(valPikkus,klastriteArv))
    with PdfPages(tee+'KlastriteGraafikud.pdf') as pdf:
        for n_clusters in range_n_clusters:
            
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True, affinity='precomputed',linkage='average')
            mudelid.append(clusterer)
            y_predict = clusterer.fit_predict(sklearnX)
            cluster_labels = clusterer.labels_

            silhouette_avg = silhouette_score(sklearnX, cluster_labels)
            silhouette_avg = round(silhouette_avg,2)
            skoorid.append(silhouette_avg)
            if silhouette_avg > piir:
                print(
                    "N klastrite arvu korral =",
                    n_clusters,
                    "Keskmine siluetikoefitsent klastril on :",
                    silhouette_avg,
                )
                fig, (ax1, ax2) = plt.subplots(1, 2)

                fig.set_size_inches(15, 5)

                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(sklearnX) + (n_clusters + 1) * 10])

                sample_silhouette_values = silhouette_samples(sklearnX, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, ith_cluster_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("Siluetianalüüs")
                ax1.set_xlabel("Siluetikoefitsent")
                ax1.set_ylabel("Klastri märgistus")
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
                
                # 2nd Plot showing the actual clusters formedcolors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                
                koosta_dendrogramm(clusterer,n_clusters)

                ax2.set_title("Puudiagramm")
                ax1.set_ylabel("Sulgudes alluvate arv, sulgudeta trajektoori indeks")

                plt.suptitle(("Siluetianalüüs "
                            "klastrite arvuga: %d" % n_clusters),
                            fontsize=14, fontweight='bold')
                pdf.savefig()
                plt.close()

    return skoorid,mudelid,formatted_dataset

# Funktsioon leidmaks siluetikoefitsendi väärtuse alusel parim klastrite arv

def leiaParimMudel(skoorid,mudelid,valPikkus):
    best = np.argmax(skoorid)
    silh_skoor=skoorid[best]
    print()
    print('Parim klastrite arv:',best+valPikkus,'---','Siluetikoefitsent:',silh_skoor)
    print()
    parim_mudel = mudelid[best]
    best += valPikkus
    y_pred = parim_mudel.labels_
    return best,y_pred

# Funktsioon, mis leiab klastris esinevate trajektooride esinemiste arvu
# Eemaldab klastrist kõik trajektoorid, mis on osakaalu protsendist väiksemad

def leiaSarnaseimadKlastris(best,formatted_dataset,y_pred,kaal):  
          
    klastridAlgne=[]
    klastrid=[]
    for i in range(best):
        klastridAlgne.append([])

    for i in range(len(formatted_dataset)):
        klastridAlgne[y_pred[i]].append(formatted_dataset[i])
    #print(klastridAlgne[0])
    unikaalsedSeisundid = []

    print()
    print("KLASTRITE ESINEMISSAGEDUSE MÕÕDIKUD")
    counter=1
    
    for klaster in klastridAlgne:
        #print(klaster)
        print()
        print('--------------klastri-NR:',counter,'------------------')
        counter+=1
        
        leidja = Counter(map(tuple,klaster))
        trajid = []
        loendus = []
        osakaal = []
        kogus = len(klaster)
        for key,value in leidja.items():
            #print(key,value)
            trajid.append(list(key))
            loendus.append(value)
            osakaal.append(round(value*100/kogus,1))
        sonastik = {'trajektoorid':trajid,'loendus': loendus,'osakaal':osakaal}
        data = pd.DataFrame(sonastik)
        data = data.sort_values(by=['loendus'],ignore_index=True,ascending=False)
        data['kumulatiivne'] = data['osakaal'].cumsum()
        data = data[data['osakaal'] > kaal]
        
        print(data.head(25))
        uusKlaster = data['trajektoorid'].tolist()
        klastrid.append(uusKlaster)

    for klaster in klastrid:
        vahe = list(set(chain(*klaster)))
        #print('vahe',vahe)
        if len(vahe) > 0:
            unikaalsedSeisundid.append(vahe)
        else:
            unikaalsedSeisundid.append([])

    return klastridAlgne,unikaalsedSeisundid

# Funktsioon, mis leiab klastris esinevate trajektooride ühisosad seisundite alusel - ühisosaks kõigis trajektoorides esinevad seisundid

def leiaUhisosa(klastrid,unikaalsed):
    #print('klastrid:',klastrid)
    #print('unikaalsed',unikaalsed)
    eksisteerivadHulk=[]

    for i in range(len(klastrid)):
        #print('uni:',unikaalsed[i])
        #print('Klas:',klastrid[i])
        eksisteerivad = []
        for unikaalne in unikaalsed[i]:
            counter = 0
            pikkus = len(klastrid[i])
            for el in klastrid[i]:
                #print(el)
                if unikaalne in el:
                    counter += 1
            if counter==pikkus:
                eksisteerivad.append(unikaalne)
        eksisteerivadHulk.append(eksisteerivad)
    #print('Koigis esinevad seisundid:',eksisteerivadHulk)
    return eksisteerivadHulk

# Funktsioon, mis korrigeerib trajektoorid ühisosa alusel
# Trajektooridesse jäetakse alles ainult ühisosas leiduvad seisundid ja seisundite mitme kordsel esinemisel jäetakse alles ainult ühe kordne esinemine
# kuna soovitakse leida seisundite muutusi

def korrigeeriTrajektoorid(klastrid,eksisteerivadHulk):
    for i in range(len(klastrid)):
        #print('-----',i,'-----')
        #print('Klas:',klastrid[i])
        #print('koigis esinevad:',eksisteerivadHulk[i])
        eksist=eksisteerivadHulk[i]
        if len(eksist) > 1:   
            for a,el in enumerate(klastrid[i]):
                klastrid[i][a] = list(OrderedDict.fromkeys(list(filter(lambda x: x in eksist , el))))    
        else:
            klastrid[i] = []

    print()
    print('KORRIGEERITUD TRAJEKTOORID')
    for i in range(len(klastrid)):
        print()
        print('------------------klastri-NR:',i+1,'-----------------------')
        print('Koigis esinevad seisundid:',eksisteerivadHulk[i])
        print()
        print('KLASTER listi kujul')
        print(klastrid[i])
    return klastrid

# Funktsioon leidmaks tulemustrajektooride pikkus, klastris esinevate trajektooride pikkuste alusel

def leiaPikkusteMediaan(klastrid):
    #print(klastrid)
    pikkused = []
    for klaster in klastrid:
        pikkused.append([])
        for el in klaster:
            pikkus = len(el)
            pikkused[-1].append(pikkus)
    #print()        
    #print(pikkused)
    for i in range(len(pikkused)):
        if len(pikkused[i])>0:
            pikkused[i] = mt.floor(st.median(pikkused[i]))
        else:
            pikkused[i] = 0
    #print(pikkused)
    return pikkused

# Funktsioon, mis koostab klastrite tulemustrajektoori vastavalt ajalises järjestuses esinevate seisundite osakaalude alusel

def leiaEnimLevinud(klastrid,pikkused,seisundite_Sonastik,pathToDir):
    steps = []
    #print(klastrid[0])
    #print(steps)
    #print(seisundite_Sonastik.keys())
    for a in range(len(klastrid)):
        steps.append([])
    print()
    

    print('TULEMUSTRAJEKTOORIDE ENNUSTUS: ')
    #print(steps)
    tulemusTrajid = []
    for x in range(len(klastrid)):
        print()
        print('-----------------------',x+1,'---------------------------')
        tulemusTrajid.append([])
        data = pd.DataFrame()
        #print(data)
        for i in range(pikkused[x]):
            steps[x].append([])
        #print(steps)
        for a in klastrid[x]:
            #print('-----------')
            #print(a)
            for b in range(pikkused[x]):
                if len(a) > b:
                    steps[x][b].append(a[b])
                else:
                    steps[x][b].append(100.0)
                
            #print('steps:',steps)    
        #print('SIIN:',steps)
        for y,step in enumerate(steps[x]):
            #print('SIIN',step)
            for i in range(len(step)):
                for voti, vaartus in seisundite_Sonastik.items():
                    if vaartus == step[i]:
                        step[i] = voti
            #print(x)
            #print(y,'---',steps)
            data['step'+ str(y+1)] = steps[x][y]
        #data = pd.DataFrame({'step1':steps[0],'step2':steps[1]})
        #print('x ---',x)
        print(data)
        #print(data.columns) 
        path = []

        for colName in data.columns:
            suurimOsak = data[colName].value_counts().idxmax()
            print(suurimOsak,'->')
            tulemusTrajid[x].append(suurimOsak)
            path.append(str(colName))
        #print(path)
        if pikkused[x] > 0:
            fig = px.sunburst(data, path=path,width=600,height=600)
            #print('teekond:',path)
            fig.write_html(pathToDir+'sb'+str(x+1)+'.html')
    return(tulemusTrajid)

# Funktsioon tulemuste visualiseerimiseks

def looGraafikud(skoorid,best,formatted_dataset,pathToDir,ravitrajektoorid,y_pred,valPikkus):
    with PdfPages(pathToDir+'GRAAFIKUD.pdf') as pdf:
        naide1 = ravitrajektoorid.get(1).get('arv_väärtus')
        naide2 = ravitrajektoorid.get(30).get('arv_väärtus')
        plt.figure(figsize=(6, 4))
        plt.title('Näidis ravitrajektoor')
        plt.plot(np.arange(len(naide1)), naide1 , "-o", c="C3")
        plt.plot(np.arange(len(naide2)), naide2 , "-o", c="C1")
        pdf.savefig()
        plt.close() 

        plt.figure(figsize=(6, 4))
        plt.title('Parima skoori leidmine')
        plt.plot(skoorid,'bo-')
        plt.xlabel('k')
        plt.ylabel('skoorid')
        plt.grid(which='major',color='grey',linestyle='--')
        plt.axvline(x=best-valPikkus,c='green',label='Parim klastrite arv')
        plt.scatter(best-valPikkus,skoorid[best-valPikkus],c="orange",s=400)
        plt.xticks(np.arange(len(skoorid)),np.arange(valPikkus,len(skoorid)+valPikkus))
        pdf.savefig()
        plt.close()
        arv = mt.ceil(mt.sqrt(best))
        # print('arv',arv)
        # print('best',best)
        if arv < 2:
            arv = 2
        plt.figure(figsize=(6, 4))
        for yi in range(best):
            plt.subplot(arv+1, arv+1, 1 + yi)
            for xx in formatted_dataset[y_pred == yi]:
                plt.plot(np.array(xx).ravel(), "k-", alpha=.2)
        plt.suptitle('Klasterdamine visuaalselt')
        pdf.savefig()
        plt.close()

# Funktsioon, mis valideerib enim levinud ravitrajektooride leidmist 

def leiaAlgtabelist(tee,tulemusTrajid):
    tulemused ={}
    algtabel = pd.read_csv(tee + 'Algtabel.csv')
    #print(algtabel)
    trajektoorid = algtabel['Trajektoorid']
    trajArv = len(trajektoorid)
    #print(trajektoorid)
    print('Väljapakutud tulemustrajektoorid: ')
    print(tulemusTrajid)
    uus = []

    for i,traj in enumerate(trajektoorid):
        # print(type(traj))
        # print(traj)
        traj = str(traj).replace("[","").replace("]","").replace("'","").split(',')
        for x,el in enumerate(traj):
            traj[x] = el.strip()
        uus.append(traj)
    # print(type(uus[0]))
    arvud = [0]*len(uus)

    m=0
    oigedKlastrid=0
    for trajL in tulemusTrajid:
        olemas = True
        for x,trajA in enumerate(uus):
            # print(trajL)
            # print(trajA)
            i = [item for item in trajL if item in trajA]
            if i==trajA:
            #lisab +1 vektorisse sellele kohale
                arvud[x] +=1 
                olemas=False
        if olemas:
            m += 1
        else:
            oigedKlastrid += 1
    arv = sum(i > 0 for i in arvud)
    return tulemused,trajArv,arv,m,oigedKlastrid

# Muuta ainult ülakomade vahel olevat müra taset
myraTase = '0'
# Sisesta kaust, kuhu soovid tulemused salvestada ja, kus asuvad valideerimise tabel ning andmete alusfail
tee = "C:/Users/Brandon Loorits/Desktop/ulikool/5.semester/loputoo/graafikud/myra"+myraTase+"/"
alusfail = "Brandon_FINAL_ALL_DATA_"+myraTase+"_NOISE.csv"
maatriks = "myra"+myraTase+".csv"

maatriksiTee = tee + maatriks
taisTee = tee+alusfail

# Sisesta siluetikoefitsendi väärtus, millest väiksemate tulemuste puhul mudelit ei koosatata
piir = 0.1
# Sisesta klastrtite arvu ülem piir, alampiiriks on valideerimistabelis leiduvate trajektooride arv
klastriteArv = 10
# Sisesta trajektooride esinemiste osakaal klastris. Trajektoorid, mille esinemiste osakaal klastris on väiksem kui sisestatud osakaal jäetakse klastrist välja
osakaal = 15.0

# Töövoo algus
alg_andmestik,patsiendid,seisundid =looAlgAndmestik(taisTee)
seisundite_Sonastik = leiaSeisundid(seisundid)
ravitrajektoorid = looRavitrajektoorid(alg_andmestik,patsiendid,seisundid)
ainult_trajektoorid = looAinultTrajektoorid(ravitrajektoorid)
valPikkus = leiaAlgtabeliPikkus(tee)

print()
print('Seisundite sõnastik')      
print(seisundite_Sonastik)
print()

skoorid,mudelid,formatted_dataset = klasterdaUus(ainult_trajektoorid,klastriteArv,maatriksiTee,piir,tee,valPikkus)
best,y_pred = leiaParimMudel(skoorid,mudelid,valPikkus)
klastrid,unikaalsed =leiaSarnaseimadKlastris(best,formatted_dataset,y_pred,osakaal)
eksisteerivadHulk = leiaUhisosa(klastrid,unikaalsed)
klastrid = korrigeeriTrajektoorid(klastrid,eksisteerivadHulk)
pikkused = leiaPikkusteMediaan(klastrid)
tulemusTrajid = leiaEnimLevinud(klastrid,pikkused,seisundite_Sonastik,tee)
looGraafikud(skoorid,best,formatted_dataset,tee,ravitrajektoorid,y_pred,valPikkus)
# Töövoo lõpp

# Valideerimine ja analüüs
tulem, algtabelitrajarv, arv, m, oigedKlastrid = leiaAlgtabelist(tee,tulemusTrajid)
tulem = sorted(tulem.values())
tulemusKlastriteArv = best
print()
print('PARIM KLASTRITE ARV:', tulemusKlastriteArv)
print('Leitud müra klastrid:',m)
print('Õigesti leitud klastrite arv:',oigedKlastrid)
print('Tulemustrajektooride arv, mis esinevad valideerimistabelis:',arv)
print('Täpsus1 - kõikidest klastritest õigesti leitud klastrite protsent:', round((tulemusKlastriteArv-m)/tulemusKlastriteArv*100,2))
print('Täpsus2 - valideerimstabelist leitud trajektooride protsent:', round(arv*100/algtabelitrajarv,2))
#print('tulemused',tulem, 'tulemuste pikkus:', len(tulem),'algtabeli pikks:',algtabelitrajarv,'hinnang:',len(tulem)*100/algtabelitrajarv)


