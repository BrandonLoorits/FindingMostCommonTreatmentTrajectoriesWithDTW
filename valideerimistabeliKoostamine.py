import pandas as pd

# myraTase ülekomade vahel olevat arvu muuta, olenevalt, millise müratasemega kausta on vaja luua valideerimistabel
myraTase = '0'
tee = "C:/Users/Brandon Loorits/Desktop/ulikool/5.semester/loputoo/graafikud/myra"+myraTase+"/"


def koostaValTabel(tee):
    #ainult_trajektoorid = ainult_trajektoorid[:1000]
    unikaalsed = [['LABA&ICS', 'ICS'],['LABA&ICS', 'SABA'],['LABA&ICS', 'Systemic glucocorticoids'],['LABA&ICS', 'ICS', 'SABA'],['LABA&ICS', 'ICS', 'Systemic glucocorticoids']]
    loendus = [2029,1891,1214,515,420]
    algtabel = pd.DataFrame({'Trajektoorid':unikaalsed, 'Loendus':loendus})  
    algtabel = algtabel.sort_values('Loendus', ascending=False, ignore_index=True)
    algtabel.index +=1
    print(algtabel[:10])
    algtabel.to_csv(tee+'Algtabel.csv')

koostaValTabel(tee)
