# FindingMostCommonTreatmentTrajectoriesWithDTW
Repositoorium on loodud lõputöö "Patsientide enim levinud ravitrajektooride leidmine DTW meetodil" jaoks. 

Repositooriumis on neli faili.

valideerimistabeliKoostamine.py on valideerimise tabeli koostamiseks
arvutaSarnasusmaatriks.py on sarnasusmaatriksi arvutamiseks dünaamilise ajadeformatsiooni algoritmi alusel
TöövoogSiluetikoefitsent.py on lõputöös väljapakutud töövoog, siluetikoefitsendi väärtuse alusel mudeli valik
TöövoogNKlastrit.py on lõputöös väljapakutud töövoog, kus on võimalik sisendiks anda klastrite arv, mille alusel klasterdamine koostatakse

Töövoo kasutamiseks on vajalik programeerimiskeele Python arenduskeskkond.
Töövoo käivitaminse juhend:
1. Esmalt loo kaust, kuhu soovid tulemusi salvestada ning kus on andmestik, millele töövoogu rakendada soovitakse
2. Käivita valideerimistabeliKoostamine.py sisendiks vajalik failitee kaustani, andmefaili nimi - kausta luuakse valideerimistabelit sisaldav fail
3. Käivita arvutaSarnasusmaatriks.py sisendiks failitee kaustani, andmefaili nimi - kausta luuakse fail, mis sisaldab sarnasusmaatriksit
4. Käivita TöövoogSiluetikoefitsent.py või TöövoogNKlastrit.py olenevalt, kas soov käivitada töövoog n klastrite arvuga 
   või klastrite arvu valik tehakse automaatselt siluetikoefitsendi väärtuse alusel
   
