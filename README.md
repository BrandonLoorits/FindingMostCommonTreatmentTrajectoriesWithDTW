# FindingMostCommonTreatmentTrajectoriesWithDTW
Repositoorium on loodud lõputöö "Patsientide enim levinud ravitrajektooride leidmine DTW meetodil" jaoks. 

Repositooriumis on neli faili.

valideerimistabeliKoostamine.py on valideerimise tabeli koostamiseks <br />
arvutaSarnasusmaatriks.py on sarnasusmaatriksi arvutamiseks dünaamilise ajadeformatsiooni algoritmi alusel <br />
TöövoogSiluetikoefitsent.py on lõputöös väljapakutud töövoog, siluetikoefitsendi väärtuse alusel mudeli valik <br />
TöövoogNKlastrit.py on lõputöös väljapakutud töövoog, kus on võimalik sisendiks anda klastrite arv, mille alusel klasterdamine koostatakse <br />

Töövoo kasutamiseks on vajalik programeerimiskeele Python arenduskeskkond. <br />

Töövoo käivitamise juhend:
1. Esmalt loo kaust, kuhu soovid tulemusi salvestada ning kus on andmestik, millele töövoogu rakendada soovitakse
2. Käivita valideerimistabeliKoostamine.py, sisendiks vajalik failitee kaustani - kausta luuakse fail, mis sisaldab valideerimistabelit
3. Käivita arvutaSarnasusmaatriks.py sisendiks failitee kaustani, andmefaili nimi - kausta luuakse fail, mis sisaldab sarnasusmaatriksit
4. Käivita TöövoogSiluetikoefitsent.py või TöövoogNKlastrit.py olenevalt, kas soov käivitada töövoog n klastrite arvuga 
   või klastrite arvu valik tehakse automaatselt siluetikoefitsendi väärtuse alusel, sisendiks kausta failitee, kus on andmete fail ja sarnasusmaatriksi ning      valideerimistabeli fail
   
Töövoo käivitamisel on võimalik sisestada erinevaid parameetreid:
1. klastritearv - klastrite arv, milleni soovitakse mudeleid treenida
2. osakaal - kui trajektoori esinemiste arvu osakaal klastris väiksem kui sisestatud osakaal, siis need trajektoorid eemaldatakse klastrist
3. piir - kui siluetikoefitsent alla etteantud piiri, siis mudelit ei salvestata
   
