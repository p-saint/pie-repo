# pie-repo
PIE Cartographie Aéroportuaire


Caractérisations des 3 features à détecter par 3 Id (ou classe, nom fonctionnel de la feature dans le GeoJSON), chaque feature a une géométrie associée (uniquement 3 type de géométries : POINT/LINE/POLYGON).

Liste des features :
** Apron
Attributs : 
-          Idapron : nom de l’apron
-          Status : usable or not usable (vient de l’AIP)
-          PCN : chaine de caractère (par ex « PCN91/F/A/W/T »)

** Runways
Divisés en plusieurs éléments, chaque élément a les attributs :

*         Runway element :
-          Idrwy : ID
-          PCN
-          Width : largeur calculé automatiquement avec la géométrie capturée
-          Length : longueur importée de l’AIP
-          Surftype : concaténation de la surface et du traitement
-          Status


*         Runway Displaced Area :
-          Id
-          Surftype
-          Status
-          DisplacementLength : mesuré avec la géométrie capturée


*         Stopway :
-          Id
-          Surftype
-          Status
-          StopwaytLength : mesuré avec la géométrie capturée

*         Runway Intersection :
-          Idrwi : ID
-          Surftype
-          PCN

*         Runway Threshold :
-          Idthr
-          Thrtype : displaced or not
-          Status
-          Brngtrue : bearing mesurés sur la géométrie capturée
-          Brngmag : bearing importé de l’AIP
-          Tora/Asda/Lda : declared distances importées de l’AIP
-          Tdze : importé de l’AIP
-          AvailPavedSurfFromThr : mesuré avec la géométrie capturée
-          MeasuredLda/Tora/Lda : declared distances mesurées avec la géométrie capturée
-          Elevation : importé avec le DTM disponible sur l’image on infotmations Geo si DTM absent
-          Cat : catégorie de l’ILS lié à ce thr
-          ROPS landing length : égale à la valeur de la measuredLda sauf si demande de mesure particulière d’un client.

*         Painted Centerline :
-          Idrwy

*         Runway Marking
-          Idrwy


** Service Roads
Attributs:
-          Idbase : toujours capturés en $UNK
-          Featbase : toujours capturé en « None »


Note sur la vérification de l'identification des features : la nature de chaque élément est vérifiée avec l'image de base, plus les charts et informations disponibles dans l'AIP (documents officiels liés à l'aéroport fournis par l'AIS). Pas de capture d'élément si la nature de l'élément n'est pas certaine à 100%.
