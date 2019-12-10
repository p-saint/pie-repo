# pie-repo
PIE Cartographie Aéroportuaire

Nous décrivons ici le fichier GeoJSON.

Les 3 features que nous cherchons à identifier se caractérisent par un Id (ou classe, nom fonctionnel de la feature dans le GeoJSON):
- Apron
- Runways
- Service Roads

De plus, chaque feature a une géométrie associée : 
- Polygon
- Point
- LineString

Chaque feature a une liste d'attributs :
## Apron 
-          Idapron : nom de l’apron
-          Status : usable or not usable (vient de l’AIP)
-          PCN : chaine de caractère (par ex « PCN91/F/A/W/T »)

## Runways
Les Runways sont subdivisés en plusieurs éléments.

Runway.runwayelement :
-          Idrwy : ID
-          PCN
-          Width : largeur calculé automatiquement avec la géométrie capturée
-          Length : longueur importée de l’AIP
-          Surftype : concaténation de la surface et du traitement
-          Status



Runway.runwaydisplacedarea :
-          Id
-          Surftype
-          Status
-          DisplacementLength : mesuré avec la géométrie capturée


Stopway :
-          Id
-          Surftype
-          Status
-          StopwaytLength : mesuré avec la géométrie capturée

Runway Intersection :
-          Idrwi : ID
-          Surftype
-          PCN

Runway.runwaythreshold :
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

Painted Centerline :
-          Idrwy

Runway.runwaymarking:
-          Idrwy

Runway.runwayexitline

Runway.runwayshoulder

## Service Roads
-          Idbase : toujours capturés en $UNK
-          Featbase : toujours capturé en « None »


Note sur la vérification de l'identification des features : la nature de chaque élément est vérifiée avec l'image de base, plus les charts et informations disponibles dans l'AIP (documents officiels liés à l'aéroport fournis par l'AIS). Pas de capture d'élément si la nature de l'élément n'est pas certaine à 100%.
