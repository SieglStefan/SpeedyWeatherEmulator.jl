using JLD2

# Datei öffnen (Pfad anpassen falls nötig)
file = jldopen("run_0001/output.jld2", "r")

# Alle gespeicherten Variablen auflisten
keys(file)

out = file["output_vector"]   # holt das gesamte Output-Objekt
close(file)

#display(typeof(out))             # zeigt dir, welcher Struct das ist
#display(fieldnames(typeof(out))) # listet die Felder im Struct auf

prog1, diag1 = out[1]
prog2, diag2 = out[2] 
prog3, diag3 = out[3]  
prog4, diag4 = out[4]  
# erstes Tupel zerlegen (erster Zeitschritt)

prog_end, diag_end = out[25]


# Spektrale Vorticity (Komplexkoeffizienten der Kugelflächenfunktionen)
vor = prog_end.vor[:,:,1]

display(vor)
display(collect(vor))


