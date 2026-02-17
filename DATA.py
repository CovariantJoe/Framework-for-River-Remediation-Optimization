"""
Placeholder file to share important data across scripts
Per the dictionaries below and throughout the project the indices from 0 to 3 represent substance pairs: 
0 : Biochar / Glyphosate
1 : Egg shell / Malathion
2 : Montmorillonite / Aldrin
3 : Leaves / Malathion

The dictionary key "Conc" indicates the pesticide concentration to fill the river to. This is used frequently

Units: 
qmax is in [g/g], b in [m3/g], and k1 in [1/s]
rho_s is in [g/m3]
Conc is in [g/m3], Dm in [m^2/s],
"""

from pathlib import Path as path


SOURCE1 = "Adsorptive features of acid-treated olive stones for drin pesticides: Equilibrium, kinetic and thermodynamic modeling studies"
SOURCE2 = "Equilibrium and kinetic mechanisms of woody biochar on aqueous glyphosate removal"
SOURCE3 = "Adsorption of malathion on thermally treated egg shell material"
SOURCE4 = "Assessment of Pesticide Contamination in the Water of the Santiago River, Mexico"
SOURCE5 = "Use of the Pesticide Toxicity Index to Determine Potential Ecological Risk in the Santiago-Guadalajara River Basin, Mexico"
SOURCE6 = "Characterization and adsorption properties of eggshells and eggshell membrane"
SOURCE7 = "An investigation on the sorption behaviour of montmorillonite for selected organochlorine pesticides from water"
SOURCE8 = "Elastic properties of dry clay mineral aggregates, suspensions and sandstones"
SOURCE9 = "New approaches to measuring biochar density and porosity"
SOURCE10 = "Improving the Interpretation of Small Molecule Diffusion Coeffients"
SOURCE12 = "Scale‑up approach for supercritical fluid extraction with ethanol–water modified carbon dioxide on Phyllanthus niruri for safe enriched herbal extracts"
SOURCE13 = "Biosorption of Malathion from Aqueous Solutions Using Herbal Leaves Powder"
SOURCE11 = "Use of Weighted Least‐Squares Method in Evaluation of the Relationship Between Dispersivity and Field Scale"

Adsorbents = [
{"Name":"Biochar","Source":SOURCE2, "Pollutant":"Glyphosate","Model":1,"Values":{"qmax":44.01/1000,"b":0.088/1000,"k1":0.038/60}, "Transport":{"rho_s":1.735*100**3,"porosity":0.70,"Source":SOURCE9}},
{"Name":"Egg shell","Source":SOURCE3, "Pollutant":"Malathion","Model":1, "Values":{"qmax":1.3/1000*330.4,"b":(3.549*1000/330.4)/1000,"k1":0.0286/60}, "Transport":{"rho_s":2.532*100**3,"porosity":0.0162,"Source":SOURCE6} },
{"Name":"Montmorillonite", "Source":SOURCE7, "Pollutant":"Aldrin","Model":2, "Values":{"k":14.259/1000,"n":0.945,"k1":0.013/60},"Transport":{"rho_s":-2/(0.13 - 1)*100**3,"porosity":0.13,"Source":SOURCE8}},
{"Name":"Leaves", "Source":SOURCE13, "Pollutant":"Malathion","Model":2, "Values":{"k":0.6569,"n":1.0930,"k1":0.025/60},"Transport":{"rho_s":1275.77*1000,"porosity":0.86,"Source":SOURCE12} }

]

Pollutants = [{"Name":"Glyphosate", "Conc":(278/1e6)*1000,"Dm":7.04e-10,"SourceDm":SOURCE10,"SourceConc":SOURCE4},
              {"Name":"Malathion","Conc":(810/1e6)*1000,"Dm":5.24e-10,"SouceDm":SOURCE10,"SourceConc":SOURCE4},
              {"Name":"Aldrin","Conc":(35.35/1e9)*1000,"Dm":5.02e-10,"SourceDm":SOURCE10,"SourceConc":SOURCE5},
              {"Name":"Malathion","Conc":(810/1e6)*1000,"Dm":5.24e-10,"SouceDm":SOURCE10,"SourceConc":SOURCE4}] 
                                    
Path = str(path(__file__).resolve().parent) # This path will always be used by the scripts
MeshPath = Path + "/mesh.xml"

# Do not modify MeshData here, it's better to pass your own mesh directly when calling the function in velocity.py or Main.py
MeshData = {"xml":MeshPath,"DBC": [[-38,-816],[78,-883] ,[-11,-5],[10,-10]]}
