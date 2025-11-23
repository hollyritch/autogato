import sys, os
from tqdm import tqdm 

cutOffElementaryCircuits = 40
cutOffLargerCycles = 40
maxThreads = 12
cutOffReactionNetworkSize = 2
path = "./XML-Files/"
speciesList = sorted(list(os.listdir(path)))
speciesXMLDict = {}
for k in tqdm(range(len(speciesList)), desc="Species"):
    species = speciesList[k]    
    print(species)
    if species != "EColiDH5Alpha":
        continue
    if species == ".DS_Store":
        continue
    if species == "arabidopsisThaliana":
        continue
    if species == "PhaelodactylumTricornicutum":
        continue
    speciesPath = path+species+"/"
    xmlFileList =  os.listdir(speciesPath)
    if species == "EColiCore":
        smallMoleculesPath = "smallMoleculesEColiCore.txt"
    else:
        smallMoleculesPath = "smallMolecules.txt"
    for i in tqdm(range(len(xmlFileList)), leave = False, desc="XML-Files"):
        xmlFile =  xmlFileList[i]
        print(species, xmlFile)
    # 2. Read Model
        xmlFilePath = speciesPath + xmlFile
        if os.path.exists("./Results") == False:
            os.makedirs("./Results")
        partitioningOutputFile = "./Results/partitioningOutput" + species + xmlFile+".txt"
        os.system("python partitionNetwork.py " + " -i " + xmlFilePath + " -c " + str(cutOffReactionNetworkSize) + " -t " + str(maxThreads) + " -o " + str(species) + " -s ./SmallMolecules/" + str(smallMoleculesPath) + " | tee " + str(partitioningOutputFile))
        pickleFileList = os.listdir("PickleFiles/" + str(species))
        analysisOutputFile = "./Results/analysisOutput" + species + xmlFile+ ".txt"
        for j in tqdm(range(len(pickleFileList)), leave = False, desc="Pickle-Files"):
            pickleFilePath = "PickleFiles/"+str(species) +  "/" + pickleFileList[j]
            
            newOrder = "python partitionAnalysis.py" + " -x " + xmlFilePath + " -i " + pickleFilePath + " -b " + str(cutOffElementaryCircuits) + " -n True " + " -s " + str(species)+ " -t " + str(maxThreads) + " -e " + str(cutOffLargerCycles) +  " -c"  + " -p" + " -d /scratch/richard/Autocatalysis/cycleData/"
            print(newOrder)
            if not os.path.exists("./Results/"+str(species)):
                os.makedirs("./Results/"+str(species))
            os.system('/usr/bin/time -f "Command: %C\nTime   : %E\nRAM(kb): %M\nStatus : %x\n" -o ./Results/'+str(species)+'/outputTimeRAM'+ str(j) + '.txt ' +newOrder)