import sys, os
from tqdm import tqdm 

#========================================================================#
#                               Test                                     #
#========================================================================#

# 1. Partition network
# 1.1 Define parameters
o = "EColiCore"
input = "./XML-Files/EColiCore/e_coli_core.xml"
c = 2
t = 16
s = "./SmallMolecules/smallMoleculesEColiCore.txt"

# 1.2 Make sure output directories exit
if os.path.exists("./Results") == False:
    os.makedirs("./Results")

# 1.3 Run partitioning
os.system("python partitionNetwork.py " + " -i " + input + " -c " + str(c) + " -t " + str(t) + " -o " + o + " -s " + s)


# 2. Analysis (example to run them all) 
# 2.1 Define paarameters 
b = 1000
e = 1000
o = "cycleData"

pickleFileList = os.listdir("PickleFiles/" + s)
for j in tqdm(range(len(pickleFileList)), leave = False, desc="Pickle-Files"):
    pickleFilePath = "PickleFiles/"+ s +  "/" + pickleFileList[j]
    newOrder = "python partitionAnalysis.py" + " -x " + input + " -i " + pickleFilePath + " -b " + str(b) + " -n True " + " -s " + s + " -t " + str(t) + " -e " + str(e) +  " -p" + " -o " + o
    
    # Make sure your output directory exits
    if not os.path.exists("./Results/"+s):
        os.makedirs("./Results/"+s)
    os.system(newOrder)