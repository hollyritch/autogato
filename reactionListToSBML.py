import networkx as nx
import libsbml 
import sys 

### Copied from https://sbml.org/software/libsbml/5.18.0/docs/formatted/python-api/libsbml-python-creating-model.html 
def check(value, message):
   """If 'value' is None, prints an error message constructed using
   'message' and then exits with status code 1.  If 'value' is an integer,
   it assumes it is a libSBML return status code.  If the code value is
   LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
   prints an error message constructed using 'message' along with text from
   libSBML explaining the meaning of the code, and exits with status code 1.
   """
   if value == None:
     raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
   elif type(value) is int:
     if value == libsbml.LIBSBML_OPERATION_SUCCESS:
       return
     else:
       err_msg = 'Error encountered trying to ' + message + '.' \
                 + 'LibSBML returned error code ' + str(value) + ': "' \
                 + libsbml.OperationReturnValue_toString(value).strip() + '"'
       raise SystemExit(err_msg)
   else:
     return
####################
####################


def addSpecies(model:libsbml.Model, G:nx.DiGraph):
  # Create two species inside this model, set the required attributes
  # for each species in SBML Level 3 (which are the 'id', 'compartment',
  # 'constant', 'hasOnlySubstanceUnits', and 'boundaryCondition'
  # attributes), and initialize the amount of the species along with the
  # units of the amount.
  for n, nvalue in G.nodes.items():
    if nvalue["Type"] == "Species":
      speciesName = nvalue["Name"]
      s = model.createSpecies()
      check(s,                                  'create species s1')
      check(s.setId(n),                         'set species s id')
      check(s.setName(speciesName),             'set species name')
      check(s.setCompartment('c'),              'set species s compartment')
      check(s.setConstant(False),               'set "constant" attribute on s')
      check(s.setInitialAmount(5),              'set initial amount for s')
      check(s.setSubstanceUnits('mole'),        'set substance units for s')
      check(s.setBoundaryCondition(False),      'set "boundaryCondition" on s')
      check(s.setHasOnlySubstanceUnits(False),  'set "hasOnlySubstanceUnits" on s')
      model.addSpecies(s)
####################
####################


def addReactions(model:libsbml, G:nx.DiGraph):  
  for n, nvalue in G.nodes.items():
    if nvalue["Type"] == "Reaction":
      reactionName = nvalue["Name"]
      r = model.createReaction()
      check(r,                                              'create reaction')
      check(r.setId(n),                                     'set reaction id')
      check(r.setName(reactionName),                        'set reaction id')
      check(r.setReversible(False),                         'set reaction reversibility flag')
      check(r.setFast(False),                               'set reaction "fast" attribute')
      
      # Add Reactants
      for inEdge in G.in_edges(n):
        s = inEdge[0]
        sName = G.nodes[s]["Name"]
        speciesRef = r.createReactant()
        check(speciesRef,                                                           'create reactant')
        check(speciesRef.setSpecies(s),                                             'assign reactant species')
        check(speciesRef.setConstant(True),                                         'set "constant" on species ref 1')
        check(speciesRef.setStoichiometry(G.edges[inEdge]["Stoichiometry"]),        'set stoichiometry')
      
      # Add Products
      for outEdge in G.out_edges(n):
        s = outEdge[1]
        sName = G.nodes[s]["Name"]
        speciesRef = r.createProduct()
        check(speciesRef,                                                         'create product')
        check(speciesRef.setSpecies(s),                                           'assign product species')
        check(speciesRef.setConstant(True),                                       'set "constant" on species ref 2')
        check(speciesRef.setStoichiometry(G.edges[outEdge]["Stoichiometry"]),     'set stoichiometry')
      model.addReaction(r)
####################
####################

G = nx.DiGraph()

metabolitePath = sys.argv[1]
reactionPath = sys.argv[2]
outputPath = sys.argv[3]
formulaSpeciesDict = {}

with open(metabolitePath, "r") as file:
    line = file.readline().strip()
    while True:
        if line == "":
            break
        id, formula = line.split(" ")
        newID = "M_" + id
        G.add_node(newID)
        G.nodes[newID]["Formula"] = formula
        G.nodes[newID]["Type"] = "Species"
        G.nodes[newID]["Name"] = newID
        formulaSpeciesDict[formula]=newID
        line = file.readline().strip()

with open(reactionPath, "r") as file:
    line = file.readline().strip()
    while True:
        if line == "":
            break
        id, reactionSmile = line.split(" ")                
        if "<=>" in reactionSmile:
          lhs, rhs = reactionSmile.split("<=>")
          reversible = True          
        else:
          lhs, rhs = reactionSmile.split("=>")
          reversible = False
        newID = "R_" + id 
        G.add_node(newID)
        G.nodes[newID]["Reversible"] = reversible
        reactants = lhs.split("|")
        products = rhs.split("|")
        for reac in reactants:
            reacID = formulaSpeciesDict[reac]
            if (reacID, newID) in G.edges():
                G.edges[(reacID, newID)]["Stoichiometry"] +=1
            else:
                G.add_edge(reacID, newID)
                G.edges[(reacID, newID)]["Stoichiometry"] =1
        for prod in products:
            prodID = formulaSpeciesDict[prod]
            if (newID, prodID) in G.edges():
                G.edges[(newID, prodID)]["Stoichiometry"] +=1
            else:
                G.add_edge(newID, prodID)
                G.edges[(newID, prodID)]["Stoichiometry"] =1        
        G.nodes[newID]["Type"] = "Reaction"
        G.nodes[newID]["Name"] = newID
        line = file.readline().strip()


##################################################################################################################################################################
##########                                                          Main                                                                                ##########
##################################################################################################################################################################

try:
    document = libsbml.SBMLDocument(3, 1)
except ValueError:
    raise SystemExit('Could not create SBMLDocumention object')

model = document.createModel()
check(model, 'create model')

# Create a unit definition we will need later.  Note that SBML Unit
# objects must have all four attributes 'kind', 'exponent', 'scale'
# and 'multiplier' defined.
 
per_second = model.createUnitDefinition()
check(per_second,                                   'create unit definition')
check(per_second.setId('per_second'),               'set unit definition id')
unit = per_second.createUnit()
check(unit,                                         'create unit on per_second')
check(unit.setKind(libsbml.UNIT_KIND_SECOND),       'set unit kind')
check(unit.setExponent(-1),                         'set unit exponent')
check(unit.setScale(0),                             'set unit scale')
check(unit.setMultiplier(1),                        'set unit multiplier')

# Create a compartment inside this model, and set the required
# attributes for an SBML compartment in SBML Level 3.

c = model.createCompartment()
check(c,                                           'create compartment')
check(c.setId('c1'),                               'set compartment id')
check(c.setConstant(True),                         'set compartment "constant"')
check(c.setSize(1),                                'set compartment "size"')
check(c.setSpatialDimensions(3),                   'set compartment dimensions')
check(c.setUnits('litre'),                         'set compartment size units')

addSpecies(model, G)
addReactions(model, G)


writer = libsbml.SBMLWriter()
writer.writeSBMLToFile(document, outputPath)