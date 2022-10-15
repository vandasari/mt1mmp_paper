import mt1mmp as mo


imageProperties = [{"DisplayImages": "Yes"}, {"SaveImages": "No", "SaveMCS": 10}]

pottsElements = [{"Name":"Dimension","xDim": 100, "yDim": 100, "zDim": 0},
                {"Steps": 1000},
                {"Temperature": 10},
                {"NeighborOrder": 2},
                {"Boundary": 2}] #-- 1: Periodic --#
                                 #-- 2: ZeroFlux --#

cellArr = [{"NumberCellsX": 1, "NumberCellsY": 1},
           {"LengthCellX": 20, "LengthCellY": 20},
           {"Gap": 0},
           {"XCenter": 50, "YCenter": 50}]

contactEnergy = [{"Type1":"Medium", "Type2":"Medium", "Value": 0.0},
                {"Type1":"Medium", "Type2":"Cancer", "Value": 0.0},
                {"Type1":"Cancer", "Type2":"Cancer", "Value": 1.0},
                {"DynamicTime": 400}]

volInit = [{"TargetVolume": 450, "LambdaVolume": 5.0},
    {"PluginType": 1}, #-- 1: VolumePlugin; 2: VolumeLocalFlex --#
    {"Circularity": 0.4}]

surfInit = [{"LambdaSurface": 5.0}]

camStruct = [{"CellType":"Medium", "CellID":0, "Molecule":"Collagen", "Density": 50.0},
            {"CellType":"Cancer", "CellID":1, "Molecule":"Integrin", "Density": 2.0},
            {"CellType":"Cancer", "CellID":1, "Molecule":"Cadherin", "Density": 4.0},
            {"BindingAffinity":"kcc", "Value": -1.0},
            {"BindingAffinity":"kcm", "Value": -7.0}]

membraneParams = [{"MembraneMT1": 0, "LambdaMT1": 300.0}]


mo.cpm(pottsElements, cellArr, contactEnergy, camStruct, volInit, surfInit, membraneParams, imageProperties)


