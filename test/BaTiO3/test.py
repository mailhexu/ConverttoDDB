#!/usr/bin/env python
import ConverttoDDB as convDDB
import os

def test_readOUTCAR():
    convDDB.readOUTCAR(outfile='OUTCAR')

def test_read_structure():
    conv=convDDB.DDBconvertor()
    #conv.read_structure('OUTCAR')

    conv.read_structure('POSCAR')
    conv.zion=[10.0,12.0,6.0]
    print(conv.unitcell)
    print("natoms:", conv.natoms)
    print("acell:", conv.acell)
    print("rprim:", conv.rprim)
    print("zion", conv.zion)
    print("znucl", conv.znucl)
    print("masspecie",conv.masspecie)
    print("massatom",conv.massatom)
    print("nametypeat",conv.nametypeat)
    print("nbpertype", conv.nbpertype)
    print("typat", conv.typat)
    print("ntypat", conv.ntypat)
    print("volume",conv.volume)
    print("rpimvasp",conv.rprimvasp)
    print("rpimd",conv.rprimd)

    conv.read_BORN('BORN')
    print(conv.bec)

    #return 
    conv.prtSYSTEM()
    conv.read_qpoints(fname='qpoints.yaml')
    conv.build_DDB_Efield()
    conv.prt_abinit_in()
    conv.runABINIT()
    conv.prtDDB()


    return
    # self.zion = []
    # self.znucl = []
    # self.masspecie = []
    # self.massatom = []
    # self.nametypeat = []
    # self.nbpertype = []
    # self.typat = []
    # self.acell = np.zeros((3))
    # self.volume = 0.0
    # self.stress = np.zeros((6))
    # self.rprimvasp = np.zeros((3, 3))
    # self.rprim = np.eye(3)
    # self.rprimd = np.zeros((3, 3))



def test_readOUTCAR():
    print os.listdir('./')
    convDDB.readOUTCAR(outfile='OUTCAR')




#test_readOUTCAR()
test_read_structure()
