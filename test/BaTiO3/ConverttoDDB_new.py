# Version 0.1
# Developers:
#
# -Aldo Romero: alromero@mail.wvu.edu
# -Eric Bousquet


# Script to build a DDB file for anaddb from Born effective charges (Z*),
# electronic dielectric constant (eps) and dynamical matrix (Dyn) of external code
# Z* and eps are/will be stored in the BORN file (phonopy format)
# Dyn have to be stored in DYNMAT file for each q-point
#  V0:  Eric Bousquet in Perl
#  B0.1:  A.H. Ronero in Python and generalization to any crystal cell
#
# Option n = 1 build the DDB from the BORN and OUTCAR files
# Option n = 2 build the DDB and the BORN files from the OUTCAR file
# Option n = 3 build an input for abinit and run it providing the psp's files and the correct abinit.files
# Option n = 4 build the DDB from from system.dat file, BORN file and qpoints.yaml



import numpy as np
import glob
import os
import os.path
import copy
import shutil
import subprocess
import yaml
import argparse
#from optparse import OptionParser


Ang2Bohr=1/0.5291772108; Bohr2Ang=0.5291772108; ev2Ha=1/27.2113845; Ha2eV=27.2113845;
zion=[]; znucl=[]; masspecie=[]; massatom=[]; nametypeat=[]; nbpertype=[]; typat=[]; acell=np.zeros((3));
rprimvasp=np.zeros((3,3))
rprim=np.zeros((3,3))
rprimd=np.zeros((3,3))
np.fill_diagonal(rprim, 1)
opt_n=1

# functions
# SUBROUTINES ##############################################################
############################################################################

############################################################################
# Mendeley Table
def Mendel_data(i):

   mendel=["-", "H","He",
       "Li","Be","B","C","N","O","F","Ne",
       "Na","Mg","Al","Si","P","S","Cl","Ar",
       "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
       "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
       "Cs","Ba",
       "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
       "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
       "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No",
       "Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Uut"]

   return mendel[i]

   ############################################################################
# Run Abinit

def runABINIT():
# Run abinit
    os.system("abinit <abinit.files> log_abinit")
    os.system("cp abinit-o_DDB abinit-o_DDB.swp")
    os.system("rm abinit-o_OUT.nc")

############################################################################
# Print BORN file
def prtBORN():

    filename="BORN"
    if (os.path.isfile(filename)):
        print "Found an existing BORN file, writing a new one"
        filename=filename+".new"
    born=open(filename,"w")
    born.write("1.0\n")

# print epsilon^infty
    for i in range(3):
       born.write("%.8f   %.8f   %.8f"%(eps_e[i,0],eps_e[i,1],eps_e[i,2]))
    born.write("\n")

# print Z*
    for ion in range(natoms):
       for i in range(3):
          born.write("%.8f   %.8f   %.8f"%(bec[ion,i,0],bec[ion,i,1],bec[ion,i,2]))
       born.write("\n")
    born.close()

############################################################################
# Print SYSTEM file
def prtSYSTEM():

    filename="system.dat"
    if (os.path.isfile(filename)):
        print "File system.dat exist, writing a new one"
        filename=filename+".new"
    system=open(filename,'w')
    system.write("natoms %d \n"%natoms) 
    system.write("ntypat %d \n"%ntypat) 
    system.write("zion ") 
    for i in range(ntypat):
        system.write(" %d "%zion[i])
    system.write("\n")
    system.write("znucl ") 
    for i in range(ntypat):
        for j in range(112):
            if (Mendel_data(j) == nametypeat[i]):
                system.write(" %d  "%j)
    system.write("\n")
    system.write("mass ") 
    for i in range(ntypat):
        system.write(" %.6f "%masspecie[i])
    system.write("\n")
    system.write("typat ") 
    for i in range(ntypat):
        system.write(" %d*%d "%(nbpertype[i],i+1))
    system.write("\n")
    system.write("acell %.8f %.8f %.8f Angstrom\n"%(acell[0],acell[1],acell[2]))
    system.write("\n")
    system.write("rprim \n")
    for i in range(3):
        system.write("%.8f %.8f %.8f\n"%(rprim[i,0],rprim[i,1],rprim[i,2]))
    system.write("\n")
    system.write("xred \n")
    for i in range(natoms):
        system.write("%.8f %.8f %.8f\n"%(xred[i,0],xred[i,1],xred[i,2]))
    system.write("\n")
    system.write("forces \n")
    for i in range(natoms):
        system.write("%.8f %.8f %.8f\n"%(forces[i,0],forces[i,1],forces[i,2]))
    system.close()

############################################################################
# Read qpoints.yaml file (from Phonopy)

def read_qpoints():
#output variables
#dynmat; # store the Dynamical Matrix for each q-point
#phi; # store the Force constant matrix for each q-point = dynmat/srqt(m_i*m_j)
    global dynmat
    global phi
    global nqs
    global qpt
    global DDB

    q=0 # store the q-point number

    nbmode=3*natoms
    OK_dyn=0

    QFILE='./qpoints.yaml'
    if os.path.isfile(QFILE) and os.access(QFILE, os.R_OK):
        data = yaml.load(open(QFILE))
        nqs=data['nqpoint']
        if (natoms==0):
            raise Exception('natoms is not defined while qpoints.yaml are going to be read. Meaning system is not defined')
        latt=np.square(acell*Ang2Bohr) # factor to get the DDB for abinit = acell**2 on idir1 and idir2
        print '  LATT = ',latt
        print "rprimvasp = ",rprimvasp
        print "Ang2Bohr = ",Ang2Bohr
        qpt=[]
        for i in range(nqs):
            qi=np.array(data['phonon'][i]['q-position'])
            qpt.append(qi)
            print "q point ",qi," read"
        qpt=np.array(qpt)
        dynmat=np.zeros((nqs,3,natoms,3,natoms),dtype=np.complex_)
        phi=np.zeros((nqs,3,natoms,3,natoms),dtype=np.complex_)
        DDB=np.zeros((nqs+2,3,natoms+2,3,natoms+2),dtype=np.complex_)

        for iq in range(nqs):
            dynmat_data = data['phonon'][iq]['dynamical_matrix']
            if (len(dynmat_data) >0):
                OK_dyn=1
                dm = []
                for row in dynmat_data:
                    vals = np.reshape(row, (-1, 2))
                    dm.append(vals[:, 0] + vals[:, 1] * 1j)
                dm = np.array(dm)
                for ipert1 in range(natoms):
                    for idir1 in range(3):
                        ii=idir1+ipert1*3
                        for ipert2 in range(natoms):
                            for idir2 in range(3):
                                jj=idir2+ipert2*3
                                if (iq==2):
                                    print idir1+1,ipert1+1,idir2+1,ipert2+1,ii+1,jj+1,dm[ii,jj]
                                dynmat[iq,idir1,ipert1,idir2,ipert2]=dm[ii,jj]
                                phi[iq,idir1,ipert1,idir2,ipert2]=dm[ii,jj]*np.sqrt(massatom[ipert1]*massatom[ipert2]);
                                DDB[iq,idir1,ipert1,idir2,ipert2]=phi[iq,idir1,ipert1,idir2,ipert2]*ev2Ha/(Ang2Bohr*Ang2Bohr)*latt[idir1]
                                if (iq==2):
                                    print idir1+1,ipert1+1,idir2+1,ipert2+1,DDB[iq,idir1,ipert1,idir2,ipert2]
                        i=i+1
            else:
                raise Exception('dynamical matrix does not exist in qpoints.yaml')
    else:
        raise Exception('Either qpoints.yaml file is missing or is not readable')

############################################################################
# Read BORN file
def readBORN():
    global bec
    global eps_e

    bfile='BORN'
    if os.path.isfile(bfile) and os.access(bfile, os.R_OK):
        eps_e=np.zeros((3,3))
        bec=np.zeros((natoms,3,3))
        with open(bfile) as f:
             b = f.readlines()
        f.close()
        eps_e=np.array(map(float,b[1].split())).reshape(3,3)
        for i in range(natoms):
            bec[i,:,:]=np.array(map(float,b[2+i].split())).reshape(3,3)
        print "\n ------------------- "
        print "Read from BORN file:"
        print "\nnatoms = ",natoms
        print "\nepsilon electronic:"
        for i in range(3):
            print "%d %.10f %.10f %.10f"%(i+1,eps_e[i,0],eps_e[i,1],eps_e[i,2])
        print "\nBorn effective charges:\nIon Dir1 BEC_[idir1,idir2]"
        for ion in range(natoms):
            for i in range(3):
                print "%d %d %.10f %.10f %.10f"%(ion+1,i+1,bec[ion,i,0],bec[ion,i,1],bec[ion,i,2])
    else:
        raise Exception('error when openning BORN file: it does not exist!')

############################################################################
# Read OUTCAR file
def readOUTCAR():
    global xred
    global natoms
    global ntypat
    global bec
    global eps_e
    global zion
    global nametypeat
    global nbpertype
    global znucl
    global forces

    outfile='OUTCAR'
    nbpertype=[]
    zion=[]
    xred=[]
    latt=[]
    lattrec=[]
    eps_e=[]
    natoms=0

    OK_nbpertype=0; OK_mass=0; OK_bec=0; OK_eps_e=0;
    OK_eps_i=0; OK_valenz=0; OK_xred=0; OK_latt=0; OK_force=0;

    if os.path.isfile(outfile) and os.access(outfile, os.R_OK):
        f = open(outfile,'r')
        for line in f.readlines():
            if "vasp.5" in line:
               version=5
               print "Vasp Version = ",line.split()[0]
            else:
               version=4
            if "NIONS" in line:
               npos=line.find("NIONS")
               natoms=int(line[npos:].split()[2])
               bec=np.zeros((natoms,3,3))
               forces=np.zeros((natoms,3))
               print "natoms = ",natoms
               if natoms==0:
                 raise Exception('error number of atoms, it is zero, going to stop')
            if "VRHFIN" in line:
               s=line.split()[1]
               s=s.replace("=","")
               s=s.replace(":","")
               nametypeat.append(s)
            if (OK_force == 1 or OK_force == 2):
                OK_force=OK_force+1
            if (OK_force>2):
                d=map(float,line.split())
                forces[OK_force-3,:]=d[3:]
                OK_force=OK_force+1
                if (OK_force>(natoms+2)):
                    OK_force=0
            if ("ions per type" in line and OK_nbpertype==0):
               l=line.split()
               for i in range(4,len(l)):
                  nbpertype.append(int(l[i]))
                  OK_nbpertype=1
            if (OK_mass==1):
               l=line.split()
               for i in range(len(nbpertype)):
                  masspecie.append(float(l[i+2]))
               OK_mass=2
            if (OK_valenz==1):
               l=line.split()
               for i in range(len(nbpertype)):
                  zion.append(float(l[i+2]))
               OK_valenz=2
            if (OK_xred <= natoms and OK_xred>0):
               l=line.split()
               xred.append(map(float,l))
               OK_xred=OK_xred+1
            if (OK_latt < 5 and OK_latt > 1):
               l=line.split()
               s=map(float,l)
               rprimvasp[OK_latt-2,:]=s[0:3]
               lattrec.append(s[3:])
               OK_latt=OK_latt+1
            if (OK_eps_e >1 and OK_eps_e < 5):
               d=map(float,line.split())
               eps_e.append(d)
               OK_eps_e=OK_eps_e+1
            if (OK_bec >1 and OK_bec < 2+4*natoms):
               index=(OK_bec-2)%4
               if (index>0):
                   d=map(float,line.split())
                   bec[int((OK_bec-2)/4),index-1,:]=d[1:]
               OK_bec=OK_bec+1

#################################
######### Epsilon and Z* ########
##
##
            if ("Mass of Ions" in line and OK_mass==0):
               OK_mass=1
            if ("Ionic Valenz" in line and OK_valenz==0):
               OK_valenz=1
            if ("TOTAL-FORCE (eV/Angst)" in line):
               OK_force=1
            if ("The static configuration has the point symmetry" in line):
               s=line.split()[7]
               s=s.replace(".","")
               point_sym=s
            if ("position of ions in fractional coordinates" in line):
               OK_xred=1
            if "direct lattice vectors" in line:
               OK_latt=OK_latt+1
            if ("MACROSCOPIC STATIC DIELECTRIC TENSOR" in line and "including local field effects in DFT" in line):
               print "electronic epsilon found!"
               OK_eps_e=OK_eps_e+1
            if (OK_eps_e==1 and "-----------------------------" in line):
               OK_eps_e=OK_eps_e+1
            if ("BORN EFFECTIVE CHARGES" in line and "cummulative output" in line):
               OK_bec=OK_bec+1
            if (OK_bec==1 and "-----------------------------" in line):
               OK_bec=OK_bec+1
        ntypat=len(nbpertype)
        print "number of species = ",ntypat
        print "atom per specie =",nbpertype
        print "nametypeat=",nametypeat
        for i in range(ntypat):
            for j in range(112):
                if (Mendel_data(j) == nametypeat[i]):
                    znucl.append(j)
        print "znucl = ",znucl
        print "Valence=",zion
        print "masspeci=",masspecie
        print "forces = ",forces
        j=0
        for i in range(ntypat):
            for k in range(nbpertype[i]):
                massatom.append(masspecie[i])
                j=j+1
        print "massatom = ",massatom
        print "point symmetry = ",point_sym
        for i in range(3):
            d=np.sqrt(np.dot(rprimvasp[i,:],rprimvasp[i,:]))
            acell[i]=d
            rprimd[:,i]=rprimvasp[i,:]
            rprim[:,i]=rprimvasp[i,:]/acell[i]
        print "rprimvasp = ",rprimvasp
        print "rprimd = ",rprimd
        print "rprim = ",rprim
        print "acell = ",acell
        print "lattrec = ",lattrec
        if (OK_eps_e>0):
           eps_e=np.array(eps_e)
           print "eps_e =",eps_e
        if (OK_bec>0):
           print "bec =",bec
        lattrec=np.array(lattrec)
        xred=np.array(xred)
        f.close()
        if (OK_bec==0 and opt_n==2):
            raise Exception('No BEC found in OUTCAR!\n This is not possible with option n=2 \n The program is going to exit')
        if (OK_eps_e==0 and opt_n==2):
            raise Exception('No Electronic Dielectric Constant found in OUTCAR!\n This is not possible with option n=2 \n The program is going to exit')
    else:
        raise Exception('error when openning OUTCAR file: it does not exist!')



############################################################################

def readSYSTEM():
    global xred
    global natoms
    global ntypat
    global forces

    OK_xred=0
    OK_rprim=0
    sfile='system.dat'

    if os.path.isfile(sfile) and os.access(sfile, os.R_OK):
       with open(sfile,'r') as f:
           data=f.readlines()
       print "\n Read from system.dat file:"
       for line in data:
           if "natom" in line:
               d=line.split()
               natoms=int(d[1])
               forces=np.zeros((natoms,3))
               print "natom = %d"%natoms
           if "ntypat" in line:
               d=line.split()
               ntypat=int(d[1])
               print "ntypat = %d"%ntypat
           if ("acell" in line):
               d=line.split()
               acell[:]=map(float,d[1:4])
               print "acell = ",acell
       if (ntypat == 0 or natoms == 0 or np.prod(acell) == 0.0):
           raise Exception('error in system.dat ntypat or natom or acell')
       j=0
       xred=np.zeros((natoms,3))
       while True:
           line=data[j]
           if "zion" in line:
               d=line.split()
               if len(d)<ntypat+1:
                  raise Exception('error in system.dat ntypat does not coincide with zion ')
               else:
                  for i in range(ntypat):
                      zion.append(float(d[i+1]))
                  print "zion = ",zion
           if "znucl" in line:
               d=line.split()
               if len(d)<ntypat+1:
                  raise Exception('error in system.dat ntypat does not coincide with znucl ')
               else:
                  for i in range(ntypat):
                      znucl.append(float(d[i+1]))
                      nametypeat.append(Mendel_data(int(d[i+1])))
                  print "znucl = ",znucl
                  print "nametypeat = ",nametypeat
           if "mass" in line:
               d=line.split()
               if len(d)<ntypat+1:
                  raise Exception('error in system.dat ntypat does not coincide with number of elements in mass ')
               else:
                  for i in range(ntypat):
                      masspecie.append(float(d[i+1]))
                  print "mass per specie = ",masspecie
           if ("atom" in line and "name" in line):
               d=line.split()
               if len(d)<ntypat+1:
                  raise Exception('error in system.dat ntypat does not coincide with number of elements in atom name ')
               else:
                  for i in range(ntypat):
                      nametypeat.append(float(d[i+1]))
                  print "atom nametypeats = ",type
           if "typat" in line and "ntypat" not in line:
               d=line.split()
               s=d[1:]
               if len(s) == ntypat:
                   for x in s:
                       if "*" in x:
                          splt=x.split("*")
                          for i in range(int(splt[0])):
                              typat.append(int(splt[1]))
                       else:
                          typat.append(int(x))
                   for i in range(1,ntypat+1):
                       nbpertype.append(typat.count(i))
                   print "typat=",typat
                   print "nbpertype=",nbpertype
               else:
                  raise Exception('error in system.dat ntypat does not coincide with number of elements in typat ')
           if ("rprimvasp" in line):
               OK_rprim=1
               d=line.split()
               if len(d)==1:
                  for i in range(3):
                     j=j+1
                     line=data[j]
                     d=map(float,line.split())
                     rprimvasp[i,:]=d
               else:
                  rprimvasp[0,:]=map(float,d[1:])
                  for i in range(2):
                     j=j+1
                     line=data[j]
                     rprimvasp[i+1,:]=map(float,line.split())
               if (np.prod(acell) != 1.0):
                   for i in range(3):
                       rprimvasp[i,:]=rprimvasp[i,:]*acell[i]
               for i in range(3):
                       rprimd[:,i]=rprimvasp[i,:]
               for i in range(3):
                   acell[i]=np.sqrt(np.dot(rprimvasp[i,:],rprimvasp[i,:]))
                   rprim[:,i]=rprimvasp[i,:]/acell[i]
               print "rprim = ",rprim
               print "rprimd = ",rprimd
               print "rprimvasp = ",rprimvasp
           if ("rprim" in line and "vasp" not in line and "rprimd" not in line):
               OK_rprim=1
               d=line.split()
               if len(d)==1:
                  for i in range(3):
                     j=j+1
                     line=data[j]
                     d=map(float,line.split())
                     rprim[i,:]=d
               else:
                  rprim[0,:]=map(float,d[1:])
                  for i in range(2):
                     j=j+1
                     line=data[j]
                     rprim[i+1,:]=map(float,line.split())
               for i in range(3):
                   rprimd[:,i]=rprim[:,i]*acell[i]
               for i in range(3):
                   rprimvasp[i,:]=rprim[:,i]
               print "rprim = ",rprim
               print "rprimd = ",rprimd
               print "rprimvasp = ",rprimvasp
           if ("rprimd" in line):
               raise Exception('reading rprimd from system.dat is not yet implemented')
           if "xred" in line:
               d=line.split()
               if len(d)==1:
                  for i in range(natoms):
                     j=j+1
                     line=data[j]
                     xred[i,:]=map(float,line.split())
               else:
                  xred[0,:]=map(float,d[1:])
                  for i in range(1,natoms):
                     j=j+1
                     line=data[j]
                     xred[i,:]=map(float,line.split())
               print "xred="
               for i in range(natoms):
                   print xred[i,:]
           if "forces" in line:
               d=line.split()
               if len(d)==1:
                  for i in range(natoms):
                     j=j+1
                     line=data[j]
                     forces[i,:]=map(float,line.split())
               else:
                  forces[0,:]=map(float,d[1:])
                  for i in range(1,natoms):
                     j=j+1
                     line=data[j]
                     forces[i,:]=map(float,line.split())
               print "xred="
               for i in range(natoms):
                   print forces[i,:]
           if (j+1<len(data)):
               j=j+1
           else:
               break
    else:
       raise Exception('error when openning system.dat file: it does not exist!')
    for i in range(natoms):
        massatom.append(masspecie[typat[i]-1])
    if (OK_rprim==0):
        for i in range(3):
            rprimvasp[i,:]=rprim[:,i]*acell[i]
        print "rprimvasp = ",rprimvasp
    print "mass atom = ",massatom
    print "***************** typat = ",typat

############################################################################
# Print ABINIT file
def prt_abinit_in():
    abin=open("abinit.in","w")
    abin.write("natom %d\n"%natoms)
    abin.write("ntypat %d\n"%ntypat)
    abin.write("znucl ")
    for i in range(ntypat):
        for j in range(112):
            s=Mendel_data(j)
            if (s == nametypeat[i]):
               abin.write(" %d"%j)
    abin.write("\n")
    abin.write("typat ")
    for i in range(ntypat):
        abin.write(" %d%s%d"%(nbpertype[i],'*',i+1))
    abin.write("\n")
    abin.write("acell ")
    for i in range(3):
        abin.write(" %.8f"%acell[i])
    abin.write(" Angstrom");
    abin.write("\n")
    abin.write("rprim ")
    for i in range(3):
        for j in range(3):
            abin.write(" %.8f"%rprim[i,j])
        abin.write("\n")
    abin.write("xred \n")
    for i in range(natoms):
        abin.write("%.8f  %.8f  %.8f\n"%(xred[i,0],xred[i,1],xred[i,2]))
    abin.write("\n")
    abin.write("ecut   5\nkptopt 1\nngkpt  1 1 1\nixc    1\nnstep  1\ntoldfe 1.0E+05\niscf   5\nprtden 0\nprtwf  0\nprteig 0\n")
    abin.close()

    abfiles=open("abinit.files","w")
    abfiles.write("abinit.in\nabinit.out\nabinit-i\nabinit-o\nabinit-t\n")
    for i in range(ntypat):
        abfiles.write("%s.psp\n"%Mendel_data(int(znucl[i])))
    abfiles.close()

############################################################################
# Build DDB from epsilon^infty and Z*
# The E-field DDB's are store in qpt=0
def build_DDB_Efield():

# first get the type number of each atom:
    k=0
    type_atom=[0]*natoms
    for i in range(ntypat):
        for j in range(nbpertype[i]):
            type_atom[k]=i
            k=k+1
    print "\n----------------------------\n DDB for e-field:"

    for ipert1 in range(natoms+2):
       for idir1 in range(3):
          for idir2 in range(3):
             for ipert2 in range(natoms+2):
                if (ipert1==natoms+1 and ipert2 != natoms and ipert2 != natoms+1):   # mixed E/R pert <--> Z*
                   if (abs(bec[ipert2,idir1,idir2])<=1.0E-8):
                       DDB[0,idir1,ipert1,idir2,ipert2]=0.0+0.0j
                   else:
                       DDB[0,idir1,ipert1,idir2,ipert2]=((bec[ipert2,idir1,idir2]-zion[type_atom[ipert2]])*2*np.pi)+0.0j
                       print "  %d   %d   %d   %d   %E  %E"% \
                          (idir1+1,ipert1+1,idir2+1,ipert2+1, \
                          np.real(DDB[0,idir1,ipert1,idir2,ipert2]), np.imag(DDB[0,idir1,ipert1,idir2,ipert2]))
                elif (ipert2==natoms+1 and ipert1!=natoms and ipert1!=natoms+1): # mixed E/R pert <--> Z*
                   if (abs(bec[ipert1,idir1,idir2])<=1.0E-8):
                       DDB[0,idir1,ipert1,idir2,ipert2]=0.0+0.0j
                   else:
                       DDB[0,idir1,ipert1,idir2,ipert2]=((bec[ipert1,idir1,idir2]-zion[type_atom[ipert1]])*2*np.pi)+0.0j
                       print "  %d   %d   %d   %d   %E  %E"% \
                          (idir1+1,ipert1+1,idir2+1,ipert2+1, \
                          np.real(DDB[0,idir1,ipert1,idir2,ipert2]), np.imag(DDB[0,idir1,ipert1,idir2,ipert2]))
                if (ipert1==natoms+1 and ipert2==natoms+1): # pure E-field pertubations <--> epsilon^infty
# here I am suing rprimd with indices as in Eric script, I need to check this
                   DDB[0,idir1,ipert1,idir2,ipert2]=((1-eps_e[idir1,idir2])*np.pi*rprimvasp[idir1,idir2]*Ang2Bohr)+0j
                   print "  %d   %d   %d   %d   %E  %E"% \
                         (idir1+1,ipert1+1,idir2+1,ipert2+1, \
                         np.real(DDB[0,idir1,ipert1,idir2,ipert2]),np.imag(DDB[0,idir1,ipert1,idir2,ipert2]))

############################################################################
# Build DDB data into abinit-o_DDB file

def prtDDB():
    filename='./abinit-o_DDB'
    if os.path.isfile(filename) and os.access(filename, os.R_OK):
       with open(filename,"r") as f:
           data=f.readlines()
       f.close()
       ifound=0
       f=open(filename+".swp1","w")
       for x in data:
           if " 1st derivatives" in x:
               ifound=1
           elif ifound==1:
               ifound=ifound+1
           elif ifound==2:
               for i in range(natoms):
                   for j in range(3):
                       f.write("   %d   %d  %.14E  %.14E\n"%(j+1,i+1,forces[i,j],0.0))
               ifound=0
           else:
                   f.write(x)
       f.close()
       #os.system("mv "+filename+".swp1 "+filename)
       ddbfile=file(filename,"a")
# start from 0!
       for q in range(nqs):
 # Gamma point
          if (qpt[q,0]==0.0 and qpt[q,1]==0.0 and qpt[q,2]==0.0):
  # Number of element at q=Gamma
             numberel_Gamma = 3*(natoms+1)*3*(natoms+1)
             ddbfile.write("\n %s%d\n%s\n"%("2nd derivatives (non-stat.)  - # elements :     ",numberel_Gamma, \
                        " qpt  0.00000000E+00  0.00000000E+00  0.00000000E+00   1.0"))
             for ipert2 in range(natoms+2):
                for idir2 in range(3):
                   for ipert1 in range(natoms+2):
                      for idir1 in range(3):
                         if (ipert1==natoms or ipert2==natoms):
                            pass
                         elif (ipert1 == natoms+1 and ipert2 != natoms and ipert2 != natoms+1):  # mixed E/R pert <--> Z*
                            ddbfile.write("   %d   %d   %d   %d  %.14E  %.14E\n"%(idir1+1,ipert1+1,idir2+1,ipert2+1, \
                                         np.real(DDB[0,idir1,ipert1,idir2,ipert2]),np.imag(DDB[0,idir1,ipert1,idir2,ipert2])))
                         elif (ipert2 == natoms+1 and ipert1 != natoms and ipert1 != natoms+1):  # mixed E/R pert <--> Z*
                            ddbfile.write("   %d   %d   %d   %d  %.14E  %.14E\n"%(idir1+1,ipert1+1,idir2+1,ipert2+1, \
                                         np.real(DDB[0,idir1,ipert1,idir2,ipert2]),np.imag(DDB[0,idir1,ipert1,idir2,ipert2])))
                         elif (ipert1 == natoms+1 and ipert2 == natoms+1):  # pure E-field pertubations <--> epsilon^infty
                            ddbfile.write("   %d   %d   %d   %d  %.14E  %.14E\n"%(idir1+1,ipert1+1,idir2+1,ipert2+1, \
                                         np.real(DDB[0,idir1,ipert1,idir2,ipert2]),np.imag(DDB[0,idir1,ipert1,idir2,ipert2])))
                         else:
                            ddbfile.write("   %d   %d   %d   %d  %.14E  %.14E\n"%(idir1+1,ipert1+1,idir2+1,ipert2+1, \
                                           np.real(DDB[q,idir1,ipert1,idir2,ipert2]),np.imag(DDB[q,idir1,ipert1,idir2,ipert2])))
          else:   # if not Gamma
             numberel=3*natoms*3*natoms
             ddbfile.write("\n%s%d\n"%(" 2nd derivatives (non-stat.)  - # elements :     ",numberel))
             ddbfile.write(" qpt  %.8E  %.8E  %.8E  %.1f\n"%(qpt[q,0], qpt[q,1], qpt[q,2],1.0))
             for ipert2 in range(natoms):
                for idir2 in range(3):
                   for ipert1 in range(natoms):
                      for idir1 in range(3):
                         ddbfile.write("   %d   %d   %d   %d  %.14E  %.14E\n"%(idir1+1,ipert1+1,idir2+1,ipert2+1,\
                                    np.real(DDB[q,idir1,ipert1,idir2,ipert2]),np.imag(DDB[q,idir1,ipert1,idir2,ipert2])))
       ddbfile.close()

       nbblock=nqs+2;
       os.system("vim -c '%s/Number of data blocks=    2/Number of data blocks=    "+str(nbblock)+"/g' -c 'wq!' abinit-o_DDB")
    else:
       raise Exception('Either abinit-o_DDB file is missing or is not readable')

################################################################################################################################
################################################################################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='ConverttoDDB.py: a python script to convert Dynamical Matrices from Phonopy (VASP) to DDB (ABINIT) format.')
   parser.add_argument("--opt_n", help="options: 1, 2, 3 or 4",
                    type=int)
   args = parser.parse_args()
   print args.opt_n

###################################################################
############# Option n=4: build the DDB from system.dat, BORN and qpoints.yaml files
###################################################################
   if (args.opt_n==4):
      readSYSTEM()
      readBORN()
      #read_qpoints()
      #build_DDB_Efield()
      #prt_abinit_in()
      #runABINIT()
      #prtDDB()
###################################################################
############# Option n=2: build the system and BORN files from VASP OUTCAR
###################################################################
   if (args.opt_n==2):
# Need to have the following file in the path of the run:
# OUTCAR
      readOUTCAR()
      prtBORN()
      prtSYSTEM()

############################################################################
# option n = 1 build the system  file from VASP OUTCAR
############################################################################
   if (args.opt_n==1):
      readOUTCAR()
      prtSYSTEM()

############################################################################
# option n = 3 build abinit input and run abinit to get the DDB header
############################################################################
   if (args.opt_n==3):

      readOUTCAR()
      prtSYSTEM()
      prt_abinit_in()
      runABINIT()

