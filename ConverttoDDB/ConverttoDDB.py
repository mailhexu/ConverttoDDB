# Version 0.1
# July 20, 2017
# Developers:
#
# -Aldo Romero: alromero@mail.wvu.edu
# -Eric Bousquet
# -Matthieu Verstraete

# Script to build a DDB file for anaddb from Born effective charges (Z*),
# electronic dielectric constant (eps) and dynamical matrix (Dyn) of external code
# Z* and eps are/will be stored in the BORN file (phonopy format)
# Dyn have to be stored in DYNMAT file for each q-point
#  V0:  Eric Bousquet in Perl
#  B0.1:  A.H. Romero in Python and generalization to any crystal cell
#
# Option n = 1 build the DDB from the BORN and OUTCAR files
# Option n = 2 build the DDB and the BORN files from the OUTCAR file
# Option n = 3 build an input for abinit and run it providing the psp's files and the correct abinit.files
# Option n = 4 build the DDB from from system.dat file, BORN file and qpoints.yaml

# Known or unidentified bugs:
#  - rotation of rprim? Is is already fixed?
#  - eigenvectors might be wrong due to different conventions used in phonopy and anaddb
#  - Recover DM/BECs  with wrong symmetry if the magnetic structure symmetry is different from the FM/PM structural symmetry (not yet verified)

import numpy as np
import glob
import os
import os.path
import copy
import shutil
import subprocess
import yaml
import argparse
from ase.data import chemical_symbols, atomic_masses
from ase.io import read, write
import phonopy
#from optparse import OptionParser

Ang2Bohr = 1 / 0.5291772108
Bohr2Ang = 0.5291772108
ev2Ha = 1 / 27.2113845
Ha2eV = 27.2113845
e_Cb = 1.602176487e-19
HaBohr32GPa = Ha2eV / np.power(Bohr2Ang, 3) * e_Cb * 1.0e+21

#print HaBohr32GPa


def Mendel_data(i):
    return chemical_symbols[i]


class DDBconvertor():
    def __init__(self):
        self.unitcell = None
        self.zion = []
        self.znucl = []
        self.masspecie = []
        self.massatom = []
        self.nametypeat = []
        self.nbpertype = []
        self.typat = []
        self.acell = np.zeros((3))
        self.volume = 0.0
        self.stress = np.zeros((6))
        self.rprimvasp = np.zeros((3, 3))
        self.rprim = np.eye(3)
        self.rprimd = np.zeros((3, 3))

        self.eps_e = None
        self.bec = None

        self.dynmat = None
        self.phi = None
        self.nqs = None
        self.qpt = None
        self.DDB = None

        self.opt_n = 1

    def set_unitcell(self, atoms=None,rotate=False):
        """
        set the unitcell and prepare the abnit parameters.
        """
        if atoms is not None:
            self.unitcell = atoms

        self.natoms = len(self.unitcell)
        self.rprimvasp = self.unitcell.get_cell()

        self.acell = np.zeros(3)
        self.rprim = np.zeros([3, 3])

        for i in range(3):
            self.acell[i] = np.sqrt(
                np.dot(self.rprimvasp[i, :], self.rprimvasp[i, :])) #* Ang2Bohr


            self.rprim[:, i] = self.rprimvasp[i, :] / self.acell[i]

            #ROTATE the matrix
            #rprimd[:,i]=rprimvasp[i,:]
            #rprim[:,i]=rprimvasp[i,:]/acell[i]
            # DO NOT rotate the matrix
            #rprimd[:, i] = rprimvasp[:, i]
            #rprim[i, :] = rprimvasp[i, :] / acell[i]


        for i in range(3):
            self.rprimvasp[i, :] = self.rprim[:, i] * self.acell[i] #* Bohr2Ang

        self.acell *= Ang2Bohr


        # rprimd & volumn
        for i in range(3):
            self.rprimd[:, i] = self.rprim[:, i] * self.acell[i]

        if rotate:
            self.rprim=self.rprim.T
            self.rprimd=self.rprimd.T


        self.volume = np.dot(self.rprimd[:, 0],
                             np.cross(self.rprimd[:, 1], self.rprimd[:, 2]))


        # xred
        #self.xred = np.zeros((self.natoms, 3))
        self.xred = self.unitcell.get_scaled_positions()

        # ntypat
        self.ntypat = len(set(self.unitcell.get_chemical_symbols()))

        # znucl
        numbers = self.unitcell.get_atomic_numbers().copy()
        self.znucl = []
        for a, Z in enumerate(numbers):
            if Z not in self.znucl:
                self.znucl.append(Z)

        # nametypeat
        self.nametypeat = [chemical_symbols[i] for i in self.znucl]

        # typat
        self.typat = []
        for Z in self.unitcell.numbers:
            for n, Zs in enumerate(self.znucl):
                if Z == Zs:
                    self.typat.append(n + 1)

        #nbpertype
        for i in range(1, self.ntypat + 1):
            self.nbpertype.append(self.typat.count(i))

        #forces TODO check the unit of the forces
        self.forces = np.zeros([self.natoms, 3])

        self.stress = np.zeros([6])

        #zion
        self.zion = np.zeros(self.ntypat)

        # masspecie
        self.masspecie = [atomic_masses[i] for i in self.znucl]

        # massatom
        self.massatom = self.unitcell.get_masses()

    def set_forces(self, forces, unit='vasp'):
        if unit == 'vasp':
            self.forces = forces * (ev2Ha / Ang2Bohr)
        elif unit == 'abinit':
            self.forces = forces

    def set_stress(self, stress, unit='vasp'):
        if unit == 'vasp':
            self.stress = stress / 10.0 / HaBohr32GPa
        elif unit == 'abinit':
            self.stress = stress

    def set_born(self, bec):
        if len(bec) == 0:
            self.bec = np.zeros([self.natoms, 3, 3])
        else:
            self.bec = bec

    def set_eps(self, eps_e):
        if len(eps_e) == 0:
            self.eps_e = np.zeros([3, 3])
        else:
            self.eps_e = eps_e

    def read_BORN(self, fname='BORN'):
        self.readBORN(bfile=fname)

    ############################################################################
    # Print BORN file
    def prtBORN(self, filename="BORN"):
        eps_e = self.eps_e
        bec = self.bec
        natoms = self.natoms
        if (len(eps_e) != 0 and len(bec) != 0):
            #filename = "BORN"
            if (os.path.isfile(filename)):
                print("Found an existing BORN file, writing a new one")
                filename = filename + ".new"
            born = open(filename, "w")
            born.write("1.0\n")

        # print epsilon^infty
        for i in range(3):
            born.write("%.8f   %.8f   %.8f " %
                       (eps_e[i, 0], eps_e[i, 1], eps_e[i, 2]))
            born.write("\n")

        # print Z*
        for ion in range(natoms):
            for i in range(3):
                born.write("%.8f   %.8f   %.8f " %
                           (bec[ion, i, 0], bec[ion, i, 1], bec[ion, i, 2]))
                born.write("\n")
                born.close()
        else:
            raise Exception('BORN charges do not exist')

    def readSYSTEM(self, sfile='system.dat'):
        OK_xred = 0
        OK_rprim = 0
        if os.path.isfile(sfile) and os.access(sfile, os.R_OK):
            with open(sfile, 'r') as f:
                data = f.readlines()
            print("\n Read from system.dat file:")
            for line in data:
                if "natom" in line:
                    d = line.split()
                    self.natoms = int(d[1])
                    self.forces = np.zeros((self.natoms, 3))
                    print("natom = %d" % self.natoms)
                if "ntypat" in line:
                    d = line.split()
                    self.ntypat = int(d[1])
                    print("ntypat = %d" % self.ntypat)
                if ("acell" in line):
                    d = line.split()
                    if (len(d) > 4):
                        self.acell[:] = np.array(map(float, d[1:4])) * Ang2Bohr
                        print("entre")
                    else:
                        self.acell[:] = np.array(map(float, d[1:4]))
                    print("acell in a.u. = ", self.acell)
            if (self.ntypat == 0 or self.natoms == 0 or
                    np.prod(self.acell) == 0.0):
                raise Exception('error in system.dat ntypat or natom or acell')
            j = 0
            self.xred = np.zeros((self.natoms, 3))
            while True:
                line = data[j]
                if "zion" in line:
                    d = line.split()
                    if len(d) < self.ntypat + 1:
                        raise Exception(
                            'error in system.dat ntypat does not coincide with zion '
                        )
                    else:
                        for i in range(self.ntypat):
                            self.zion.append(float(d[i + 1]))
                        print("zion = ", self.zion)
                if "znucl" in line:
                    d = line.split()
                    if len(d) < self.ntypat + 1:
                        raise Exception(
                            'error in system.dat ntypat does not coincide with znucl '
                        )
                    else:
                        for i in range(self.ntypat):
                            self.znucl.append(float(d[i + 1]))
                            self.nametypeat.append(Mendel_data(int(d[i + 1])))
                        print("znucl = ", self.znucl)
                        print("nametypeat = ", self.nametypeat)
                if "mass" in line:
                    d = line.split()
                    if len(d) < self.ntypat + 1:
                        raise Exception(
                            'error in system.dat ntypat does not coincide with number of elements in mass '
                        )
                    else:
                        for i in range(self.ntypat):
                            self.masspecie.append(float(d[i + 1]))
                        print("mass per specie = ", self.masspecie)
                if ("atom" in line and "name" in line):
                    d = line.split()
                    if len(d) < self.ntypat + 1:
                        raise Exception(
                            'error in system.dat ntypat does not coincide with number of elements in atom name '
                        )
                    else:
                        for i in range(self.ntypat):
                            self.nametypeat.append(float(d[i + 1]))
                        print("atom nametypeats = ", type)
                if ("stress" in line):
                    d = line.split()
                    self.stress = np.array(map(float, d[1:]))
                if "typat" in line and "ntypat" not in line:
                    d = line.split()
                    s = d[1:]
                    if len(s) == self.ntypat:
                        for x in s:
                            if "*" in x:
                                splt = x.split("*")
                                for i in range(int(splt[0])):
                                    self.typat.append(int(splt[1]))
                            else:
                                self.typat.append(int(x))
                        for i in range(1, self.ntypat + 1):
                            self.nbpertype.append(self.typat.count(i))
                        print("typat=", self.typat)
                        print("nbpertype=", self.nbpertype)
                    else:
                        raise Exception(
                            'error in system.dat ntypat does not coincide with number of elements in typat '
                        )
                if ("rprimvasp" in line):
                    OK_rprim = 1
                    d = line.split()
                    if len(d) == 1:
                        for i in range(3):
                            j = j + 1
                            line = data[j]
                            d = map(float, line.split())
                            self.rprimvasp[i, :] = d
                    else:
                        self.rprimvasp[0, :] = map(float, d[1:])
                        for i in range(2):
                            j = j + 1
                            line = data[j]
                            self.rprimvasp[i + 1, :] = map(float, line.split())
                    if (np.prod(self.acell) != 1.0):
                        for i in range(3):
                            self.rprimvasp[i, :] = self.rprimvasp[
                                i, :] * self.acell[i] * Bohr2Ang
                    for i in range(3):
                        self.rprimd[:, i] = self.rprimvasp[i, :]
                    for i in range(3):
                        self.acell[i] = np.sqrt(
                            np.dot(self.rprimvasp[i, :], self.rprimvasp[
                                i, :])) * Ang2Bohr
                        self.rprim[:, i] = self.rprimvasp[i, :] / self.acell[i]
                    print("rprim = ", self.rprim)
                    print("rprimd = ", self.rprimd)
                    print("rprimvasp = ", self.rprimvasp)

                if ("rprim" in line and "vasp" not in line and
                        "rprimd" not in line):
                    OK_rprim = 1
                    d = line.split()
                    if len(d) == 1:
                        for i in range(3):
                            j = j + 1
                            line = data[j]
                            d = map(float, line.split())
                            self.rprim[i, :] = d
                    else:
                        self.rprim[0, :] = map(float, d[1:])
                        for i in range(2):
                            j = j + 1
                            line = data[j]
                            self.rprim[i + 1, :] = map(float, line.split())
                    for i in range(3):
                        self.rprimd[:, i] = self.rprim[:, i] * self.acell[i]
                    for i in range(3):
                        self.rprimvasp[i, :] = self.rprim[:, i] * self.acell[
                            i] * Bohr2Ang
                    print("rprim = ", self.rprim)
                    print("rprimd = ", self.rprimd)
                    print("rprimvasp = ", self.rprimvasp)
                if ("rprimd" in line):
                    raise Exception(
                        'reading rprimd from system.dat is not yet implemented')
                if "xred" in line:
                    d = line.split()
                    if len(d) == 1:
                        for i in range(self.natoms):
                            j = j + 1
                            line = data[j]
                            self.xred[i, :] = map(float, line.split())
                    else:
                        self.xred[0, :] = map(float, d[1:])
                        for i in range(1, self.natoms):
                            j = j + 1
                            line = data[j]
                            self.xred[i, :] = map(float, line.split())
                    print("xred=")
                    for i in range(self.natoms):
                        print(self.xred[i, :])
                if "forces" in line:
                    d = line.split()
                    if len(d) == 1:
                        for i in range(self.natoms):
                            j = j + 1
                            line = data[j]
                            self.forces[i, :] = map(float, line.split())
                    else:
                        self.forces[0, :] = map(float, d[1:])
                        for i in range(1, self.natoms):
                            j = j + 1
                            line = data[j]
                            self.forces[i, :] = map(float, line.split())
                    print("xred=")
                    for i in range(self.natoms):
                        print(self.forces[i, :])
                if (j + 1 < len(data)):
                    j = j + 1
                else:
                    break
        else:
            raise Exception(
                'error when openning system.dat file: it does not exist!')
        for i in range(self.natoms):
            self.massatom.append(self.masspecie[self.typat[i] - 1])
        if (OK_rprim == 0):
            for i in range(3):
                self.rprimvasp[i, :] = self.rprim[:, i] * self.acell[
                    i] * Bohr2Ang
            print("rprimvasp = ", self.rprimvasp)
        print("mass atom = ", self.massatom)
        print("***************** typat = ", self.typat)
        volume = np.dot(self.rprimd[:, 0],
                        np.cross(self.rprimd[:, 1], self.rprimd[:, 2]))
        print("volume =", volume)

    ############################################################################
    # Print SYSTEM file
    def prtSYSTEM(self, filename="system.dat"):
        natoms = self.natoms
        ntypat = self.ntypat
        acell = self.acell
        masspecie = self.masspecie
        rprim = self.rprim
        xred = self.xred
        zion = self.zion
        nametypeat = self.nametypeat
        nbpertype = self.nbpertype
        forces = self.forces
        stress = self.stress

        #TODO Not needed.
        if (os.path.isfile(filename)):
            print("File system.dat exist, moving the old one ")
            os.system("mv system.dat system.dat.old")
        system = open(filename, 'w')
        system.write("natoms %d \n" % natoms)
        system.write("ntypat %d \n" % ntypat)
        system.write("zion ")
        for i in range(ntypat):
            system.write(" %d " % zion[i])
        system.write("\n")
        system.write("znucl ")
        for i in range(ntypat):
            for j in range(112):
                if (Mendel_data(j) == nametypeat[i]):
                    system.write(" %d  " % j)
        system.write("\n")
        system.write("mass ")
        for i in range(ntypat):
            system.write(" %.6f " % masspecie[i])
        system.write("\n")
        system.write("typat ")
        for i in range(ntypat):
            system.write(" %d*%d " % (nbpertype[i], i + 1))
        system.write("\n")
        system.write("acell %.16f %.16f %.16f \n" %
                     (acell[0], acell[1], acell[2]))
        system.write("\n")
        system.write("rprim \n")
        for i in range(3):
            system.write("%.16f %.16f %.16f\n" %
                         (rprim[i, 0], rprim[i, 1], rprim[i, 2]))
        system.write("\n")
        system.write("xred \n")
        for i in range(natoms):
            system.write("%.16f %.16f %.16f\n" %
                         (xred[i, 0], xred[i, 1], xred[i, 2]))
        system.write("\n")
        system.write("forces \n")
        for i in range(natoms):
            system.write("%.16f %.16f %.16f\n" %
                         (forces[i, 0], forces[i, 1], forces[i, 2]))
        system.write("stress ")
        system.write("%.16f %.16f %.16f %.16f %.16f %.16f\n" % (
            stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]))
        system.close()

    def read_qpoints(self, fname='./qpoints.yaml'):
        #output variables
        #dynmat; # store the Dynamical Matrix for each q-point
        #phi; # store the Force constant matrix for each q-point = dynmat/srqt(m_i*m_j)
        #global dynmat
        #global phi
        #global nqs
        #global qpt
        #global DDB

        natoms = self.natoms
        rprimvasp = self.rprimvasp
        rprimd = self.rprimd
        massatom = self.massatom

        q = 0  # store the q-point number

        nbmode = 3 * natoms
        OK_dyn = 0

        if os.path.isfile(fname) and os.access(fname, os.R_OK):
            data = yaml.load(open(fname))
            nqs = data['nqpoint']
            if (self.natoms == 0):
                raise Exception(
                    'natoms is not defined while qpoints.yaml are going to be read. Meaning system is not defined'
                )
            print("rprimvasp = ", rprimvasp)
            qpt = []
            for i in range(nqs):
                qi = np.array(data['phonon'][i]['q-position'])
                qpt.append(qi)
                print("q point ", qi, " read")
            qpt = np.array(qpt)
            dynmat = np.zeros((nqs, 3, natoms, 3, natoms), dtype=np.complex_)
            phi = np.zeros((nqs, 3, natoms, 3, natoms), dtype=np.complex_)
            DDB = np.zeros(
                (nqs + 2, 3, natoms + 2, 3, natoms + 2), dtype=np.complex_)
            for iq in range(nqs):
                dynmat_data = data['phonon'][iq]['dynamical_matrix']
                if (len(dynmat_data) > 0):
                    OK_dyn = 1
                    dm = []
                    for row in dynmat_data:
                        vals = np.reshape(row, (-1, 2))
                        dm.append(vals[:, 0] + vals[:, 1] * 1j)
                    dm = np.array(dm)
                    print(dm.shape)
                    print(natoms)
                    for ipert1 in range(natoms):
                        for ipert2 in range(natoms):
                            dm1 = dm[ipert1 * 3:(ipert1 + 1) * 3, ipert2 * 3:(
                                ipert2 + 1) * 3]
                            print(ipert1, ipert2, ipert1 * 3, (ipert1 + 1) * 3,
                                  ipert2 * 3, (ipert2 + 1) * 3, dm1)
                            dm1 = np.matmul(rprimd,
                                            np.matmul(dm1, rprimd.transpose()))
                            for idir1 in range(3):
                                for idir2 in range(3):
                                    dynmat[iq, idir1, ipert1, idir2,
                                           ipert2] = dm1[idir1, idir2]
                                    phi[iq, idir1, ipert1, idir2,
                                        ipert2] = dm1[idir1, idir2] * np.sqrt(
                                            massatom[ipert1] *
                                            massatom[ipert2])
                                    DDB[iq, idir1, ipert1, idir2,
                                        ipert2] = phi[iq, idir1, ipert1, idir2,
                                                      ipert2] * ev2Ha / (
                                                          Ang2Bohr * Ang2Bohr)
                                    #DDB[iq,idir1,ipert1,idir2,ipert2]=phi[iq,idir1,ipert1,idir2,ipert2]*ev2Ha/(Ang2Bohr*Ang2Bohr)*acell[idir1]*acell[idir2]
                            i = i + 1
                else:
                    raise Exception(
                        'dynamical matrix does not exist in qpoints.yaml')
        else:
            raise Exception(
                'Either qpoints.yaml file is missing or is not readable')
        self.dynmat = dynmat
        self.phi = phi
        self.nqs = nqs
        self.qpt = qpt
        self.DDB = DDB

    ############################################################################
    # Build DDB from epsilon^infty and Z*
    # The E-field DDB's are stored at qpt=0
    def build_DDB_Efield(self):
        #rprim = self.rprim
        natoms = self.natoms
        ntypat = self.ntypat
        nbpertype = self.nbpertype
        rprimd = self.rprimd
        DDB = self.DDB
        bec = self.bec
        zion = self.zion
        acell = self.acell
        volume = self.volume

        print("stored volume = ", volume)
        # first get the type number of each atom:
        k = 0
        type_atom = [0] * natoms
        for i in range(ntypat):
            for j in range(nbpertype[i]):
                type_atom[k] = i
                k = k + 1
        print("\n----------------------------\n DDB for e-field:")

        gprimd = np.linalg.inv(rprimd)
        dij = np.identity(3)
        for ipert1 in range(natoms):  # ipert1 is atom position deriv
            ipert2 = natoms + 1  # E field deriv
            dm1 = np.matmul(
                rprimd,
                np.matmul(
                    bec[ipert1, :, :] - dij[:, :] * zion[type_atom[ipert1]],
                    gprimd))
            #dm1=np.matmul(rprimd,np.matmul(bec[ipert1,:,:]-dij[:,:]*zion[type_atom[ipert1]], rprim.transpose()))
            for idir1 in range(3):
                for idir2 in range(3):
                    DDB[0, idir1, ipert1, idir2, ipert2] = dm1[
                        idir1, idir2] * 2 * np.pi + 0.0j
                    DDB[0, idir2, ipert2, idir1, ipert1] = dm1[
                        idir1, idir2] * 2 * np.pi + 0.0j
                    print(" %3d %3d %3d %3d  %E  %E" %
                          (idir1 + 1, ipert1 + 1, idir2 + 1, ipert2 + 1,
                           np.real(DDB[0, idir1, ipert1, idir2, ipert2]),
                           np.imag(DDB[0, idir1, ipert1, idir2, ipert2])))
                    print(" %3d %3d %3d %3d  %E  %E" %
                          (idir2 + 1, ipert2 + 1, idir1 + 1, ipert1 + 1,
                           np.real(DDB[0, idir2, ipert2, idir1, ipert1]),
                           np.imag(DDB[0, idir2, ipert2, idir1, ipert1])))

    #                dij=1
    #                if (idir1 != idir2):
    #                   dij=0
    #                if (ipert1==natoms+1 and ipert2 != natoms and ipert2 != natoms+1):   # mixed E/R pert <--> Z*
    #                   if (abs(bec[ipert2,idir1,idir2])<=1.0E-8):
    #                       DDB[0,idir1,ipert1,idir2,ipert2]=0.0+0.0j
    #                   else:
    #                       DDB[0,idir1,ipert1,idir2,ipert2]=((bec[ipert2,idir1,idir2]-dij*zion[type_atom[ipert2]])*2*np.pi)+0.0j
    #                       print("  %d   %d   %d   %d   %E  %E"% \
    #                          (idir1+1,ipert1+1,idir2+1,ipert2+1, \
    #                          np.real(DDB[0,idir1,ipert1,idir2,ipert2]), np.imag(DDB[0,idir1,ipert1,idir2,ipert2])))_
    #                elif (ipert2==natoms+1 and ipert1!=natoms and ipert1!=natoms+1): # mixed E/R pert <--> Z*
    #                   if (abs(bec[ipert1,idir1,idir2])<=1.0E-8):
    #                       DDB[0,idir1,ipert1,idir2,ipert2]=0.0+0.0j
    #                   else:
    #                       DDB[0,idir1,ipert1,idir2,ipert2]=((bec[ipert1,idir1,idir2]-dij*zion[type_atom[ipert1]])*2*np.pi)+0.0j
    #                       print "  %d   %d   %d   %d   %E  %E"% \
    #                          (idir1+1,ipert1+1,idir2+1,ipert2+1, \
    #                          np.real(DDB[0,idir1,ipert1,idir2,ipert2]), np.imag(DDB[0,idir1,ipert1,idir2,ipert2]))

    #               pure E-field pertubations <--> epsilon^infty
    #if (ipert1==natoms+1 and ipert2==natoms+1):
        factor = volume / (acell[idir1] *
                           acell[idir2])  #FIXME factor?? not used here
        ipert1 = natoms + 1
        ipert2 = natoms + 1
        #dm1=np.matmul(rprimd,np.matmul(dij[:,:]-eps_e[:,:], rprimd.transpose()))
        dm1 = np.matmul(gprimd.transpose(),
                        np.matmul(dij[:, :] - self.eps_e[:, :], gprimd))
        for idir1 in range(3):
            for idir2 in range(3):
                DDB[0, idir1, ipert1, idir2, ipert2] = dm1[
                    idir1, idir2] * np.pi * volume + 0.0j

        #DDB[0,idir1,ipert1,idir2,ipert2]=((dij-eps_e[idir1,idir2])*np.pi*factor)+0j
        print(" %3d %3d %3d %3d  %E  %E" %
              (idir1 + 1, ipert1 + 1, idir2 + 1, ipert2 + 1,
               np.real(DDB[0, idir1, ipert1, idir2, ipert2]),
               np.imag(DDB[0, idir1, ipert1, idir2, ipert2])))

    def prtDDB(self, filename='./abinit-o_DDB'):
        natoms = self.natoms
        forces = self.forces
        stress = self.stress
        nqs = self.nqs
        qpt = self.qpt
        DDB = self.DDB

        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            with open(filename, "r") as f:
                data = f.readlines()
            f.close()
            ifound = 0
            f = open(filename + ".swp1", "w")
            for x in data:
                if " 1st derivatives" in x:
                    ifound = 1
                    f.write(x)
                elif ifound == 1:
                    for i in range(natoms):
                        for j in range(3):
                            f.write(" %3d %3d % .14E  %.14E\n" %
                                    (j + 1, i + 1, forces[i, j], 0.0))
                    for i in range(3):
                        f.write(" %3d %3d % .14E  %.14E\n" %
                                (i + 1, natoms + 3, stress[i], 0.0))
                    for i in range(3):
                        f.write(" %3d %3d % .14E  %.14E\n" %
                                (i + 1, natoms + 4, stress[i + 3], 0.0))
                    ifound = 1000
                elif (ifound == 0):
                    f.write(x)
            f.close()
            os.system("mv " + filename + ".swp1 " + filename)
            ddbfile = file(filename, "a")
            # start from 0!
            for q in range(nqs):
                # Gamma point
                if (qpt[q, 0] == 0.0 and qpt[q, 1] == 0.0 and
                        qpt[q, 2] == 0.0):
                    # Number of element at q=Gamma
                    numberel_Gamma = 3 * (natoms + 1) * 3 * (natoms + 1)
                    ddbfile.write("\n%s%8d\n%s\n" % (
                        " 2nd derivatives (non-stat.)  - # elements :",
                        numberel_Gamma,
                        " qpt  0.00000000E+00  0.00000000E+00  0.00000000E+00   1.0"
                    ))
                    for ipert2 in range(natoms + 2):
                        for idir2 in range(3):
                            for ipert1 in range(natoms + 2):
                                for idir1 in range(3):
                                    if (ipert1 == natoms or ipert2 == natoms):
                                        pass
                                    elif (ipert1 == natoms + 1 and
                                          ipert2 != natoms and ipert2 != natoms
                                          + 1):  # mixed E/R pert <--> Z*
                                        ddbfile.write(
                                            " %3d %3d %3d %3d % .14E  %.14E\n"
                                            % (idir1 + 1, ipert1 + 1,
                                               idir2 + 1, ipert2 + 1,
                                               np.real(DDB[0, idir1, ipert1,
                                                           idir2, ipert2]),
                                               np.imag(DDB[0, idir1, ipert1,
                                                           idir2, ipert2])))
                                    elif (ipert2 == natoms + 1 and
                                          ipert1 != natoms and ipert1 != natoms
                                          + 1):  # mixed E/R pert <--> Z*
                                        ddbfile.write(
                                            " %3d %3d %3d %3d % .14E  %.14E\n"
                                            % (idir1 + 1, ipert1 + 1,
                                               idir2 + 1, ipert2 + 1,
                                               np.real(DDB[0, idir1, ipert1,
                                                           idir2, ipert2]),
                                               np.imag(DDB[0, idir1, ipert1,
                                                           idir2, ipert2])))
                                    elif (ipert1 == natoms + 1 and
                                          ipert2 == natoms +
                                          1):  # pure E-field pertubations <--> epsilon^infty
                                        ddbfile.write(
                                            " %3d %3d %3d %3d % .14E  %.14E\n"
                                            % (idir1 + 1, ipert1 + 1,
                                               idir2 + 1, ipert2 + 1,
                                               np.real(DDB[0, idir1, ipert1,
                                                           idir2, ipert2]),
                                               np.imag(DDB[0, idir1, ipert1,
                                                           idir2, ipert2])))
                                    else:
                                        ddbfile.write(
                                            " %3d %3d %3d %3d % .14E  %.14E\n"
                                            % (idir1 + 1, ipert1 + 1,
                                               idir2 + 1, ipert2 + 1,
                                               np.real(DDB[q, idir1, ipert1,
                                                           idir2, ipert2]),
                                               np.imag(DDB[q, idir1, ipert1,
                                                           idir2, ipert2])))
                else:  # if not Gamma
                    numberel = 3 * natoms * 3 * natoms
                    ddbfile.write("\n%s%d\n" % (
                        " 2nd derivatives (non-stat.)  - # elements :     ",
                        numberel))
                    ddbfile.write(" qpt  %.8E  %.8E  %.8E  %.1f\n" %
                                  (qpt[q, 0], qpt[q, 1], qpt[q, 2], 1.0))
                    for ipert2 in range(natoms):
                        for idir2 in range(3):
                            for ipert1 in range(natoms):
                                for idir1 in range(3):
                                    ddbfile.write(
                                        " %3d %3d %3d %3d % .14E  %.14E\n" %
                                        (idir1 + 1, ipert1 + 1, idir2 + 1,
                                         ipert2 + 1, np.real(DDB[
                                             q, idir1, ipert1, idir2, ipert2]),
                                         np.imag(DDB[q, idir1, ipert1, idir2,
                                                     ipert2])))
            ddbfile.close()

            nbblock = nqs + 2
            os.system(
                "vim -c '%s/Number of data blocks=    2/Number of data blocks=    "
                + str(nbblock) + "/g' -c 'wq!' abinit-o_DDB")
        else:
            raise Exception(
                'Either abinit-o_DDB file is missing or is not readable')

    def readBORN(self, bfile='BORN'):
        natoms = self.natoms
        eps_e = np.zeros((3, 3))
        bec = np.zeros((natoms, 3, 3))
        if os.path.isfile(bfile) and os.access(bfile, os.R_OK):
            with open(bfile) as f:
                b = f.readlines()
            f.close()
            eps_e = np.array(map(float, b[1].split())).reshape(3, 3)
            for i in range(natoms):
                bec[i, :, :] = np.array(map(float, b[2 + i].split())).reshape(
                    3, 3)
            print("\n ------------------- ")
            print("Read from BORN file:")
            print("\nnatoms = ", natoms)
            print("\nepsilon electronic:")
            for i in range(3):
                print("%d %.10f %.10f %.10f" % (i + 1, eps_e[i, 0],
                                                eps_e[i, 1], eps_e[i, 2]))
            print("\nBorn effective charges:\nIon Dir1 BEC_[idir1,idir2]")
            for ion in range(natoms):
                for i in range(3):
                    print("%d %d %.10f %.10f %.10f" %
                          (ion + 1, i + 1, bec[ion, i, 0], bec[ion, i, 1],
                           bec[ion, i, 2]))
        else:
            #raise Exception('error when openning BORN file: it does not exist!')
            print("no BORN file present, assuming you know what you are doing")
            print("Assigning zeros to anything coming from BORN file")
        self.eps_e = eps_e
        self.bec = bec
        return eps_e, bec

    ############################################################################
    # Run Abinit

    def runABINIT(self):
        # Run abinit
        os.system("abinit <abinit.files> log_abinit")
        os.system("cp abinit-o_DDB abinit-o_DDB.swp1")
        os.system("rm abinit-o_OUT.nc")
        filename = "abinit-o_DDB"
        with open(filename) as f:
            b = f.readlines()
        f.close()
        filename = "abinit-o_DDB"
        f = open(filename, "w")
        for x in b:
            if "zion" in x:
                f.write('%s' % x[:10])
                for z in self.zion:
                    f.write("  %.14e" % z)
                f.write("\n")
            else:
                f.write("%s" % x)
        f.close()

    def read_structure(self, fname='OUTCAR',rotate=False):
        """
        read the cell structure from file. The file can be any of the types readable with ase.io.read, like vasp poscar, outcar, vasprun.xml, abinit input, output , cif and many many others. Full list see https://wiki.fysik.dtu.dk/ase/ase/io/io.html.
        TODO: add abinit DDB netcdf file support?
        """
        self.set_unitcell(read(fname),rotate=rotate)

    def read_born_from_outcar(self, fname='OUTCAR'):
        """
        read BECs from vasp OUTCAR file.
        TODO: implement unitcell != primitive cell case: This can be
        done by using symmetrized borns and then use phonopy to recover
        full BECs by symmetry of primitive cell.
        """
        born, epsilon = phonopy.interface.vasp._read_born_and_epsilon(
            filename=fname)
        if len(borns) == 0:
            print(
                "Warning: Born effective charge are not found and set to zero.\n"
            )
        if len(epsilon) == 0:
            print(
                "Warning: Born effective charge are not found and set to zero.\n"
            )
        self.set_bec(born)
        self.set_bec(epsilon)

    def prt_abinit_in(self):
        """
        """
        abin = open("abinit.in", "w")
        abin.write("natom %d\n" % self.natoms)
        abin.write("ntypat %d\n" % self.ntypat)
        abin.write("znucl ")
        for i in range(self.ntypat):
            for j in range(112):
                s = Mendel_data(j)
                if (s == self.nametypeat[i]):
                    abin.write(" %d" % j)
        abin.write("\n")
        abin.write("typat ")
        for i in range(self.ntypat):
            abin.write(" %d%s%d" % (self.nbpertype[i], '*', i + 1))
        abin.write("\n")
        abin.write("acell ")
        for i in range(3):
            abin.write(" %.8f" % self.acell[i])
        #abin.write(" Angstrom");
        abin.write("\n")
        abin.write("rprim ")
        for i in range(3):
            for j in range(3):
                abin.write(" %.8f" % self.rprim[i, j])
            abin.write("\n")
        abin.write("xred \n")
        for i in range(self.natoms):
            abin.write("%.8f  %.8f  %.8f\n" %
                       (self.xred[i, 0], self.xred[i, 1], self.xred[i, 2]))
        abin.write("\n")
        abin.write(
            "ecut   10\nkptopt 1\nngkpt  1 1 1\nixc    1\nnstep  1\ntoldfe 1.0E+05\niscf   5\nprtden 0\nprtwf  0\nprteig 0\n chkprim 0\n"
        )
        #    abin.write("ecut   10\nkptopt 1\nngkpt  1 1 1\nixc    1\nnstep  1\ntoldfe 1.0E+05\niscf   5\nprtden 0\nprtwf  0\nprteig 0\n")
        abin.close()

        abfiles = open("abinit.files", "w")
        abfiles.write("abinit.in\nabinit.out\nabinit-i\nabinit-o\nabinit-t\n")
        for i in range(self.ntypat):
            abfiles.write("%s.psp\n" % Mendel_data(int(self.znucl[i])))
        abfiles.close()

    def readOUTCAR(self, outfile='OUTCAR', opt_n=2):
        self.xred, self.atoms, self.ntypat, self.bec, self.eps_e, self.zion, self.nametypeat, self.nbpertype, self.znucl, self.forces, self.stress = readOUTCAR(
            outfile=outfile, opt_n=opt_n)


def readOUTCAR(outfile='OUTCAR', opt_n=2):
    """
    read infomation from OUTCAR.
    Parameters:
    ==================
    outfile: filename of vasp output file which has the forces, BECs, epsilons.
    Returns:
    ==================
    xred, natoms, ntypat, bec, eps_e, zion, nametypeat, nbpertype, znucl, forces
    """
    nbpertype = []
    zion = []
    xred = []
    lattrec = []
    eps_e = []
    natoms = 0
    znucl = []

    nametypeat = []
    masspecie = []
    rprimvasp = np.zeros([3, 3])
    acell = np.zeros(3)
    rprimd = np.zeros([3, 3])
    rprim = np.zeros([3, 3])
    massatom = []

    OK_nbpertype = 0
    OK_mass = 0
    OK_bec = 0
    OK_eps_e = 0
    OK_eps_i = 0
    OK_valenz = 0
    OK_xred = 0
    OK_latt = 0
    OK_force = 0

    if os.path.isfile(outfile) and os.access(outfile, os.R_OK):
        f = open(outfile, 'r')
        for line in f.readlines():
            if "vasp.5" in line:
                version = 5
                print("Vasp Version = ", line.split()[0])
            else:
                version = 4
            if "NIONS" in line:
                npos = line.find("NIONS")
                natoms = int(line[npos:].split()[2])
                bec = np.zeros((natoms, 3, 3))
                forces = np.zeros((natoms, 3))
                print("natoms = ", natoms)
                if natoms == 0:
                    raise Exception(
                        'error number of atoms, it is zero, going to stop')
            if "VRHFIN" in line:  # reading atom symbol
                s = line.split()[1]
                s = s.replace("=", "")
                s = s.replace(":", "")
                nametypeat.append(s)
            if (OK_force == 1 or OK_force == 2):
                OK_force = OK_force + 1
            if (OK_force > 2):
                d = map(float, line.split())
                forces[OK_force - 3, :] = np.array(d[3:])
                OK_force = OK_force + 1
                if (OK_force > (natoms + 2)):
                    OK_force = 0
            if ("in kB" in line):
                d = line.split()
                stress = np.array(map(float, d[2:]))
            if ("ions per type" in line and OK_nbpertype == 0):
                l = line.split()
                for i in range(4, len(l)):
                    nbpertype.append(int(l[i]))
                    OK_nbpertype = 1
            if (OK_mass == 1):
                l = line.split()
                for i in range(len(nbpertype)):
                    masspecie.append(float(l[i + 2]))
                OK_mass = 2
            if (OK_valenz == 1):
                l = line.split()
                for i in range(len(nbpertype)):
                    zion.append(float(l[i + 2]))
                OK_valenz = 2
            if (OK_xred <= natoms and OK_xred > 0):
                l = line.split()
                xred.append(map(float, l))
                OK_xred = OK_xred + 1
            if (OK_latt < 5 and OK_latt > 1):
                l = line.split()
                s = map(float, l)
                rprimvasp[OK_latt - 2, :] = s[0:3]
                lattrec.append(s[3:])
                OK_latt = OK_latt + 1
            if (OK_eps_e > 1 and OK_eps_e < 5):
                d = map(float, line.split())
                eps_e.append(d)
                OK_eps_e = OK_eps_e + 1
            if (OK_bec > 1 and OK_bec < 2 + 4 * natoms):
                index = (OK_bec - 2) % 4
                if (index > 0):
                    d = map(float, line.split())
                    bec[int((OK_bec - 2) / 4), index - 1, :] = d[1:]
                OK_bec = OK_bec + 1

#################################
######### Epsilon and Z* ########
##
##
            if ("Mass of Ions" in line and OK_mass == 0):
                OK_mass = 1
            if ("Ionic Valenz" in line and OK_valenz == 0):
                OK_valenz = 1
            if ("TOTAL-FORCE (eV/Angst)" in line):
                OK_force = 1
            if ("The static configuration has the point symmetry" in line):
                s = line.split()[7]
                s = s.replace(".", "")
                point_sym = s
            else:
                point_sym = 1
            if ("position of ions in fractional coordinates" in line):
                OK_xred = 1
            if "direct lattice vectors" in line:
                OK_latt = OK_latt + 1
            if ("MACROSCOPIC STATIC DIELECTRIC TENSOR" in line and
                    "including local field effects in DFT" in line):
                print("electronic epsilon found!")
                OK_eps_e = OK_eps_e + 1
            if (OK_eps_e == 1 and "-----------------------------" in line):
                OK_eps_e = OK_eps_e + 1
            if ("BORN EFFECTIVE CHARGES" in line and
                    "cummulative output" in line):
                OK_bec = OK_bec + 1
            if (OK_bec == 1 and "-----------------------------" in line):
                OK_bec = OK_bec + 1
        ntypat = len(nbpertype)
        print("number of species = ", ntypat)
        print("atom per specie =", nbpertype)
        print("nametypeat=", nametypeat)
        for i in range(ntypat):
            for j in range(112):
                if (Mendel_data(j) == nametypeat[i]):
                    znucl.append(j)
        print("znucl = ", znucl)
        print("Valence=", zion)
        print("masspeci=", masspecie)
        forces = forces * (ev2Ha / Ang2Bohr)
        stress = stress / 10.0 / HaBohr32GPa
        print("forces = ", forces)
        print("stress =", stress)
        j = 0
        for i in range(ntypat):
            for k in range(nbpertype[i]):
                massatom.append(masspecie[i])
                j = j + 1
        print("massatom = ", massatom)
        print("point symmetry = ", point_sym)
        for i in range(3):
            d = np.sqrt(np.dot(rprimvasp[i, :], rprimvasp[i, :]))
            acell[i] = d
            #ERIC THIS IS THE PLACE TO CHANGE ORIENTATION VECTORS
            #ROTATE the matrix
            #rprimd[:,i]=rprimvasp[i,:]
            #rprim[:,i]=rprimvasp[i,:]/acell[i]
            # DO NOT rotate the matrix
            rprimd[:, i] = rprimvasp[:, i]
            rprim[i, :] = rprimvasp[i, :] / acell[i]
            acell[i] = d * Ang2Bohr
        print("rprimvasp = ", rprimvasp)
        print("rprimd = ", rprimd)
        print("rprim (normalized) = ", rprim)
        print("acell in a.u. = ", acell)
        print("lattrec = ", lattrec)
        if (OK_eps_e > 0):
            eps_e = np.array(eps_e)
            print("eps_e =", eps_e)
        if (OK_bec > 0):
            print("bec =", bec)
        lattrec = np.array(lattrec)
        xred = np.array(xred)
        f.close()
        if (OK_bec == 0 and opt_n == 2):
            raise Exception(
                'No BEC found in OUTCAR!\n This is not possible with option n=2 \n The program is going to exit'
            )
        if (OK_eps_e == 0 and opt_n == 2):
            raise Exception(
                'No Electronic Dielectric Constant found in OUTCAR!\n This is not possible with option n=2 \n The program is going to exit'
            )
    else:
        raise Exception('error when openning OUTCAR file: it does not exist!')
    # global xred
    # global natoms
    # global ntypat
    # global bec
    # global eps_e
    # global zion
    # global nametypeat
    # global nbpertype
    # global znucl
    # global forces
    return (xred, natoms, ntypat, bec, eps_e, zion, nametypeat, nbpertype,
            znucl, forces, stress)


############################################################################

############################################################################
# Build DDB data into abinit-o_DDB file

################################################################################################################################
################################################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='ConverttoDDB.py: a python script to convert Dynamical Matrices from Phonopy (VASP) to DDB (ABINIT) format.'
    )
    parser.add_argument("--opt_n", help="options: 1, 2, 3 or 4", type=int)
    args = parser.parse_args()
    print(args.opt_n)

    ###################################################################
    ############# Option n=4: build the DDB from system.dat, BORN and qpoints.yaml files
    ###################################################################
    if (args.opt_n == 4):
        conv = DDBconvertor()
        conv.readSYSTEM()
        conv.readBORN()
        conv.read_qpoints()
        conv.build_DDB_Efield()
        conv.prt_abinit_in()
        conv.runABINIT()
        conv.prtDDB()
###################################################################
############# Option n=2: build the system and BORN files from VASP OUTCAR
###################################################################
    if (args.opt_n == 2):
        # Need to have the following file in the path of the run:
        # OUTCAR
        conv = DDBconvertor()
        conv.readOUTCAR()
        conv.prtBORN()
        conv.prtSYSTEM()

############################################################################
# option n = 1 build the system  file from VASP OUTCAR
############################################################################
    if (args.opt_n == 1):
        conv = DDBconvertor()
        conv.readOUTCAR()
        conv.prtSYSTEM()

############################################################################
# option n = 3 build abinit input and run abinit to get the DDB header
############################################################################
    if (args.opt_n == 5):
        conv = DDBconvertor()
        conv.read_structure()
        conv.read_qpoints()
        conv.read_BORN()
        conv.prtDDB()
