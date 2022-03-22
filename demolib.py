from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import PyMol

from PyQuante.Molecule import Molecule
from PyQuante import SCF

# more imports
from scipy.integrate import simps
import numpy as np  # Numpy import
import pandas as pd

import matplotlib.pyplot as plt

from PyQuante.Constants import ang2bohr
import imolecule

import random
import io
import base64
from IPython.display import HTML


def visualize_Mol(molecule):
    mol = molecule.copy()
    # convert angstrom to bohr
    for atom in mol:
        coords = [a / ang2bohr for a in atom.pos()]
        atom.update_coords(coords)
    # create as xyz string
    xyz_str = mol.as_string()
    return imolecule.draw(xyz_str, format='xyz', shader="phong")


def linkFragments1(reactants):

    linking_reaction = AllChem.ReactionFromSmarts(
        '[cH1,nH1:1].[cH1:2]>>[*:1]-[*:2]')
    fusion_reaction = AllChem.ReactionFromSmarts(
        '[a:7]~1~[a:8]~[a:9]~[cH:1][cH:2]~[a:10]~1.[a:5][c;H1:3][c;H1:4][a:6]>>[*:7]~1~[*:8]~[*:9]~[*:1](:[*:5])~[*:2](:[*:6])~[*:10]~1.[*:3][*:4]')

    #############

    products = []

    # Linking reactions:
    for r1 in reactants:
        for r2 in reactants:
            for prod in linking_reaction.RunReactants((r1, r2)):
                product_SMILES = AllChem.MolToSmiles(prod[0])
                products.append(product_SMILES)

    # Fusion reactions:
    for r1 in reactants:
        for r2 in reactants:
            for prod in fusion_reaction.RunReactants((r1, r2)):
                product_SMILES = AllChem.MolToSmiles(prod[0])
                products.append(product_SMILES)

    ###############
    # get ony unique values
    products = list(set(products))

    return products


def linkFragments2(reactants, gen1):

    linking_reaction = AllChem.ReactionFromSmarts(
        '[cH1,nH1:1].[cH1:2]>>[*:1]-[*:2]')
    fusion_reaction = AllChem.ReactionFromSmarts(
        '[a:7]~1~[a:8]~[a:9]~[cH:1][cH:2]~[a:10]~1.[a:5][c;H1:3][c;H1:4][a:6]>>[*:7]~1~[*:8]~[*:9]~[*:1](:[*:5])~[*:2](:[*:6])~[*:10]~1.[*:3][*:4]')

    # First for loop: linking reactions
    products = []

    for r1 in reactants:
        for r3 in gen1:
            for prod in linking_reaction.RunReactants((r1, r3)):
                product_SMILES = AllChem.MolToSmiles(prod[0])
                products.append(product_SMILES)

    # Second nested for-loops: fusion reactions.
    for r1 in reactants:
        for r3 in gen1:
            for prod in fusion_reaction.RunReactants((r1, r3)):
                product_SMILES = AllChem.MolToSmiles(prod[0])
                products.append(product_SMILES)

    products = list(set(products))

    return products


def randomMoleculeSubset(n, products):

    random_ind = random.sample(range(len(products)), 200)
    mList = []
    products = list(products)
    for ind in random_ind:
        m = Chem.MolFromSmiles(products[ind], sanitize=False)
        m.UpdatePropertyCache()
        mList.append(m)
    return mList, random_ind


def embedVideo(afile):
    video = io.open(afile, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))


def ScharberEfficiency(molecule, power, wavelength, flux):
    mol_mindo3 = mindo3_calc(molecule)
    donorHOMO, donorLUMO = getHOMO_LUMO(mol_mindo3)
    gap = donorLUMO - donorHOMO

    fullPower = simps(power, wavelength)
    # This is the LUMO energy, in eV, of the acceptor molecule, PCBM.
    pcbm_l = -4.3

    # This is the open circuit voltage
    voc = abs(donorHOMO) - abs(pcbm_l) - 0.3

    # This is the bandgap, in nano-meters, band gap
    bandgap_ev = abs(donorLUMO - donorHOMO)
    bandgap = 1239.84187 / bandgap_ev

    wavelength_range = []
    power_range = []
    flux_range = []

    # determine what we will integrate --> only those wavelengths of the solar
    # spectrum shorter than the band gap wavelength
    for i in range(len(wavelength)):
        # we'll integrate all wavelengths that are shorter than the band gap.
        if wavelength[i] < bandgap:
            wavelength_range.append(wavelength[i])
            flux_range.append(flux[i])
        else:
            break

    # This is the short circuit current
    jsc = simps(flux_range, wavelength_range)
    # This is called the Fill Factor
    jsc = jsc * 0.65  # Take away 0.65 for 26%

    # Finally, this is the efficiency
    pce = 100 * voc * jsc / fullPower  # Take away 0.65 for 17%

    return pce


def randomMolecule(products):

    random_ind = random.sample(range(len(products)), 1)[0]

    products = list(products)

    # Convert SMILES to rdkit molecule
    m = Chem.MolFromSmiles(products[random_ind])
    # Add all the hydrogen atoms:
    m2 = Chem.AddHs(m)
    # Get the coordinates of the atoms in 3D space.
    AllChem.EmbedMolecule(m2)
    AllChem.UFFOptimizeMolecule(m2)
    confo1 = m2.GetConformer()
    # and empty list, where we'll add all the coordinates and atom types
    coordList = []
    for i, atom in enumerate(m2.GetAtoms()):
        coord = confo1.GetAtomPosition(i)
        coordList.append((atom.GetAtomicNum(), (coord.x, coord.y, coord.z)))
    molTest = Molecule('CEPTest',
                       atomlist=coordList, units='Angstroms')
    return molTest,m,random_ind


def getHOMO_LUMO(molecule_mindo3):
    #########################################################
    nclosed = molecule_mindo3.nclosed
    donor_HOMOh = molecule_mindo3.orbe[nclosed - 1]
    donor_LUMOh = molecule_mindo3.orbe[nclosed] - 5

    gap = donor_LUMOh - donor_HOMOh
    #######################################################
    return (donor_HOMOh, donor_LUMOh)


def mindo3_calc(molecule):
    ##################################################
    mol_mindo3 = SCF(molecule, method="MINDO3")
    mol_mindo3.iterate(maxiter=200)
    ##################################################
    return mol_mindo3


def rdkit_to_PyQuante(mol_smiles):
    ###########################################################
    # Convert SMILES to rdkit molecule
    m = Chem.MolFromSmiles(mol_smiles)

    # Get conformer from rdkit
    m2 = Chem.AddHs(m)
    AllChem.EmbedMolecule(m2)
    AllChem.UFFOptimizeMolecule(m2)
    confo1 = m2.GetConformer()

    # Get coordinates
    coordList = []
    for i, atom in enumerate(m2.GetAtoms()):
        coord = confo1.GetAtomPosition(i)
        coordList.append((atom.GetAtomicNum(), (coord.x, coord.y, coord.z)))

    # Create the PyQuante Molecule
    molTest = Molecule('CEPTest',
                       atomlist=coordList, units='Angstroms')

    ###############################################################
    return molTest


def loadSolarSpectrum():
    df = pd.io.parsers.read_csv('files/AM15_PhotFlux_rev.csv')
    wavelength = df['Wavelength (nm)']
    power = df[' Global Tilt (W m-2 nm-1)']
    flux = df[' Flux (W m-2)']
    plt.plot(wavelength, power)
    plt.xlabel('wavelength')
    plt.ylabel('power')
    plt.title('The Beautiful Solar Spectrum')
    plt.show()
    return power, wavelength, flux
