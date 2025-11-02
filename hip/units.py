from collections import OrderedDict

import numpy as np

# unit conversions based on NIST values

# HORM used before:
# hartree_to_ev = 27.2114
# bohr_to_angstrom = 0.529177
# ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

# https://physics.nist.gov/cgi-bin/cuu/Value?evhr
# 1 eV = 3.674 932 217 5665 x 10-2 Eh
ev_to_hartree = 3.6749322175665e-2
hartree_to_ev = 1 / ev_to_hartree

# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
# 1 bohr = 5.291 772 105 44 x 10-11 m
bohr_to_angstrom = 0.529177210544
angstrom_to_bohr = 1 / bohr_to_angstrom

ev_angstrom_2_to_hartree_bohr_2 = (bohr_to_angstrom**2) / hartree_to_ev

# from geometric.nifty
# Dictionary of atomic masses ; also serves as the list of elements (periodic table)
#
# Atomic mass data was updated on 2020-05-07 from NIST:
# "Atomic Weights and Isotopic Compositions with Relative Atomic Masses"
# https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
# using All Elements -> preformatted ASCII table.
#
# The standard atomic weight was provided in several different formats:
# Two numbers in brackets as in [1.00784,1.00811] : The average value of the two limits is used.
# With parentheses(uncert) as in 4.002602(2) : The parentheses was split off and all significant digits are used.
# A single number in brackets as in [98] : The single number was used
# Not provided (for Am, Z=95 and up): The mass number of the lightest isotope was used
PeriodicTable = OrderedDict(
    [
        ("H", 1.007975),
        ("He", 4.002602),  # First row
        ("Li", 6.9675),
        ("Be", 9.0121831),
        ("B", 10.8135),
        ("C", 12.0106),
        ("N", 14.006855),
        ("O", 15.99940),
        ("F", 18.99840316),
        ("Ne", 20.1797),  # Second row Li-Ne
        ("Na", 22.98976928),
        ("Mg", 24.3055),
        ("Al", 26.9815385),
        ("Si", 28.085),
        ("P", 30.973762),
        ("S", 32.0675),
        ("Cl", 35.4515),
        ("Ar", 39.948),  # Third row Na-Ar
        ("K", 39.0983),
        ("Ca", 40.078),
        ("Sc", 44.955908),
        ("Ti", 47.867),
        ("V", 50.9415),
        ("Cr", 51.9961),
        ("Mn", 54.938044),
        ("Fe", 55.845),
        ("Co", 58.933194),  # Fourth row K-Kr
        ("Ni", 58.6934),
        ("Cu", 63.546),
        ("Zn", 65.38),
        ("Ga", 69.723),
        ("Ge", 72.63),
        ("As", 74.921595),
        ("Se", 78.971),
        ("Br", 79.904),
        ("Kr", 83.798),
        ("Rb", 85.4678),
        ("Sr", 87.62),
        ("Y", 88.90584),
        ("Zr", 91.224),
        ("Nb", 92.90637),
        ("Mo", 95.95),
        ("Tc", 98.0),
        ("Ru", 101.07),
        ("Rh", 102.9055),  # Fifth row Rb-Xe
        ("Pd", 106.42),
        ("Ag", 107.8682),
        ("Cd", 112.414),
        ("In", 114.818),
        ("Sn", 118.71),
        ("Sb", 121.76),
        ("Te", 127.6),
        ("I", 126.90447),
        ("Xe", 131.293),
        ("Cs", 132.905452),
        ("Ba", 137.327),
        ("La", 138.90547),
        ("Ce", 140.116),
        ("Pr", 140.90766),
        ("Nd", 144.242),
        ("Pm", 145.0),
        ("Sm", 150.36),  # Sixth row Cs-Rn
        ("Eu", 151.964),
        ("Gd", 157.25),
        ("Tb", 158.92535),
        ("Dy", 162.5),
        ("Ho", 164.93033),
        ("Er", 167.259),
        ("Tm", 168.93422),
        ("Yb", 173.054),
        ("Lu", 174.9668),
        ("Hf", 178.49),
        ("Ta", 180.94788),
        ("W", 183.84),
        ("Re", 186.207),
        ("Os", 190.23),
        ("Ir", 192.217),
        ("Pt", 195.084),
        ("Au", 196.966569),
        ("Hg", 200.592),
        ("Tl", 204.3835),
        ("Pb", 207.2),
        ("Bi", 208.9804),
        ("Po", 209.0),
        ("At", 210.0),
        ("Rn", 222.0),
        ("Fr", 223.0),
        ("Ra", 226.0),
        ("Ac", 227.0),
        ("Th", 232.0377),
        ("Pa", 231.03588),
        ("U", 238.02891),
        ("Np", 237.0),
        ("Pu", 244.0),  # Seventh row Fr-Og
        ("Am", 241.0),
        ("Cm", 243.0),
        ("Bk", 247.0),
        ("Cf", 249.0),
        ("Es", 252.0),
        ("Fm", 257.0),
        ("Md", 258.0),
        ("No", 259.0),
        ("Lr", 262.0),
        ("Rf", 267.0),
        ("Db", 268.0),
        ("Sg", 271.0),
        ("Bh", 272.0),
        ("Hs", 270.0),
        ("Mt", 276.0),
        ("Ds", 281.0),
        ("Rg", 280.0),
        ("Cn", 285.0),
        ("Nh", 284.0),
        ("Fl", 289.0),
        ("Mc", 288.0),
        ("Lv", 293.0),
        ("Ts", 292.0),
        ("Og", 294.0),
    ]
)

# On 2020-05-07, these values were revised to CODATA 2018 values
# hartree-joule relationship   4.359 744 722 2071(85) e-18
# Hartree energy in eV         27.211 386 245 988(53)
# Avogadro constant            6.022 140 76 e23         (exact)
# molar gas constant           8.314 462 618            (exact)
# Boltzmann constant           1.380649e-23             (exact)
# Bohr radius                  5.291 772 109 03(80) e-11
# speed of light in vacuum     299 792 458 (exact)
# reduced Planck's constant    1.054571817e-34 (exact)
# calorie-joule relationship   4.184 J (exact; from NIST)

## Boltzmann constant in kJ mol^-1 k^-1
kb = 0.008314462618  # Previous value: 0.0083144100163
kb_si = 1.380649e-23

# Conversion factors
bohr2ang = 0.529177210903  # Previous value: 0.529177210
ang2bohr = 1.0 / bohr2ang
au2kcal = 627.5094740630558  # Previous value: 627.5096080306
kcal2au = 1.0 / au2kcal
au2kj = 2625.4996394798254  # Previous value: 2625.5002
kj2au = 1.0 / au2kj
grad_au2gmx = 49614.75258920567  # Previous value: 49614.75960959161
grad_gmx2au = 1.0 / grad_au2gmx
au2ev = 27.211386245988
ev2au = 1.0 / au2ev
au2evang = 51.422067476325886  # Previous value: 51.42209166566339
evang2au = 1.0 / au2evang
c_lightspeed = 299792458.0
hbar = 1.054571817e-34
avogadro = 6.02214076e23
au_mass = 9.1093837015e-31  # Atomic unit of mass in kg
amu_mass = 1.66053906660e-27  # Atomic mass unit in kg
amu2au = amu_mass / au_mass
cm2au = (
    100 * c_lightspeed * (2 * np.pi * hbar) * avogadro / 1000 / au2kj
)  # Multiply to convert cm^-1 to Hartree
ambervel2au = 9.349961132249932e-04  # Multiply to go from AMBER velocity unit Ang/(1/20.455 ps) to bohr/atu.


ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

ATOMIC_NUMBER_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_ATOMIC_NUMBER.items()}
