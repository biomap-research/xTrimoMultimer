# Copyright 2022 BioMap Technologies Limited
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that residue_constants generates correct values."""

import numpy as np
import pytest
from xtrimomultimer.np import residue_constants
from tests.utils import assert_len


@pytest.mark.parametrize(
    "residue_name, chi_num",
    [
        ("ALA", 0),
        ("CYS", 1),
        ("HIS", 2),
        ("MET", 3),
        ("LYS", 4),
        ("ARG", 4),
    ],
)
def test_chi_angles_atoms(residue_name, chi_num):
    chi_angles_atoms = residue_constants.chi_angles_atoms[residue_name]
    assert_len(chi_angles_atoms, chi_num)
    for chi_angle_atoms in chi_angles_atoms:
        assert_len(chi_angle_atoms, 4)


def test_chi_groups_for_atom():
    for k, chi_groups in residue_constants.chi_groups_for_atom.items():
        res_name, atom_name = k
        for chi_group_i, atom_i in chi_groups:
            assert (
                atom_name
                == residue_constants.chi_angles_atoms[res_name][chi_group_i][atom_i]
            )


@pytest.mark.parametrize(
    "atom_name, num_residue_atoms",
    [
        ("ALA", 5),
        ("ARG", 11),
        ("ASN", 8),
        ("ASP", 8),
        ("CYS", 6),
        ("GLN", 9),
        ("GLU", 9),
        ("GLY", 4),
        ("HIS", 10),
        ("ILE", 8),
        ("LEU", 8),
        ("LYS", 9),
        ("MET", 8),
        ("PHE", 11),
        ("PRO", 7),
        ("SER", 6),
        ("THR", 7),
        ("TRP", 14),
        ("TYR", 12),
        ("VAL", 7),
    ],
)
def test_residue_atoms(atom_name, num_residue_atoms):
    residue_atoms = residue_constants.residue_atoms[atom_name]
    assert_len(residue_atoms, num_residue_atoms)


def test_standard_atom_mask_shape():
    assert residue_constants.STANDARD_ATOM_MASK.shape == (
        21,
        37,
    )


def test_standard_atom_mask_values():
    str_to_row = lambda s: [c == "1" for c in s]  # More clear/concise.
    np.testing.assert_array_equal(
        residue_constants.STANDARD_ATOM_MASK,
        np.array(
            [
                # NB This was defined by c+p but looks sane.
                str_to_row("11111                                "),  # ALA
                str_to_row("111111     1           1     11 1    "),  # ARG
                str_to_row("111111         11                    "),  # ASP
                str_to_row("111111          11                   "),  # ASN
                str_to_row("11111     1                          "),  # CYS
                str_to_row("111111     1             11          "),  # GLU
                str_to_row("111111     1              11         "),  # GLN
                str_to_row("111 1                                "),  # GLY
                str_to_row("111111       11     1    1           "),  # HIS
                str_to_row("11111 11    1                        "),  # ILE
                str_to_row("111111      11                       "),  # LEU
                str_to_row("111111     1       1               1 "),  # LYS
                str_to_row("111111            11                 "),  # MET
                str_to_row("111111      11      11          1    "),  # PHE
                str_to_row("111111     1                         "),  # PRO
                str_to_row("11111   1                            "),  # SER
                str_to_row("11111  1 1                           "),  # THR
                str_to_row("111111      11       11 1   1    11  "),  # TRP
                str_to_row("111111      11      11         11    "),  # TYR
                str_to_row("11111 11                             "),  # VAL
                str_to_row("                                     "),  # UNK
            ]
        ),
    )


def test_standard_atom_mask_row_totals():
    # Check each row has the right number of atoms.
    for row, restype in enumerate(residue_constants.restypes):  # A, R, ...
        long_restype = residue_constants.restype_1to3[restype]  # ALA, ARG, ...
        atoms_names = residue_constants.residue_atoms[
            long_restype
        ]  # ['C', 'CA', 'CB', 'N', 'O'], ...
        assert_len(
            atoms_names,
            residue_constants.STANDARD_ATOM_MASK[row, :].sum(),
            long_restype,
        )


def test_atom_types():
    assert residue_constants.atom_type_num == 37

    assert residue_constants.atom_types[0] == "N"
    assert residue_constants.atom_types[1] == "CA"
    assert residue_constants.atom_types[2] == "C"
    assert residue_constants.atom_types[3] == "CB"
    assert residue_constants.atom_types[4] == "O"

    assert residue_constants.atom_order["N"] == 0
    assert residue_constants.atom_order["CA"] == 1
    assert residue_constants.atom_order["C"] == 2
    assert residue_constants.atom_order["CB"] == 3
    assert residue_constants.atom_order["O"] == 4
    assert residue_constants.atom_type_num == 37


def testRestypes():
    three_letter_restypes = [
        residue_constants.restype_1to3[r] for r in residue_constants.restypes
    ]
    for restype, exp_restype in zip(
        three_letter_restypes, sorted(residue_constants.restype_1to3.values())
    ):
        assert restype == exp_restype
    assert residue_constants.restype_num == 20


def testSequenceToOneHotHHBlits():
    one_hot = residue_constants.sequence_to_onehot(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ-", residue_constants.HHBLITS_AA_TO_ID
    )
    exp_one_hot = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(one_hot, exp_one_hot)


def test_sequence_to_one_hot_standard():
    one_hot = residue_constants.sequence_to_onehot(
        "ARNDCQEGHILKMFPSTWYV", residue_constants.restype_order
    )
    np.testing.assert_array_equal(one_hot, np.eye(20))


def test_sequence_to_one_hot_unknown_mapping():
    seq = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expected_out = np.zeros([26, 21])
    for row, position in enumerate(
        [
            0,
            20,
            4,
            3,
            6,
            13,
            7,
            8,
            9,
            20,
            11,
            10,
            12,
            2,
            20,
            14,
            5,
            1,
            15,
            16,
            20,
            19,
            17,
            20,
            18,
            20,
        ]
    ):
        expected_out[row, position] = 1
    aa_types = residue_constants.sequence_to_onehot(
        sequence=seq,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    assert (aa_types == expected_out).all()


@pytest.mark.parametrize(
    "seq",
    [
        ("lowercase", "aaa"),  # Insertions in A3M.
        ("gaps", "---"),  # Gaps in A3M.
        ("dots", "..."),  # Gaps in A3M.
        ("metadata", ">TEST"),  # FASTA metadata line.
    ],
)
def testSequenceToOneHotUnknownMappingError(seq):
    with pytest.raises(ValueError):
        residue_constants.sequence_to_onehot(
            sequence=seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
