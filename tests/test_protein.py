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

"""Tests for protein."""

import os
import pathlib

from xtrimomultimer.np import protein
from xtrimomultimer.np import residue_constants
import numpy as np
import pytest
from tests.utils import assert_len

# Internal import (7716).
PROJ_DIR = pathlib.Path(__file__).parent.parent.absolute()
TEST_DATA_DIR = "tests/test_data/"
TEST_PDB_FILE = "2rbg.pdb"


def _check_shapes(prot, num_res):
    """Check that the processed shapes are correct."""
    num_atoms = residue_constants.atom_type_num
    assert (num_res, num_atoms, 3) == prot.atom_positions.shape
    assert (num_res,) == prot.aatype.shape
    assert (num_res, num_atoms) == prot.atom_mask.shape
    assert (num_res,) == prot.residue_index.shape
    assert (num_res,) == prot.chain_index.shape
    assert (num_res, num_atoms) == prot.b_factors.shape


@pytest.mark.parametrize(
    "pdb_file, chain_id, num_res, num_chains",
    [
        pytest.param(TEST_PDB_FILE, "A", 282, 1, id="chain_A"),
        pytest.param(TEST_PDB_FILE, "B", 282, 1, id="chain_B"),
        pytest.param(TEST_PDB_FILE, None, 564, 2, id="multichain"),
    ],
)
def test_from_pdb_str(pdb_file, chain_id, num_res, num_chains):
    pdb_file = os.path.join(PROJ_DIR, TEST_DATA_DIR, pdb_file)
    with open(pdb_file) as f:
        pdb_string = f.read()
    prot = protein.from_pdb_string(pdb_string, chain_id)
    _check_shapes(prot, num_res)
    assert prot.aatype.min() >= 0
    # Allow equal since unknown restypes have index equal to restype_num.
    assert prot.aatype.max() <= residue_constants.restype_num
    assert_len(np.unique(prot.chain_index), num_chains)


def test_to_pdb():
    with open(os.path.join(PROJ_DIR, TEST_DATA_DIR, TEST_PDB_FILE)) as f:
        pdb_string = f.read()
    prot = protein.from_pdb_string(pdb_string)
    pdb_string_reconstr = protein.to_pdb(prot)

    for line in pdb_string_reconstr.splitlines():
        assert_len(line, 80)

    prot_reconstr = protein.from_pdb_string(pdb_string_reconstr)

    np.testing.assert_array_equal(prot_reconstr.aatype, prot.aatype)
    np.testing.assert_array_almost_equal(
        prot_reconstr.atom_positions, prot.atom_positions
    )
    np.testing.assert_array_almost_equal(prot_reconstr.atom_mask, prot.atom_mask)
    np.testing.assert_array_equal(prot_reconstr.residue_index, prot.residue_index)
    np.testing.assert_array_equal(prot_reconstr.chain_index, prot.chain_index)
    np.testing.assert_array_almost_equal(prot_reconstr.b_factors, prot.b_factors)


def test_ideal_atom_mask():
    with open(os.path.join(PROJ_DIR, TEST_DATA_DIR, TEST_PDB_FILE)) as f:
        pdb_string = f.read()
    prot = protein.from_pdb_string(pdb_string)
    ideal_mask = protein.ideal_atom_mask(prot)
    non_ideal_residues = set([102] + list(range(127, 286)))
    for i, (res, atom_mask) in enumerate(zip(prot.residue_index, prot.atom_mask)):
        if res in non_ideal_residues:
            assert not np.all(atom_mask == ideal_mask[i]), f"{res}"
        else:
            assert np.all(atom_mask == ideal_mask[i]), f"{res}"


def test_too_many_chains():
    num_res = protein.PDB_MAX_CHAINS + 1
    num_atom_type = residue_constants.atom_type_num
    with pytest.raises(ValueError):
        _ = protein.Protein(
            atom_positions=np.random.random([num_res, num_atom_type, 3]),
            aatype=np.random.randint(0, 21, [num_res]),
            atom_mask=np.random.randint(0, 2, [num_res]).astype(np.float32),
            residue_index=np.arange(1, num_res + 1),
            chain_index=np.arange(num_res),
            b_factors=np.random.uniform(1, 100, [num_res]),
        )
