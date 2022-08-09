# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
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

from typing import Tuple
import pytest
import torch

from xtrimomultimer.utils.geometry import Vec3Array
from xtrimomultimer.utils.geometry.test_utils import (
    assert_vectors_close,
    assert_vectors_equal,
)


@pytest.mark.parametrize(
    "shape", [tuple([1]), (1, 1), (2, 1), (3, 5), (1, 1, 1), (2, 5, 6)]
)
def test_vec3array_shape(shape):
    vec = Vec3Array.zeros(shape)
    assert vec.shape == torch.Size(shape)


@pytest.mark.parametrize("shape", [(1), (1, 1), (2, 1), (3, 5), (1, 1, 1), (2, 5, 6)])
def test_vec3array_zeros(shape):
    zero_vec = Vec3Array.zeros(shape)
    zero_tensor = torch.zeros(shape, dtype=torch.float32)
    assert torch.equal(zero_vec.x, zero_tensor)
    assert torch.equal(zero_vec.y, zero_tensor)
    assert torch.equal(zero_vec.z, zero_tensor)


@pytest.mark.parametrize(
    "vec",
    [
        Vec3Array.zeros((1)),
        Vec3Array.zeros((1, 1)),
        Vec3Array.zeros((2, 2)),
        Vec3Array.zeros((3, 3, 3)),
    ],
)
def test_vec3array_clone(vec: Vec3Array):
    clone_vec = vec.clone()
    assert id(vec) != id(clone_vec)
    assert_vectors_equal(vec, clone_vec)


@pytest.mark.parametrize(
    "val_a,val_b,shape",
    [
        (0, 0, (1)),
        (1, 1, (1, 1)),
        (2, 3, (2, 2)),
        (3, 1.5, (3, 3, 3)),
        (0, 0.1, (1, 1, 1)),
    ],
)
def test_vec3array_add(val_a: float, val_b: float, shape):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_b = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
    )
    vector_out = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * (val_a + val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a + val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a + val_b),
    )
    vector_c = vector_a + vector_b
    assert_vectors_equal(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,val_b,shape",
    [
        (0, 0, (1)),
        (1, 1, (1, 1)),
        (2, 3, (2, 2)),
        (3, 1.5, (3, 3, 3)),
        (0, 0.1, (1, 1, 1)),
    ],
)
def test_vec3array_sub(val_a: float, val_b: float, shape):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_b = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
    )
    vector_out = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * (val_a - val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a - val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a - val_b),
    )
    vector_c = vector_a - vector_b
    assert_vectors_equal(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,shape",
    [
        (0, (1)),
        (1, (1, 1)),
        (2, (2, 2)),
        (3, (3, 3, 3)),
        (0, (1, 1, 1)),
    ],
)
def test_vec3array_neg(val_a: float, shape):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_b = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a * -1,
        torch.ones(shape, dtype=torch.float32) * val_a * -1,
        torch.ones(shape, dtype=torch.float32) * val_a * -1,
    )
    vector_c = -vector_a
    assert_vectors_equal(vector_c, vector_b)


@pytest.mark.parametrize(
    "val_a,shape",
    [
        (0, (1)),
        (1, (1, 1)),
        (2, (2, 2)),
        (3, (3, 3, 3)),
        (0, (1, 1, 1)),
    ],
)
def test_vec3array_pos(val_a: float, shape):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_c = +vector_a
    assert_vectors_equal(vector_c, vector_a)


@pytest.mark.parametrize(
    "val_a,val_b,shape",
    [
        (0, 0, (1)),
        (1.2, 2.1, (1, 1)),
        (2.0 / 3, 3.0 / 5, (2, 2)),
        (1.0 / 3, 1.5, (3, 3, 3)),
        (0, 0.1, (1, 1, 1)),
    ],
)
def test_vec3array_mul(val_a: float, val_b: float, shape: Tuple):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_b = torch.ones(1, dtype=torch.float32) * val_b
    vector_out = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
    )
    vector_c = vector_a * vector_b
    assert_vectors_close(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,val_b,shape",
    [
        (0, 0, (1)),
        (1.2, 2.1, (1, 1)),
        (2.0 / 3, 3.0 / 5, (2, 2)),
        (1.0 / 3, 1.5, (3, 3, 3)),
        (0, 0.1, (1, 1, 1)),
    ],
)
def test_vec3array_mul_val(val_a: float, val_b: float, shape):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )

    vector_out = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
        torch.ones(shape, dtype=torch.float32) * (val_a * val_b),
    )
    vector_c = vector_a * val_b
    assert_vectors_close(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,val_b,shape,val_out",
    [
        (0, 0, (1), 0.0),
        (1.2, 2.1, (2), 7.56),
        (2.0 / 3, 3.0 / 5, (3), 1.2),
        (0, 0.1, (5), 0.0),
    ],
)
def test_vec3array_dot(val_a: float, val_b: float, shape: Tuple, val_out: float):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_b = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
        torch.ones(shape, dtype=torch.float32) * val_b,
    )
    vector_c = vector_a.dot(vector_b)
    vector_out = torch.ones(shape, dtype=torch.float32) * val_out
    assert torch.allclose(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,shape",
    [
        (1, (1)),
        (2.0, (1)),
        (3.0, (1)),
    ],
)
def test_vec3array_norm2(val_a: float, shape: Tuple):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_c = vector_a.norm2()
    vector_out = vector_a.dot(vector_a)
    assert torch.allclose(vector_c, vector_out)


@pytest.mark.parametrize(
    "val_a,shape,val_out",
    [
        (1, (1), 1.7320508075688772),
        (2.0, (1), 3.4641),
        (3.0, (1), 5.1962),
    ],
)
def test_vec3array_norm(val_a: float, shape: Tuple, val_out: float):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
        torch.ones(shape, dtype=torch.float32) * val_a,
    )
    vector_c = vector_a.norm()
    vector_out = torch.ones(shape, dtype=torch.float32) * val_out
    assert torch.allclose(vector_c, vector_out)


@pytest.mark.parametrize(
    "x,y,z,shape",
    [
        (1, 1, 1, (1)),
        (2.0, 1, 1, (1)),
        (3.0, 0.5, 0.333, (1)),
    ],
)
def test_vec3array_normalized(x: float, y: float, z: float, shape: Tuple):
    vector_a = Vec3Array(
        torch.ones(shape, dtype=torch.float32) * x,
        torch.ones(shape, dtype=torch.float32) * y,
        torch.ones(shape, dtype=torch.float32) * z,
    )
    vector_c = vector_a.normalized()
    vector_out = torch.ones(shape, dtype=torch.float32)
    assert torch.allclose(vector_c.dot(vector_c), vector_out)
