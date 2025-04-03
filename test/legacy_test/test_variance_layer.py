#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from utils import dygraph_guard, static_guard

import paddle


def ref_var(x, axis=None, unbiased=True, keepdim=False):
    ddof = 1 if unbiased else 0
    if isinstance(axis, int):
        axis = (axis,)
    if axis is not None:
        axis = tuple(axis)
    return np.var(x, axis=axis, ddof=ddof, keepdims=keepdim)


class TestVarAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.shape = [1, 3, 4, 10]
        self.axis = [1, 3]
        self.keepdim = False
        self.unbiased = True
        self.set_attrs()
        self.x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def set_attrs(self):
        pass

    def static(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape, self.dtype)
                out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x}, fetch_list=[out])
        return res[0]

    def dygraph(self):
        with dygraph_guard():
            x = paddle.to_tensor(self.x)
            out = paddle.var(x, self.axis, self.unbiased, self.keepdim)
        return out.numpy()

    def test_api(self):
        out_ref = ref_var(self.x, self.axis, self.unbiased, self.keepdim)

        out_dygraph = self.dygraph()
        np.testing.assert_allclose(out_ref, out_dygraph, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_dygraph.shape).all())

        out_static = self.static()
        np.testing.assert_allclose(out_ref, out_static, rtol=1e-05)
        self.assertTrue(np.equal(out_ref.shape, out_static.shape).all())


class TestVarAPI_dtype(TestVarAPI):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPI_axis_int(TestVarAPI):
    def set_attrs(self):
        self.axis = 2


class TestVarAPI_axis_list(TestVarAPI):
    def set_attrs(self):
        self.axis = [1, 2]


class TestVarAPI_axis_tuple(TestVarAPI):
    def set_attrs(self):
        self.axis = (1, 3)


class TestVarAPI_keepdim(TestVarAPI):
    def set_attrs(self):
        self.keepdim = False


class TestVarAPI_unbiased(TestVarAPI):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPI_alias(unittest.TestCase):
    def test_alias(self):
        with dygraph_guard():
            x = paddle.to_tensor(np.array([10, 12], 'float32'))
            out1 = paddle.var(x).numpy()
            out2 = paddle.tensor.var(x).numpy()
            out3 = paddle.tensor.stat.var(x).numpy()
            np.testing.assert_allclose(out1, out2, rtol=1e-05)
            np.testing.assert_allclose(out1, out3, rtol=1e-05)


class TestVarError(unittest.TestCase):
    def test_error(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', [2, 3, 4], 'int32')
                self.assertRaises(TypeError, paddle.var, x)


class TestVarAPIZeroSize(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float64'
        self.axis = None
        self.unbiased = True
        self.keepdim = False
        self.set_attrs()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.base.core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.input = np.array([], dtype=self.dtype).reshape([2, 0, 4])

    def set_attrs(self):
        pass

    def test_api(self):
        # Dynamic graph test
        with dygraph_guard():
            x_tensor = paddle.to_tensor(self.input)
            out_dy = paddle.var(
                x_tensor, self.axis, self.unbiased, self.keepdim
            )

        # Static graph test
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x_static = paddle.static.data('X', self.input.shape, self.dtype)
                out_static = paddle.var(
                    x_static, self.axis, self.unbiased, self.keepdim
                )
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.input}, fetch_list=[out_static])

        # Reference result (expecting nan for zero-size inputs)
        out_ref = ref_var(self.input, self.axis, self.unbiased, self.keepdim)

        # Verify results
        np.testing.assert_allclose(out_dy.numpy(), out_ref, equal_nan=True)
        np.testing.assert_allclose(res[0], out_ref, equal_nan=True)


class TestVarAPIZeroSize_Float32(TestVarAPIZeroSize):
    def set_attrs(self):
        self.dtype = 'float32'


class TestVarAPIZeroSize_Axis_0(TestVarAPIZeroSize):
    def set_attrs(self):
        self.axis = 0


class TestVarAPIZeroSize_Axis_Neg1(TestVarAPIZeroSize):
    def set_attrs(self):
        self.axis = -1


class TestVarAPIZeroSize_Axis_List(TestVarAPIZeroSize):
    def set_attrs(self):
        self.axis = [0, 1]


class TestVarAPIZeroSize_Axis_Tuple(TestVarAPIZeroSize):
    def set_attrs(self):
        self.axis = (1, 2)


class TestVarAPIZeroSize_Unbiased_False(TestVarAPIZeroSize):
    def set_attrs(self):
        self.unbiased = False


class TestVarAPIZeroSize_Keepdim_True(TestVarAPIZeroSize):
    def set_attrs(self):
        self.keepdim = True


if __name__ == '__main__':
    unittest.main()
