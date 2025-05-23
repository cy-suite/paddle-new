# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import re
import subprocess

import yaml
from op_gen import (
    PD_MANUAL_OP_LIST,
    OpCompatParser,
    OpInfoParser,
    to_pascal_case,
)

PD_MANUAL_API_LIST = {
    'embedding_grad',
    'assign',
}

H_FILE_TEMPLATE = """

#pragma once

#include <vector>

#include "paddle/utils/optional.h"
#include "paddle/pir/core/value.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_api.h"

{body}

"""

CPP_FILE_TEMPLATE = """

#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

{body}

"""


NAMESPACE_TEMPLATE = """
namespace {namespace} {{
{body}
}} // namespace {namespace}
"""


API_DECLARE_TEMPLATE = """
{ret_type} {api_name}({args});
"""


API_IMPL_TEMPLATE = """
{ret_type} {api_name}({args}){{
    {inner_code}
}}

"""

API_INNER_CODE_TEMPLATE = """
    {check_data_type}
    {handle_optional_inputs}
    {in_combine}
    {compute_op}
    {handle_optional_outputs}
    {set_null_type}
    {out_split}
    {return_result}"""


OP_DISPATCH_TEMPLATE = """
    if ({cond}) {{
        {inner_code}
    }}"""

OP_DISPATCH_ERROR_TEMPLATE = """
    PADDLE_THROW(phi::errors::Unimplemented(
        "The kernel of ({op_name}) for input Value is unimplemented, please check the type of input Value."));"""


CHECK_DATA_TYPE_TEMPLATE = """
    {function}({inputs}, "{op_name}");"""

IF_TEMPLATE = """
    if ({condition}) {{
    {check_statement}
    }}"""

ELSE_IF_TEMPLATE = """
    else if ({condition}) {{
    {check_statement}
    }}"""

ELSE_TEMPLATE = """
    else {{
    {check_statement}
    }}"""


OPTIONAL_VECTOR_VALUE_INPUT_TEMPLATE = """
    paddle::optional<pir::Value> optional_{name};
    if (!{name}) {{
        optional_{name} = paddle::make_optional<pir::Value>(pir::Value());
    }} else {{
        auto optional_{name}_combine_op = ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>({name}.get());
        optional_{name} = paddle::make_optional<pir::Value>(optional_{name}_combine_op.out());
    }}"""

OPTIONAL_VALUE_INPUT_TEMPLATE = """
    paddle::optional<pir::Value> optional_{name};
    if (!{name}) {{
        optional_{name} = paddle::make_optional<pir::Value>(pir::Value());
    }} else {{
        optional_{name} = {name};
    }}"""

OPTIONAL_OPRESULT_OUTPUT_TEMPLATE = """
    paddle::optional<pir::Value> optional_{name};
    if (!IsEmptyValue({op_name}_op.result({index}))) {{
        optional_{name} = paddle::make_optional<pir::Value>({op_name}_op.result({index}));
    }}"""

OPTIONAL_VECTOR_OPRESULT_OUTPUT_TEMPLATE = """
    paddle::optional<std::vector<pir::Value>> optional_{name};
    if (!IsEmptyValue({op_name}_op.result({index}))) {{
        auto optional_{name}_slice_op = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>({op_name}_op.result({index}));
        optional_{name} = paddle::make_optional<std::vector<pir::Value>>(optional_{name}_slice_op.outputs());
    }}"""

SET_NULL_TYPE_TEMPLATE = """
    if (!{input}) {{
        {op_name}_op.result({index}).set_type(pir::Type());
    }}"""


COMBINE_OP_TEMPLATE = """
    auto {op_name} = ApiBuilder::Instance().GetBuilder()->Build<pir::CombineOp>({in_name});"""

SPLIT_OP_TEMPLATE = """
    auto {op_name} = ApiBuilder::Instance().GetBuilder()->Build<pir::SplitOp>({in_name});"""

COMPUTE_OP_TEMPLATE = """
    paddle::dialect::{op_class_name} {op_inst_name} = ApiBuilder::Instance().GetBuilder()->Build<paddle::dialect::{op_class_name}>({args});"""

OP_INPUT = 'pir::Value'
DENSE_TENSOR_TYPE = "paddle::dialect::DenseTensorType"
DATA_TYPE = "paddle::dialect::DataTypeAttribute"
VECTOR_TYPE = 'pir::VectorType'
INTARRAY_ATTRIBUTE = "paddle::dialect::IntArrayAttribute"

VALUE_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'pir::Value',
    'paddle::dialect::SelectedRowsType': 'pir::Value',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'std::vector<pir::Value>',
}
OPTIONAL_VALUE_TYPE_MAP = {
    'paddle::dialect::DenseTensorType': 'paddle::optional<pir::Value>',
    'paddle::dialect::SelectedRowsType': 'paddle::optional<pir::Value>',
    'pir::VectorType<paddle::dialect::DenseTensorType>': 'paddle::optional<std::vector<pir::Value>>',
}


def get_op_class_name(op_name):
    return to_pascal_case(op_name) + 'Op'


class CodeGen:
    def __init__(self) -> None:
        pass

    def _parse_yaml(self, op_yaml_files, op_compat_yaml_file):
        op_compat_parser = OpCompatParser(op_compat_yaml_file)

        op_yaml_items = []
        for yaml_file in op_yaml_files:
            with open(yaml_file, "r") as f:
                ops = yaml.safe_load(f)
                op_yaml_items = op_yaml_items + ops

        op_info_items = []
        for op in op_yaml_items:
            op_compat_item = op_compat_parser.get_compat(op['name'])
            if (
                op_compat_item is None
                and op['name'].endswith(('_grad', '_grad_'))
                and 'forward' in op
            ):
                op_compat_item = op_compat_parser.get_compat(
                    op['forward']['name']
                )

            if (
                op_compat_item is not None
                and op_compat_item['op'] == "pow"
                and 'scalar' in op_compat_item
            ):
                op_compat_item = op_compat_item.pop('scalar')
            if 'support_tensor' in op.keys() and op['support_tensor']:
                (
                    scalar_item,
                    int_array_item,
                ) = op_compat_parser.parse_support_tensor(op)
                op_compat_item['scalar'] = scalar_item
                op_compat_item['int_array'] = int_array_item

            op_info_items.append(OpInfoParser(op, op_compat_item))
        return op_info_items

    def _need_skip(self, op_info, op_name):
        return (
            op_info.infer_meta_func is None and op_name not in PD_MANUAL_OP_LIST
        )

    def _is_optional_input(self, op_info, input_name):
        name_list = op_info.input_name_list
        optional_list = op_info.input_optional_list
        if (
            input_name in name_list
            and optional_list[name_list.index(input_name)] == 'true'
        ):
            return True
        return False

    def _is_optional_output(self, op_info, output_name):
        op_names = op_info.op_phi_name
        for name in op_names:
            if name.endswith(('_grad', '_grad_')):
                return False
        inplace_map = op_info.inplace_map
        input_optional_list = op_info.input_optional_list
        input_name_list = op_info.input_name_list
        if inplace_map is None:
            return False

        if output_name in inplace_map.keys():
            input_index = input_name_list.index(inplace_map[output_name])
            if input_optional_list[input_index] == 'true':
                return True
        return False

    # =====================================
    # Gen declare functions
    # =====================================
    def _gen_api_inputs(self, op_info):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        ret = []
        for name, type, optional in zip(name_list, type_list, optional_list):
            if optional == 'true':
                ret.append(f'const {OPTIONAL_VALUE_TYPE_MAP[type]}& {name}')
            else:
                ret.append(f'const {VALUE_TYPE_MAP[type]}& {name}')
        return ', '.join(ret)

    def _gen_api_attrs(
        self, op_info, with_default, is_mutable_attr, is_vector_mutable_attr
    ):
        name_list = op_info.attribute_name_list
        type_list = op_info.attribute_build_arg_type_list
        default_value_list = op_info.attribute_default_value_list
        mutable_name_list = op_info.mutable_attribute_name_list
        mutable_type_list = op_info.mutable_attribute_type_list
        assert len(name_list) == len(type_list) == len(default_value_list)
        no_mutable_attr = []
        mutable_attr = []
        for name, type, default_value in zip(
            name_list, type_list, default_value_list
        ):
            if is_mutable_attr and name in mutable_name_list:
                if (
                    mutable_type_list[mutable_name_list.index(name)][0]
                    == INTARRAY_ATTRIBUTE
                    and is_vector_mutable_attr
                ):
                    mutable_attr.append(f'std::vector<{OP_INPUT}> {name}')
                else:
                    mutable_attr.append(f'{OP_INPUT} {name}')
                continue
            if with_default and default_value is not None:
                if type in ['float', 'double']:
                    default_value = default_value.strip('"')
                no_mutable_attr.append(f'{type} {name} = {default_value}')
            else:
                no_mutable_attr.append(f'{type} {name}')
        return ', '.join(mutable_attr + no_mutable_attr)

    def _gen_api_args(
        self,
        op_info,
        with_default_attr,
        is_mutable_attr,
        is_vector_mutable_attr,
    ):
        inputs = self._gen_api_inputs(op_info)
        attrs = self._gen_api_attrs(
            op_info, with_default_attr, is_mutable_attr, is_vector_mutable_attr
        )
        return (inputs + ', ' + attrs).strip(', ')

    def _gen_ret_type(self, op_info):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list
        intermediate_list = op_info.output_intermediate_list
        assert len(name_list) == len(type_list) == len(intermediate_list)

        output_num = len(type_list) - intermediate_list.count('true')
        if output_num > 1:
            ret = []
            for name, type, intermediate in zip(
                name_list, type_list, intermediate_list
            ):
                if intermediate == 'true':
                    continue
                if self._is_optional_output(op_info, name):
                    ret.append(OPTIONAL_VALUE_TYPE_MAP[type])
                else:
                    ret.append(VALUE_TYPE_MAP[type])
            return 'std::tuple<{}>'.format(', '.join(ret))
        elif output_num == 1:
            index = intermediate_list.index('false')
            name = name_list[index]
            if self._is_optional_output(op_info, name):
                return OPTIONAL_VALUE_TYPE_MAP[type_list[index]]
            else:
                return VALUE_TYPE_MAP[type_list[index]]
        elif output_num == 0:
            return 'void'

    def _gen_one_declare(
        self, op_info, op_name, is_mutable_attr, is_vector_mutable_attr
    ):
        return API_DECLARE_TEMPLATE.format(
            ret_type=self._gen_ret_type(op_info),
            api_name=op_name,
            args=self._gen_api_args(
                op_info, True, is_mutable_attr, is_vector_mutable_attr
            ),
        )

    def _gen_h_file(self, op_info_items, namespaces, h_file_path):
        declare_str = ""
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    self._need_skip(op_info, op_name)
                    or op_name in PD_MANUAL_API_LIST
                ):
                    continue
                declare_str += self._gen_one_declare(
                    op_info, op_name, False, False
                )
                if len(op_info.mutable_attribute_name_list) > 0:
                    declare_str += self._gen_one_declare(
                        op_info, op_name, True, False
                    )
                    if INTARRAY_ATTRIBUTE in {
                        type[0] for type in op_info.mutable_attribute_type_list
                    }:
                        declare_str += self._gen_one_declare(
                            op_info, op_name, True, True
                        )
        body = declare_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(h_file_path, 'w') as f:
            f.write(H_FILE_TEMPLATE.format(body=body))

    # =====================================
    # Gen impl functions
    # =====================================
    def _gen_handle_optional_inputs(self, op_info):
        name_list = op_info.input_name_list
        optional_list = op_info.input_optional_list
        type_list = op_info.input_type_list
        assert len(name_list) == len(optional_list) == len(type_list)
        ret = ""
        for name, optional, type in zip(name_list, optional_list, type_list):
            if optional == 'true':
                if VECTOR_TYPE in type:
                    ret += OPTIONAL_VECTOR_VALUE_INPUT_TEMPLATE.format(
                        name=name
                    )
                else:
                    ret += OPTIONAL_VALUE_INPUT_TEMPLATE.format(name=name)
        return ret

    def _gen_handle_optional_outputs(self, op_info, op_name):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list
        intermediate_list = op_info.output_intermediate_list
        ret = ""
        for i, (name, type, intermediate) in enumerate(
            zip(name_list, type_list, intermediate_list)
        ):
            if intermediate == 'true':
                continue
            if self._is_optional_output(op_info, name):
                if VECTOR_TYPE in type:
                    ret += OPTIONAL_VECTOR_OPRESULT_OUTPUT_TEMPLATE.format(
                        name=name,
                        op_name=op_name,
                        index=i,
                    )
                else:
                    ret += OPTIONAL_OPRESULT_OUTPUT_TEMPLATE.format(
                        name=name,
                        op_name=op_name,
                        index=i,
                    )
        return ret

    def _gen_set_null_type(self, op_info, op_name):
        name_list = op_info.output_name_list
        inplace_map = op_info.inplace_map
        if inplace_map is None:
            return ""

        ret = ""
        for i, out_name in enumerate(name_list):
            if self._is_optional_output(op_info, out_name):
                in_name = inplace_map[out_name]
                ret += SET_NULL_TYPE_TEMPLATE.format(
                    input=in_name, op_name=op_name, index=i
                )
        return ret

    def _gen_in_combine(self, op_info, is_mutable_attr, is_vector_mutable_attr):
        name_list = op_info.input_name_list
        type_list = op_info.input_type_list
        optional_list = op_info.input_optional_list
        assert len(name_list) == len(type_list) == len(optional_list)
        combine_op = ""
        combine_op_list = []
        for name, type, optional in zip(name_list, type_list, optional_list):
            if optional == 'false' and VECTOR_TYPE in type:
                op_name = f'{name}_combine_op'
                combine_op += COMBINE_OP_TEMPLATE.format(
                    op_name=op_name, in_name=name
                )
                combine_op_list.append(op_name)
            else:
                combine_op_list.append(None)

        if is_mutable_attr:
            name_list = op_info.mutable_attribute_name_list
            type_list = op_info.mutable_attribute_type_list
            assert len(name_list) == len(type_list)
            for name, type in zip(name_list, type_list):
                if type[0] == INTARRAY_ATTRIBUTE and is_vector_mutable_attr:
                    op_name = f'{name}_combine_op'
                    combine_op += COMBINE_OP_TEMPLATE.format(
                        op_name=op_name, in_name=name
                    )
                    combine_op_list.append(op_name)
                else:
                    combine_op_list.append(None)

        return combine_op, combine_op_list

    def _gen_compute_op_args(
        self, op_info, in_combine_op_list, is_mutable_attr
    ):
        input_name_list = op_info.input_name_list
        all_attr_list = op_info.attribute_name_list
        no_mutable_attr_list = op_info.non_mutable_attribute_name_list
        mutable_attr_list = op_info.mutable_attribute_name_list
        assert len(input_name_list) + len(mutable_attr_list) == len(
            in_combine_op_list
        ) or len(input_name_list) == len(in_combine_op_list)
        ret = []
        if is_mutable_attr:
            name_list = input_name_list + mutable_attr_list
        else:
            name_list = input_name_list

        for input_name, combine_op in zip(name_list, in_combine_op_list):
            if combine_op is None:
                if self._is_optional_input(op_info, input_name):
                    ret.append(f'optional_{input_name}.get()')
                else:
                    ret.append(input_name)
            else:
                ret.append(f'{combine_op}.out()')
        if is_mutable_attr:
            ret += list(no_mutable_attr_list)
        else:
            ret += list(all_attr_list)
        return ', '.join(ret)

    def _gen_compute_op(
        self, op_info, op_name, in_combine_op_list, is_mutable_attr
    ):
        op_class_name = to_pascal_case(op_name) + 'Op'
        op_inst_name = op_name + '_op'
        return (
            COMPUTE_OP_TEMPLATE.format(
                op_class_name=op_class_name,
                op_inst_name=op_inst_name,
                args=self._gen_compute_op_args(
                    op_info, in_combine_op_list, is_mutable_attr
                ),
            ),
            op_inst_name,
        )

    def _gen_out_split_and_ret_list(self, op_info, op_inst_name):
        name_list = op_info.output_name_list
        type_list = op_info.output_type_list
        intermediate_list = op_info.output_intermediate_list
        optional_list = op_info.output_optional_list
        assert (
            len(name_list)
            == len(type_list)
            == len(intermediate_list)
            == len(optional_list)
        )

        split_op_str = ""
        ret_list = []
        for i, (name, type, intermediate) in enumerate(
            zip(name_list, type_list, intermediate_list)
        ):
            if intermediate == 'true':
                continue
            if self._is_optional_output(op_info, name):
                ret_list.append(f'optional_{name}')
            elif VECTOR_TYPE in type:
                split_op_name = f'{name}_split_op'
                split_op_str += SPLIT_OP_TEMPLATE.format(
                    op_name=split_op_name, in_name=f'{op_inst_name}.result({i})'
                )
                ret_list.append(f'{split_op_name}.outputs()')
            else:
                ret_list.append(f'{op_inst_name}.result({i})')
        return split_op_str, ret_list

    def _gen_return_result(self, ret_list):
        if len(ret_list) > 1:
            return 'return std::make_tuple({});'.format(', '.join(ret_list))
        elif len(ret_list) == 1:
            return f'return {ret_list[0]};'
        elif len(ret_list) == 0:
            return 'return;'

    def _gen_check_data_type(self, op_info, op_name):
        mapping_input_name_to_type = dict(
            zip(op_info.input_name_list, op_info.input_type_list)
        )
        mapping_attr_name_to_type = dict(
            zip(op_info.attribute_name_list, op_info.attribute_type_list)
        )

        mapping_name_to_type = {
            **mapping_input_name_to_type,
            **mapping_attr_name_to_type,
        }

        mapping_input_name_to_optional = dict(
            zip(op_info.input_name_list, op_info.input_optional_list)
        )

        if (
            op_name in ["real_grad", "imag_grad"]
            or len(mapping_name_to_type) == 0
        ):
            return ""
        try:
            data_type_candidates = op_info.kernel_map['data_type']['candidates']
        except (KeyError, TypeError):
            data_type_candidates = None

        mapping_type_to_function_name = {
            f"{VECTOR_TYPE}<{DENSE_TENSOR_TYPE}>": 'CheckVectorOfValueDataType',
            DENSE_TENSOR_TYPE: 'CheckValueDataType',
            DATA_TYPE: 'CheckDataType',
        }

        if data_type_candidates is None or len(data_type_candidates) == 0:
            if len(op_info.input_name_list) == 0:
                return ""
            ret = ""
            for name in op_info.input_name_list[::-1]:
                type = mapping_input_name_to_type[name]
                optional = mapping_input_name_to_optional[name]
                if (
                    function_name := mapping_type_to_function_name.get(
                        type, None
                    )
                ) is None:
                    continue

                if optional == 'false':
                    if ret == "":
                        return CHECK_DATA_TYPE_TEMPLATE.format(
                            function=function_name,
                            inputs=f"{name}, \"{name}\"",
                            op_name=op_name,
                        )
                    else:
                        ret += ELSE_TEMPLATE.format(
                            check_statement=CHECK_DATA_TYPE_TEMPLATE.format(
                                function=function_name,
                                inputs=f"{name}, \"{name}\"",
                                op_name=op_name,
                            ).strip("\n")
                        )
                        return ret
                else:
                    if ret == "":
                        template = IF_TEMPLATE
                    else:
                        template = ELSE_IF_TEMPLATE

                    ret += template.format(
                        condition=name,
                        check_statement=CHECK_DATA_TYPE_TEMPLATE.format(
                            function=function_name,
                            inputs=f"{name}.get(), \"{name}\"",
                            op_name=op_name,
                        ).strip("\n"),
                    )
            return ret

        elif len(data_type_candidates) == 1:
            name = data_type_candidates[0]
            if name not in mapping_name_to_type:
                return ""
            type = mapping_name_to_type[name]
            if (
                function_name := mapping_type_to_function_name.get(type, None)
            ) is None:
                return ""
            return CHECK_DATA_TYPE_TEMPLATE.format(
                function=function_name,
                inputs=f"{name}, \"{name}\"",
                op_name=op_name,
            )
        elif len(data_type_candidates) == 2:
            dtype_name = data_type_candidates[0]
            value_name = data_type_candidates[1]
            dtype_type = mapping_name_to_type.get(dtype_name, None)
            value_type = mapping_name_to_type.get(value_name, None)
            if DENSE_TENSOR_TYPE != value_type or DATA_TYPE != dtype_type:
                return ""
            function_name = 'CheckDataTypeOrValue'
            return CHECK_DATA_TYPE_TEMPLATE.format(
                function=function_name,
                inputs=f"{dtype_name}, \"{dtype_name}\", {value_name}, \"{value_name}\"",
                op_name=op_name,
            )
        return ""

    def _gen_one_impl(
        self, op_info, op_name, is_mutable_attr, is_vector_mutable_attr
    ):
        ret = ''
        dispatch_kernel = None
        if op_info.kernel_map and 'dispatch' in op_info.kernel_map:
            dispatch_kernel = op_info.kernel_map['dispatch']

        if dispatch_kernel and len(dispatch_kernel.keys()) > 1:
            api_inner_code = ''
            for kernel_name in dispatch_kernel.keys():
                dispatch_input_type = dispatch_kernel[kernel_name][0]
                input_name = op_info.input_name_list
                input_optional = op_info.input_optional_list
                cond_list = []
                for i, type in enumerate(dispatch_input_type):
                    name = input_name[i]
                    optional = input_optional[i]
                    if type == 'dense':
                        if optional == 'true':
                            cond_list.append(
                                f'(!{name} || {name}->type().isa<paddle::dialect::DenseTensorType>())'
                            )
                        else:
                            cond_list.append(
                                f'{name}.type().isa<paddle::dialect::DenseTensorType>()'
                            )
                    elif type == 'selected_rows':
                        if optional == 'true':
                            cond_list.append(
                                f'(!{name} || {name}->type().isa<paddle::dialect::SelectedRowsType>())'
                            )
                        else:
                            cond_list.append(
                                f'{name}.type().isa<paddle::dialect::SelectedRowsType>()'
                            )

                ret_type = self._gen_ret_type(op_info)
                in_combine, in_combine_op_list = self._gen_in_combine(
                    op_info, is_mutable_attr, is_vector_mutable_attr
                )

                if op_name.endswith('_') and not kernel_name.endswith('_'):
                    kernel_name = kernel_name + '_'
                compute_op, op_inst_name = self._gen_compute_op(
                    op_info, kernel_name, in_combine_op_list, is_mutable_attr
                )
                if ret_type == 'void':
                    compute_op += f' (void){op_inst_name};'

                out_split, ret_list = self._gen_out_split_and_ret_list(
                    op_info, op_inst_name
                )

                if_inner_code = API_INNER_CODE_TEMPLATE.format(
                    check_data_type=self._gen_check_data_type(
                        op_info, kernel_name
                    ),
                    handle_optional_inputs=self._gen_handle_optional_inputs(
                        op_info
                    ),
                    in_combine=in_combine,
                    compute_op=compute_op,
                    handle_optional_outputs=self._gen_handle_optional_outputs(
                        op_info, kernel_name
                    ),
                    set_null_type=self._gen_set_null_type(op_info, kernel_name),
                    out_split=out_split,
                    return_result=self._gen_return_result(ret_list),
                )

                if_inner_code = if_inner_code.split('\n')
                if_inner_code = '\n'.join(
                    ['    ' + code for code in if_inner_code]
                )

                api_inner_code += OP_DISPATCH_TEMPLATE.format(
                    cond=' && '.join(cond_list), inner_code=if_inner_code
                )

            api_inner_code += OP_DISPATCH_ERROR_TEMPLATE.format(op_name=op_name)
            ret = API_IMPL_TEMPLATE.format(
                ret_type=ret_type,
                api_name=op_name,
                args=self._gen_api_args(
                    op_info, False, is_mutable_attr, is_vector_mutable_attr
                ),
                inner_code=api_inner_code,
            )

        else:
            ret_type = self._gen_ret_type(op_info)
            in_combine, in_combine_op_list = self._gen_in_combine(
                op_info, is_mutable_attr, is_vector_mutable_attr
            )
            compute_op, op_inst_name = self._gen_compute_op(
                op_info, op_name, in_combine_op_list, is_mutable_attr
            )
            if ret_type == 'void':
                compute_op += f' (void){op_inst_name};'

            out_split, ret_list = self._gen_out_split_and_ret_list(
                op_info, op_inst_name
            )

            kernel_name = (
                list(dispatch_kernel.keys())[0]
                if dispatch_kernel and len(dispatch_kernel.keys()) == 1
                else op_name
            )
            if op_name.endswith('_') and not kernel_name.endswith('_'):
                kernel_name = kernel_name + '_'
            api_inner_code = API_INNER_CODE_TEMPLATE.format(
                check_data_type=self._gen_check_data_type(op_info, kernel_name),
                handle_optional_inputs=self._gen_handle_optional_inputs(
                    op_info
                ),
                in_combine=in_combine,
                compute_op=compute_op,
                handle_optional_outputs=self._gen_handle_optional_outputs(
                    op_info, op_name
                ),
                set_null_type=self._gen_set_null_type(op_info, op_name),
                out_split=out_split,
                return_result=self._gen_return_result(ret_list),
            )

            ret = API_IMPL_TEMPLATE.format(
                ret_type=ret_type,
                api_name=op_name,
                args=self._gen_api_args(
                    op_info, False, is_mutable_attr, is_vector_mutable_attr
                ),
                inner_code=api_inner_code,
            )

        ret = re.sub(r' +\n', "", ret)
        return ret

    def _gen_cpp_file(self, op_info_items, namespaces, cpp_file_path):
        impl_str = ""
        for op_info in op_info_items:
            for op_name in op_info.op_phi_name:
                # NOTE:When infer_meta_func is None, the Build() function generated in pd_op
                # is wrong, so temporarily skip the automatic generation of these APIs
                if (
                    self._need_skip(op_info, op_name)
                    or op_name in PD_MANUAL_API_LIST
                ):
                    continue
                impl_str += self._gen_one_impl(op_info, op_name, False, False)
                if len(op_info.mutable_attribute_name_list) > 0:
                    impl_str += self._gen_one_impl(
                        op_info, op_name, True, False
                    )
                    if INTARRAY_ATTRIBUTE in {
                        type[0] for type in op_info.mutable_attribute_type_list
                    }:
                        impl_str += self._gen_one_impl(
                            op_info, op_name, True, True
                        )
        body = impl_str
        for namespace in reversed(namespaces):
            body = NAMESPACE_TEMPLATE.format(namespace=namespace, body=body)
        with open(cpp_file_path, 'w') as f:
            f.write(CPP_FILE_TEMPLATE.format(body=body))

    def gen_h_and_cpp_file(
        self,
        op_yaml_files,
        op_compat_yaml_file,
        namespaces,
        h_file_path,
        cpp_file_path,
    ):
        if os.path.exists(h_file_path):
            os.remove(h_file_path)
        if os.path.exists(cpp_file_path):
            os.remove(cpp_file_path)
        op_info_items = self._parse_yaml(op_yaml_files, op_compat_yaml_file)

        self._gen_h_file(op_info_items, namespaces, h_file_path)
        self._gen_cpp_file(op_info_items, namespaces, cpp_file_path)
        try:
            subprocess.run(['clang-format', '-i', h_file_path])
            subprocess.run(['clang-format', '-i', cpp_file_path])
        except Exception as e:
            pass


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Generate Dialect API Files By Yaml'
    )
    parser.add_argument('--op_yaml_files', type=str)
    parser.add_argument('--op_compat_yaml_file', type=str)
    parser.add_argument('--namespaces', type=str)
    parser.add_argument('--api_def_h_file', type=str)
    parser.add_argument('--api_def_cc_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = ParseArguments()

    op_yaml_files = args.op_yaml_files.split(",")
    op_compat_yaml_file = args.op_compat_yaml_file
    if args.namespaces is not None:
        namespaces = args.namespaces.split(",")
        api_def_h_file = args.api_def_h_file
        api_def_cc_file = args.api_def_cc_file

        code_gen = CodeGen()
        code_gen.gen_h_and_cpp_file(
            op_yaml_files,
            op_compat_yaml_file,
            namespaces,
            api_def_h_file,
            api_def_cc_file,
        )
