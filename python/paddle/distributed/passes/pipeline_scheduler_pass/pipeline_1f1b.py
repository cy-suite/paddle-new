# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import logging

import paddle
from paddle.base import core
from paddle.framework import (
    _current_expected_place_ as _get_device,
)

from ...utils.log_utils import get_logger
from ..pass_base import register_pass
from ..pass_utils import (
    forward_complete_op_role,
)
from .pipeline_pass_base import PipelinePassBase

RECV_FORWARD = "recv_forward"
FORWARD = "forward"
BACKWARD = "backward"
SEND_BACKWARD = "send_backward"
OPT = "optimizer"

logger = get_logger(logging.INFO)


@register_pass("pipeline_scheduler_1F1B")
class Pipeline1F1BPass(PipelinePassBase):

    def __init__(self):
        super().__init__()
        self.jobs_in_stable_phase = [BACKWARD, FORWARD]
        self.jobs_in_stable_phase_in_pir = [
            BACKWARD,
            RECV_FORWARD,
            SEND_BACKWARD,
            FORWARD,
        ]
        self.set_attr("enable_backward_forward_overlap", 0)

    def _create_job_list(self):
        if self._in_pir_mode:
            return self._create_job_list_in_pir()
        else:
            raise NotImplementedError(
                "_create_job_list() only support PIR now."
            )

    def _create_job_list_in_pir(self):
        num_micro_batches = self.get_attr("num_micro_batches")
        pp_stage = self.get_attr("pp_stage")
        pp_degree = self.get_attr("pp_degree")

        job_list = []
        assert (
            pp_degree <= num_micro_batches
        ), "Num of micro batches should larger than or equal to pp degree."

        micro_batch_in_warmup = pp_degree - pp_stage
        micro_batch_in_1f1b = num_micro_batches - micro_batch_in_warmup

        forward_micro_batch_id = 0
        for i in range(micro_batch_in_warmup):
            recv_fwd_job = core.Job(RECV_FORWARD)
            recv_fwd_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(recv_fwd_job)

            forward_job = core.Job(FORWARD)
            forward_job.set_micro_batch_id(forward_micro_batch_id)
            job_list.append(forward_job)
            forward_micro_batch_id += 1

        backward_micro_batch_id = 0
        for i in range(micro_batch_in_1f1b):
            for job_type in self.jobs_in_stable_phase_in_pir:
                job = core.Job(job_type)
                micro_batch_id = (
                    forward_micro_batch_id
                    if job_type.startswith(FORWARD)
                    or job_type.startswith(RECV_FORWARD)
                    else backward_micro_batch_id
                )
                job.set_micro_batch_id(micro_batch_id)
                job_list.append(job)
            forward_micro_batch_id += 1
            backward_micro_batch_id += 1

        for i in range(micro_batch_in_warmup):
            backward_job = core.Job(BACKWARD)
            backward_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(backward_job)

            send_bwd_job = core.Job(SEND_BACKWARD)
            send_bwd_job.set_micro_batch_id(backward_micro_batch_id)
            job_list.append(send_bwd_job)

            backward_micro_batch_id += 1

        opt_job = core.Job(OPT)
        opt_job.set_micro_batch_id(0)
        job_list.append(opt_job)
        return job_list

    def _partial_programs(self, program):
        raise NotImplementedError("pipeline_1f1b_pass() only support PIR now.")

    def _partial_pir_programs(self, program):
        enable_send_recv_overlap = self.get_attr("enable_send_recv_overlap")
        assert (
            not enable_send_recv_overlap
        ), "PIR does not support 1F1B with enable_send_recv_overlap yet."

        types = [RECV_FORWARD, FORWARD, BACKWARD, SEND_BACKWARD, OPT]
        prog_splitter = ProgramSplitter(program, types)
        sub_program_list = prog_splitter.split_programs()

        for i in range(len(types)):
            logger.debug(
                f"type = {types[i]}, sub_programs = {sub_program_list[i]}\n"
            )
        logger.debug(
            f"jobs_in_stable_phase = {self.jobs_in_stable_phase_in_pir}"
        )
        return types, sub_program_list


class ProgramSplitter:
    def __init__(self, main_program, job_types):
        assert job_types == [
            RECV_FORWARD,
            FORWARD,
            BACKWARD,
            SEND_BACKWARD,
            OPT,
        ]
        self._overlap_send_recv(main_program)
        forward_complete_op_role(main_program)
        self.job_types = job_types
        self.complete_ops = main_program.global_block().ops
        self.programs = self._clone_programs(main_program)
        self.ops_dict = {
            key: prog.global_block().ops for key, prog in self.programs.items()
        }
        self.blocks_dict = {
            key: prog.global_block() for key, prog in self.programs.items()
        }

        self.cur_place = self._get_cur_place()

    def _overlap_send_recv(self, program):
        # TODO(liym27): This function should not be in ProgramSplitter, move it to pipeline_pass_base.py after vpp fixed.
        for block in program.blocks:
            for op in block.ops:
                if op.name() == "pd_op.send_v2":
                    op.set_bool_attr("dynamic_shape", False)
                    op.set_bool_attr("use_calc_stream", True)
                    ring_id = op.attrs()["ring_id"]
                    op.set_execution_stream("send_recv_stream")
                    op.set_scheduling_priority(0)
                elif op.name() == "pd_op.recv_v2":
                    op.set_bool_attr("dynamic_shape", False)
                    op.set_bool_attr("use_calc_stream", True)
                    op.set_execution_stream("send_recv_stream")
                    op.set_scheduling_priority(0)

    def _clone_programs(self, program):
        prog_dict = {}
        for job_type in self.job_types:
            prog_dict[job_type] = program.clone()
        return prog_dict

    def _get_cur_place(self):
        place = _get_device()
        if isinstance(place, paddle.framework.CUDAPlace):
            place = paddle.framework.CUDAPlace(
                paddle.distributed.ParallelEnv().dev_id
            )
        cur_place = paddle.base.libpaddle.Place()
        cur_place.set_place(place)
        return cur_place

    def split_programs(self):
        region = "opt"
        for op_idx in range(len(self.complete_ops) - 1, -1, -1):
            op = self.complete_ops[op_idx]
            if op.op_role != -1:
                if op.op_role == 1:
                    region = "bwd"
                elif op.op_role == 0:
                    region = "fwd"
                elif op.op_role == 2:
                    region = "opt"

            if region == "opt":
                self._erase_op_from_other_programs(op_idx, OPT)
            elif region == "bwd" and op.name() == "pd_op.send_v2":
                self._handle_func(op_idx, SEND_BACKWARD, self.job_types[4:])
                self._erase_op_from_other_programs(op_idx, SEND_BACKWARD)
            elif region == "bwd" and op.name() != "pd_op.send_v2":
                self._handle_func(op_idx, BACKWARD, self.job_types[3:])
                self._erase_op_from_other_programs(op_idx, BACKWARD)
            elif region == "fwd" and op.name() != "pd_op.recv_v2":
                self._handle_func(op_idx, FORWARD, self.job_types[2:])
                self._erase_op_from_other_programs(op_idx, FORWARD)
            elif region == "fwd" and op.name() == "pd_op.recv_v2":
                self._handle_func(op_idx, RECV_FORWARD, self.job_types[1:])
                self._erase_op_from_other_programs(op_idx, RECV_FORWARD)
        progs = []
        for job_type in self.job_types:
            progs.append(self.programs[job_type])
        return progs

    def _erase_op_from_other_programs(self, op_idx, keep_job_type):
        for job_type in self.job_types:
            if job_type != keep_job_type:
                self.ops_dict[job_type][op_idx].erase()

    def _handle_func(self, op_idx, cur_job_type, suffixed_job_types):
        for idx in range(self.complete_ops[op_idx].num_results()):
            if self._result_is_used(suffixed_job_types, op_idx, idx):
                var_name = self._get_or_create_var_name(
                    self.ops_dict[cur_job_type], op_idx, idx
                )
            for job_type in suffixed_job_types:
                if self._result_is_used([job_type], op_idx, idx):
                    self._add_dependency_if_necessary(
                        cur_job_type, job_type, op_idx, idx, var_name
                    )
                    self._add_kwarg_and_replace(
                        self.blocks_dict[job_type],
                        self.ops_dict[job_type],
                        op_idx,
                        idx,
                        var_name,
                    )

    def _add_dependency_if_necessary(
        self, cur_job_type, next_job_type, op_idx, rst_idx, var_name
    ):
        if not (
            cur_job_type == BACKWARD and next_job_type == SEND_BACKWARD
        ) and not (cur_job_type == RECV_FORWARD and next_job_type == FORWARD):
            return

        first_used_idx = None
        first_used_op = None
        for used_op in (
            self.ops_dict[next_job_type][op_idx].result(rst_idx).all_used_ops()
        ):
            used_idx = self.ops_dict[next_job_type].index(used_op)
            if first_used_idx is None or used_idx < first_used_idx:
                first_used_idx = used_idx
                first_used_op = used_op
        self._add_dependency(
            self.ops_dict[cur_job_type][op_idx], first_used_op, var_name
        )

    def _add_dependency(self, recorder_op, waiter_op, name):
        '''
        Add the extra event dependency of the two operators.
        This function mainly aims for the cross-programs in pipeline parallelism,
        especial for the 'send_v2' 'recv_v2' etc.
        '''
        if not recorder_op.has_attr("force_record_event"):
            recorder_op.set_bool_attr("force_record_event", True)
        recorder_op.set_str_attr("event_to_record", name)
        waiter_op.set_str_array_attr("events_to_wait", [name])

    def _result_is_used(self, job_types, op_idx, rst_idx):
        is_used = False
        for job_type in job_types:
            is_used = (
                is_used
                or self.ops_dict[job_type][op_idx].result(rst_idx).use_empty()
                is False
            )
        return is_used

    def _get_or_create_var_name(self, cur_sub_ops, op_idx, rst_idx):
        var_name = None
        # case1: get var_name in current sub-program
        op = cur_sub_ops[op_idx]
        if op.name() == "pd_op.data" or op.name() == "builtin.parameter":
            var_name = op.result(rst_idx).name
        else:
            # case2: get var_name from shadow_output in complete program
            result_var = self.complete_ops[op_idx].result(rst_idx)
            shadow_output_op = None
            for used_op in result_var.all_used_ops():
                if used_op.name() == "builtin.shadow_output":
                    shadow_output_op = used_op
            if shadow_output_op is not None:
                var_name = shadow_output_op.attrs()["output_name"]

        if var_name is None:
            # case3: create var_name in current sub-program
            paddle.pir.set_insertion_point_after(op)
            var_name = (
                f"var_{op_idx}_{self.complete_ops[op_idx].name()}_{rst_idx}"
            )
            paddle._C_ops.set_persistable_value(op.result(rst_idx), var_name)
        return var_name

    def _add_kwarg_and_replace(self, block, ops, op_idx, rst_idx, var_name):
        ori_result = ops[op_idx].result(rst_idx)
        new_result_var = block.add_kwarg(var_name, ori_result.type())
        new_result_var.place_attr = self.cur_place
        new_result_var.persistable = ori_result.persistable
        ops[op_idx].result(rst_idx).replace_all_uses_with(new_result_var)
