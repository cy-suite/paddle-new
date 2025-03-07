# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

from __future__ import annotations

from functools import partial

import paddle
from paddle import framework
from paddle.distributed.communication.batch_isend_irecv import (
    P2POp,
    batch_isend_irecv,
)

from ..utils.log_util import logger
from .pipeline_parallel import (
    FakeMicroDataset,
    HybridParallelOptimizer,
    PipelineParallel,
)
from .pp_utils.batch_comm_helper import BatchCommHelper
from .zero_bubble_utils import WeightGradStore

__all__ = []


def detach_and_requires_grad(x):
    o = x.detach()
    o.stop_gradient = False
    return o


class DualPipeVParallel(PipelineParallel):
    """
    An implementation of the DualPipeV, based on
    https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipe.py.
    """

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self.overlapped_forward_backward = hasattr(
            type(self._layers), "overlapped_forward_backward"
        )
        logger.info(
            f"Using DualPipeVParallel with overlapping forward backward={self.overlapped_forward_backward}"
        )

        self.num_ranks = self.num_stages
        self.group_rank = self.pp_group.get_group_rank(self.pp_group.rank)
        self.prev_rank = self.pp_group.ranks[
            (self.group_rank - 1) % self.pp_group.world_size
        ]
        self.next_rank = self.pp_group.ranks[
            (self.group_rank + 1) % self.pp_group.world_size
        ]

        # NOTE(zhangyuqin1998): The first rank has to broadcast the meta information
        # of the P2P communication after the first forward.
        self.need_broadcast_meta = self.is_pipeline_first_stage()
        self.need_recv_meta = not self.is_pipeline_first_stage()
        self._p2p_helper = BatchCommHelper(self._using_cache)

    def is_pipeline_first_stage(self):
        return self.group_rank == 0

    def is_pipeline_last_stage(self):
        return self.group_rank == self.num_ranks - 1

    def _reset_states(self):
        self.input_tensors: tuple[
            list[list[paddle.Tensor]], list[list[paddle.Tensor]]
        ] = ([], [])
        self.output_tensors: tuple[
            list[list[paddle.Tensor]], list[list[paddle.Tensor]]
        ] = ([], [])
        self.input_grad_tensors: tuple[
            list[list[paddle.Tensor]], list[list[paddle.Tensor]]
        ] = ([], [])
        self.output_grad_tensors: tuple[
            list[list[paddle.Tensor]], list[list[paddle.Tensor]]
        ] = ([], [])
        self.loss_tensors: list[paddle.Tensor] = []

        # The first value in the list corresponds to phase 0, and the second value corresponds to phase 1.
        self.current_f_acc_id: list[int] = [0, 0]
        self.current_b_acc_id: list[int] = [0, 0]
        self.current_send_f_acc_id: list[int] = [0, 0]
        self.current_send_b_acc_id: list[int] = [0, 0]
        self.current_recv_f_acc_id: list[int] = [0, 0]
        self.current_recv_b_acc_id: list[int] = [0, 0]
        self.comm_ops: list[P2POp] = []
        self.to_free: list[paddle.Tensor] = []

    def _get_forward_inputs(self, micro_datasets, phase, acc_id):
        is_first_stage = self.is_pipeline_first_stage() and phase == 0
        if is_first_stage:
            assert micro_datasets is not None
            self.input_tensors[phase].append([next(micro_datasets[phase])[0]])
        if self.forward_only:
            self.input_tensors[phase][acc_id] = None
        return self.input_tensors[phase][acc_id]

    def _get_forward_labels(self, micro_datasets, phase, acc_id):
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            assert micro_datasets is not None
            labels = next(micro_datasets[phase])[1]
            self._check_micro_batch_data_valid(labels)
            return labels
        else:
            return None

    def _compute_forward_loss(self, micro_datasets, phase, acc_id, logits):
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            labels = self._get_forward_labels(micro_datasets, phase, acc_id)
            loss_tensor = self._layers._loss_fn[0](*logits, labels)
            self._store_forward_loss(phase, loss_tensor)

    def _store_forward_tensors(self, phase, outputs):
        if self.is_pipeline_last_stage() and phase == 0:
            self.input_tensors[1].append(
                [detach_and_requires_grad(output) for output in outputs]
            )
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if not is_last_stage:
            self.output_tensors[phase].append(outputs)

    def _forward_compute(self, phase: int, micro_datasets=None) -> None:
        acc_id = self.current_f_acc_id[phase]
        self.current_f_acc_id[phase] += 1

        inputs = self._get_forward_inputs(micro_datasets, phase, acc_id)

        outputs = self._layers.forward(*inputs, chunk_id=phase)
        outputs = [outputs] if isinstance(outputs, paddle.Tensor) else outputs

        self._compute_forward_loss(micro_datasets, phase, acc_id, outputs)
        self._store_forward_tensors(phase, outputs)

    def _get_backward_inputs(self, phase, acc_id):
        outputs = self.output_tensors[phase][acc_id]
        self.output_tensors[phase][acc_id] = None
        output_grads = self.output_grad_tensors[phase][acc_id]
        self.output_grad_tensors[phase][acc_id] = None
        non_empty = [
            (t, g) for t, g in zip(outputs, output_grads) if g is not None
        ]
        outputs, output_grads = list(zip(*non_empty))
        return outputs, output_grads

    def _store_backward_tensors(self, phase, acc_id):
        inputs = self.input_tensors[phase][acc_id]
        self.input_tensors[phase][acc_id] = None
        input_grads = [t.grad for t in inputs if not t.stop_gradient]
        if self.is_pipeline_last_stage() and phase == 1:
            self.output_grad_tensors[0].append(input_grads)
        else:
            self.input_grad_tensors[phase].append(input_grads)

    def _store_forward_loss(self, phase, loss_tensor):
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            assert isinstance(
                loss_tensor, paddle.Tensor
            ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

            with paddle.amp.auto_cast(enable=False):
                if self.accumulate_steps > 1 and not self._delay_scale_loss:
                    loss_tensor = loss_tensor / self.accumulate_steps
            self.loss_tensors.append(loss_tensor)

    def _backward_compute(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return

        acc_id = self.current_b_acc_id[phase]
        self.current_b_acc_id[phase] += 1

        is_last_stage = self.is_pipeline_first_stage() and phase == 1

        WeightGradStore.enabled = enable_zb
        with paddle.amp.auto_cast(enable=False):
            if is_last_stage:
                loss = self.loss_tensors[acc_id]
                if self.scaler:
                    paddle.autograd.backward(self.scaler.scale(loss))
                else:
                    paddle.autograd.backward(loss)
            else:
                outputs, output_grads = self._get_backward_inputs(phase, acc_id)
                if len(outputs) > 0:
                    outputs = [t for t in outputs if not t.stop_gradient]
                    paddle.autograd.backward(
                        tensors=outputs,
                        grad_tensors=output_grads,
                    )
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()

        self._store_backward_tensors(phase, acc_id)

    def _forward_backward_compute(
        self,
        phase0: int,
        phase1: int,
        micro_datasets=None,
    ) -> None:
        if self.forward_only:
            self._forward_compute(phase0, micro_datasets)
            return

        if not self.overlapped_forward_backward:
            self._forward_compute(phase0, micro_datasets)
            self._backward_compute(phase1)
            return

        # pre-forward
        acc_id0 = self.current_f_acc_id[phase0]
        self.current_f_acc_id[phase0] += 1

        inputs0 = self._get_forward_inputs(micro_datasets, phase0, acc_id0)
        labels0 = self._get_forward_labels(micro_datasets, phase0, acc_id0)

        # pre-backward
        acc_id1 = self.current_b_acc_id[phase1]
        self.current_b_acc_id[phase1] += 1

        is_last_stage1 = self.is_pipeline_first_stage() and phase1 == 1
        if is_last_stage1:
            loss1 = self.loss_tensors[acc_id1]
            outputs1, output_grads1 = None, None
        else:
            loss1 = None
            outputs1, output_grads1 = self._get_backward_inputs(phase1, acc_id1)
            if len(outputs1) > 0:
                outputs1 = [t for t in outputs1 if not t.stop_gradient]

        # forward & backward
        module0 = partial(self._layers.forward, chunk_id=phase0)
        module1 = partial(self._layers.forward, chunk_id=phase1)
        outputs0, loss0 = self._layers.overlapped_forward_backward(
            module0,
            inputs0,
            self._layers._loss_fn[0],
            labels0,
            module1,
            loss1,
            outputs1,
            output_grads1,
            self.scaler,
        )

        # post-forward
        self._store_forward_tensors(phase0, outputs0)
        self._store_forward_loss(phase0, loss0)

        # post-backward
        self._store_backward_tensors(phase1, acc_id1)

    def _commit_and_wait_comm(self) -> None:
        if not self.comm_ops or len(self.comm_ops) == 0:
            return
        reqs = batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []
        self._free_tensors()

    def _weight_pass(self) -> None:
        if self.forward_only:
            return

        self._commit_and_wait_comm()

        # Assume FIFO
        WeightGradStore.pop()

    def _free_tensors(self) -> None:
        self._release_output(self.to_free)
        self.to_free = []

    def _recv_forward(self, phase: int) -> None:
        if (self.is_pipeline_first_stage() and phase == 0) or (
            self.is_pipeline_last_stage() and phase == 1
        ):
            return

        self.current_recv_f_acc_id[phase] += 1

        tensors = self._p2p_helper.append_irecv(
            self.comm_ops,
            self.prev_rank if phase == 0 else self.next_rank,
            self.pp_group,
        )
        self.input_tensors[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        if (self.is_pipeline_first_stage() and phase == 1) or (
            self.is_pipeline_last_stage() and phase == 0
        ):
            return

        acc_id = self.current_send_f_acc_id[phase]
        self.current_send_f_acc_id[phase] += 1
        tensors = self.output_tensors[phase][acc_id]

        self._p2p_helper.append_isend(
            self.comm_ops,
            tensors,
            self.next_rank if phase == 0 else self.prev_rank,
            self.pp_group,
            self.need_broadcast_meta,
        )
        self.need_broadcast_meta = False

        self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_pipeline_first_stage() and phase == 1) or (
            self.is_pipeline_last_stage() and phase == 0
        ):
            return

        self.current_recv_b_acc_id[phase] += 1
        tensors = self._p2p_helper.append_irecv(
            self.comm_ops,
            self.next_rank if phase == 0 else self.prev_rank,
            self.pp_group,
        )
        self.output_grad_tensors[phase].append(tensors)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_pipeline_first_stage() and phase == 0) or (
            self.is_pipeline_last_stage() and phase == 1
        ):
            return

        acc_id = self.current_send_b_acc_id[phase]
        self.current_send_b_acc_id[phase] += 1
        tensors = self.input_grad_tensors[phase][acc_id]
        self.input_grad_tensors[phase][acc_id] = None

        self._p2p_helper.append_isend(
            self.comm_ops,
            tensors,
            self.prev_rank if phase == 0 else self.next_rank,
            self.pp_group,
        )

    def _forward_pass(
        self,
        phase: int,
        micro_datasets=None,
        recv: bool = True,
        send: bool = True,
    ) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()

        self._forward_compute(phase, micro_datasets)

        if send:
            self._send_forward(phase)

    def _backward_pass(
        self,
        phase: int,
        enable_zb: bool = False,
        recv: bool = True,
        send: bool = True,
    ) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()

        self._backward_compute(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_pass(
        self,
        phase0: int,
        phase1: int,
        micro_datasets=None,
        recv0: bool = True,
    ) -> None:
        if recv0:
            self._recv_forward(phase0)
        self._recv_backward(phase1)
        self._commit_and_wait_comm()

        self._forward_backward_compute(phase0, phase1, micro_datasets)

        self._send_forward(phase0)
        self._send_backward(phase1)

    def _wrap_data(self, data, phase):
        """
        for backward compatibility, wrap data to Fake FakeMicroDataset if it is of type list or tuple
        """
        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            return data

        micro_dataset = FakeMicroDataset(
            data,
            self.is_pipeline_first_stage() and phase == 0,
            self.is_pipeline_first_stage() and phase == 1,
            self.accumulate_steps,
            self.micro_batch_size,
        )
        return micro_dataset

    def _prepare_training(self, data, optimizer, lr_scheduler):
        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'

        assert (
            framework._dygraph_tracer()._has_grad
        ), 'Please enable the generation of gradients.'

        if self.is_pipeline_first_stage():
            assert (
                data is not None
            ), "For the first and the last stage, the data must be set."
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        self.register_sharding_comm_overlap_hook(optimizer)

        return data

    def _broadcast_final_loss(self):
        loss_sum_tensor = paddle.zeros([1], "float32")
        if self.is_pipeline_first_stage():
            assert (
                len(self.loss_tensors) > 0
            ), "train_batch() in last stage should obtain valid loss"
            for loss in self.loss_tensors:
                loss_sum_tensor += loss.detach().astype("float32")
            if self._delay_scale_loss:
                loss_sum_tensor /= self.accumulate_steps

        paddle.distributed.all_reduce(
            loss_sum_tensor, group=self.pp_group, sync_op=True
        )
        return loss_sum_tensor

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
    ):
        self.scaler = scaler

        rank = self.group_rank
        num_ranks = self.num_ranks
        assert (
            self.accumulate_steps > 0 and self.accumulate_steps >= num_ranks * 2
        ), f"{self.accumulate_steps=}, {num_ranks=}"
        self.forward_only = forward_only

        self._reset_states()

        # NOTE(zhangyuqin1998): Tensors to be sent or received must have a
        # consistent shape and data type throughout the entire pipeline. We
        # broadcast the meta info in the first forward of the first rank.
        self._p2p_helper.recv_meta_from_head(self.pp_group, self.need_recv_meta)
        self.need_recv_meta = False

        micro_dataset_phase0 = self._wrap_data(data, 0)
        micro_dataset_phase1 = self._wrap_data(data, 1)
        micro_datasets = [micro_dataset_phase0, micro_dataset_phase1]

        # Step 1: nF0
        step_1 = (num_ranks - rank - 1) * 2
        for i in range(step_1):
            self._forward_pass(0, micro_datasets)

        # Step 2: nF0F1
        step_2 = rank + 1
        self._recv_forward(0)
        for i in range(step_2):
            self._forward_pass(0, micro_datasets, recv=False, send=False)
            self._recv_forward(0)
            self._forward_pass(
                1,
                micro_datasets,
                send=(not self.is_pipeline_last_stage()) or (i < step_2 - 1),
            )
            self._send_forward(0)

        # Step 3: nB1W1F1 (Use zero bubble)
        step_3 = num_ranks - rank - 1
        for i in range(step_3):
            self._backward_pass(1, enable_zb=True)
            self._recv_forward(1)
            self._weight_pass()
            self._forward_pass(1, micro_datasets, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        step_4 = self.accumulate_steps - num_ranks * 2 + rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_pipeline_last_stage():
                    # NOTE: We don't overlap these two passes to further reduce bubble size.
                    self._forward_pass(
                        0, micro_datasets, recv=False, send=False
                    )
                    self._send_forward(1)
                    self._backward_pass(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)
                else:
                    self._forward_backward_pass(
                        0, 1, micro_datasets, recv0=False
                    )
            else:
                self._forward_backward_pass(0, 1, micro_datasets)
            self._forward_backward_pass(1, 0, micro_datasets)

        # Step 5: nB1F1B0
        step_5 = num_ranks - rank - 1
        for i in range(step_5):
            self._backward_pass(1)
            self._forward_backward_pass(1, 0, micro_datasets)

        # Step 6: nB1B0 (The second half of the passes use zero bubble)
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            self._backward_pass(1, enable_zb=enable_zb)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            self._backward_pass(0, enable_zb=enable_zb)

        # Step 7: nWB0 (Use zero bubble)
        step_7 = num_ranks - rank - 1
        for i in range(step_7):
            self._weight_pass()
            self._backward_pass(0, enable_zb=True)

        # Step 8: nW
        step_8 = rank + 1
        for i in range(step_8):
            self._weight_pass()
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()

        self._layers.allreduce_shared_weight_gradients()

        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss()

        self._reset_states()
        return train_loss

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss
