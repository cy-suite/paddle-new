# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from paddle.distributed import fleet

from . import (  # noqa: F401
    hybrid_parallel_util,
    log_util,
    mix_precision_utils,
    sequence_parallel_utils,
    tensor_parallel_utils,
)
from .fs import HDFSClient, LocalFS
from .ps_util import DistributedInfer

if TYPE_CHECKING:
    from paddle.nn import Layer

__all__ = ["LocalFS", "recompute", "DistributedInfer", "HDFSClient"]


def recompute(
    function: Layer | Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """
    recompute intermediate activations to save the memory.

    Parameters:
        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model
              whose intermediate activations will be released to save memory in forward stage and will be recomputed
              in backward stage for gradient calculation.
        *args(Tensor): inputs to the function.
        **kwargs(Dict): Kwargs should only contain two kinds of key-value params, the one is part of function's key-value params,
                        and the other contains ``preserve_rng_state`` and ``use_reentrant``. the key-value pair of ``preserve_rng_state``,
                        which is used to indicate whether to save the forward rng. If it is True, then the last forward rng value
                        will be restored when the forward recalculation of backpropagation is performed, its default value is True.
                        the key-value pair of ``use_reentrant`` is used to indicate which implementation of recompute you will be used.
                        ``use_reentrant=True`` means to use the PyLayer implementation of recompute, ``use_reentrant=False`` means to
                        use the Hook implementation of recompute, its default value is True.
    Returns:
        Output of function on args.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED, env:GPU)
            >>> import numpy as np
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle import nn
            >>> from paddle.io import BatchSampler, DataLoader, Dataset
            >>> from paddle.distributed.fleet.utils import recompute

            >>> # Define the process mesh for distributed training
            >>> mesh = dist.ProcessMesh([[0, 1, 2, 3], [4, 5, 6, 7]], dim_names=['dp', 'mp'])
            >>> dist.set_mesh(mesh)

            >>> class RandomDataset(Dataset):
            ...     def __init__(self, seq_len, hidden, num_samples=100):
            ...         super().__init__()
            ...         self.seq_len = seq_len
            ...         self.hidden = hidden
            ...         self.num_samples = num_samples
            ...     def __getitem__(self, index):
            ...         inputs = np.random.rand(self.seq_len, self.hidden).astype('float32')
            ...         labels = np.random.rand(self.seq_len, self.hidden).astype('float32')
            ...         return (inputs, labels)
            ...     def __len__(self):
            ...         return self.num_samples

            >>> class MlpModel(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.w0 = paddle.nn.Linear(1024, 4096, bias_attr=False)
            ...         self.w1 = paddle.nn.Linear(4096, 1024, bias_attr=False)
            ...     def forward(self, x):
            ...         y = self.w0(x)
            ...         z = recompute(self.w1, y, use_reentrant=True)
            ...         return z

            >>> # Initialize the model and optimizer
            >>> with paddle.LazyGuard():
            ...     model = MlpModel()
            >>> opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

            >>> # Configure distributed strategy
            >>> parallel_config = {
            ...     "dp_config": {'sharding_level': 1},
            ...     "mp_config": {"parallelize_plan": {"w0": dist.ColWiseParallel(), "w1": dist.RowWiseParallel()}},
            ... }
            >>> dist_model, dist_opt = dist.parallelize(model, opt, config=parallel_config)
            >>> for p in dist_model.parameters():
            ...     p.initialize()

            >>> # Prepare the dataset and dataloader
            >>> dataset = RandomDataset(128, 1024)
            >>> sampler = BatchSampler(dataset, batch_size=4)
            >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
            >>> dataloader = dist.shard_dataloader(dataloader, meshes=[mesh], shard_dims="dp")

            >>> # Convert to static graph and enable recompute
            >>> loss_fn = nn.MSELoss()
            >>> dist_strategy = dist.Strategy()
            >>> dist_strategy._recompute.enable = True
            >>> dist_model = dist.to_static(dist_model, dataloader, loss_fn, opt, strategy=dist_strategy)
            >>> dist_model.train()

            >>> # Training loop
            >>> for step, (inputs, labels) in enumerate(dataloader()):
            ...     loss = dist_model(inputs, labels)
            ...     print('step: ', step, ' loss: ', loss)

    """

    return fleet.recompute.recompute(function, *args, **kwargs)
