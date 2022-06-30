# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

def checkpoint_wrapper(module, *args, **kwargs):
    if os.environ.get("USE_PTD_AC", "False") == "True":
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper as ptd_checkpoint_wrapper,
                apply_activation_checkpointing_wrapper,
            )
            #module = apply_activation_checkpointing_wrapper(module, checkpoint_wrapper_fn=ptd_checkpoint_wrapper)
            module = ptd_checkpoint_wrapper(module)
        except ImportError:
            raise ImportError(
                "Cannot find  torch.distributed.algorithms._checkpoint.checkpoint_wrapper."
            )
    else:
        try:
            from metaseq.modules.checkpoint_activation_wrapper.checkpoint_activations import (
                checkpoint_wrapper as _checkpoint_wrapper,
            )

            module = _checkpoint_wrapper(module, *args, **kwargs)

            if hasattr(module, "extra_repr"):
                orig_extra_repr = module.extra_repr
            else:
                orig_extra_repr = None

            def extra_repr():
                return (
                    f"[Metaseq Built-in checkpointed] {orig_extra_repr()}" if orig_extra_repr is not None else ""
                )

            module.extra_repr = extra_repr
        except ImportError:
            raise ImportError(
                "Cannot find fairscale.nn.misc.checkpoint_activations. "
                "Please install fairscale with: pip install fairscale"
            )

    return module
