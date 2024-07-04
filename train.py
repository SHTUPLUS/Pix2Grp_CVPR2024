"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lavis.runners.runner_base import setup_output_dir
from git import Git

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--job-name", default='default', help="job_comment")
    parser.add_argument("--git-commit", action="store_true")
    

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    job_id = f"{now()}-{args.job_name}-train"
    
    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.

    output_dir = cfg.run_cfg.output_dir 
    result_dir, output_dir = setup_output_dir(cfg_output_dir=output_dir, job_id=job_id)
    logger = setup_logger(str(output_dir), name='lavis', job_id=job_id)
    
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
        result_dir=result_dir, output_dir=output_dir
    )

    if get_rank() == 0:
        if args.git_commit:
            commit_code(logger, job_id)

        fetch_git_status(logger)

    runner.train()



# fetch git status
def fetch_git_status(logger):
    git = Git(".")
    commit_log = git.log().split("\n")
    branch_name = git.branch().split("\n")
    curr_branch = "master"
    for each in branch_name:
        if each.startswith("*"):
            curr_branch = each.strip("*").strip(" ")

    commit_id = commit_log[0]
    commit_date = commit_log[2]
    commit_comment = commit_log[4]
    status = git.status()

    logger.info(
        "\ncodebase git HEAD info:\nbranch: %s\n%s\n%s\n%s"
        % (curr_branch, commit_id, commit_date, commit_comment)
    )
    if "working directory clean" not in status and "working tree clean" not in status:
        logger.warning(
            "there has some un-commit modify in codebase, may cause this experiment un-reproducible"
        )


def commit_code(logger, job_id):
    stream = os.popen(f"git commit -a -m 'run for experiment: {job_id}'")
    output = stream.read()
    logger.info("commit code success")
    logger.info(output)




if __name__ == "__main__":
    main()
