#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

# python modules
import os
import sys

# set logger
from logging import getLogger, StreamHandler, Formatter

logger = getLogger("Turbo-Workflows").getChild(__name__)

# file-manager related path lists
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
file_manager_source_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
file_manager_root = os.path.abspath(os.path.join(file_manager_source_dir, "../"))
file_manager_config_dir = os.path.join(
    os.path.abspath(os.environ["HOME"]), ".turbofilemanager_config"
)
file_manager_config_template_dir = os.path.join(
    file_manager_source_dir,
    "template",
)
file_manager_test_dir = os.path.join(file_manager_root, "tests")

if __name__ == "__main__":
    from logging import getLogger

    log_level = "DEBUG"
    logger = getLogger("Turbo-Workflows")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter(
        "Module-%(name)s, LogLevel-%(levelname)s, Line-%(lineno)d %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    logger.info(file_manager_source_dir)
    logger.info(file_manager_root)
