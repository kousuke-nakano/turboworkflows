# -*- coding: utf-8 -*-
import os
import time

# define logger
from logging import getLogger, StreamHandler, Formatter

# file-manager modules
from .machine_handler import Machine, Machines_handler
from .file_manager_env import file_manager_test_dir

logger = getLogger("Turbo-Workflows").getChild(__name__)


class Data_transfer:
    def __init__(
        self,
        client_machine_name,
        server_machine_name,
        safe_mode=False,
        # bwlimit=1000
    ):

        self.client_machine = Machine(client_machine_name)
        self.server_machine = Machine(server_machine_name)
        self.machine_handler = Machines_handler(
            client_machine_name=client_machine_name,
            server_machine_name=server_machine_name,
        )
        self.safe_mode = safe_mode

    def put_objects(self, from_objects=[]):

        client_home = self.client_machine.file_manager_root
        server_home = self.server_machine.file_manager_root

        if self.safe_mode:
            if not self.client_machine.is_dir(client_home):
                logger.error(f"{client_home} is not found.")
                raise FileNotFoundError
            if not self.server_machine.is_dir(server_home):
                logger.error(f"{server_home} is not found.")
                raise FileNotFoundError

        if len(from_objects) == 0:
            logger.info("from_objects is not specified")
            logger.info(
                f"All files and dirs in local dir. will be rsynced to the corresponding remote dir."
            )

            local_current_dir = os.path.abspath(os.getcwd())

            if not (
                self.client_machine.machine_type == "local"
                and self.server_machine.machine_type == "local"
            ):
                if client_home not in local_current_dir:
                    logger.error(f"client_home = {client_home}")
                    logger.error(
                        "server-client_manager.py works only in the local_home dir."
                    )
                    raise ValueError
            client_dir = local_current_dir.replace(client_home, client_home)
            server_dir = local_current_dir.replace(client_home, server_home)

            self.machine_handler.put_dir(from_dir=client_dir, to_dir=server_dir)

        else:
            logger.info("from_objects is specified")
            logger.info(
                "The files and dirs in local dir. will be rsynced to the corresponding remote dir."
            )

            for object in from_objects:
                object_abs = os.path.abspath(object)
                if client_home not in object_abs:
                    logger.error(f"client_home = {client_home}")
                    logger.error(
                        "server-client_manager.py works only in the client_home dir."
                    )
                    raise ValueError
                from_object = object_abs.replace(client_home, client_home)
                to_object = object_abs.replace(client_home, server_home)
                if self.client_machine.is_file(file_name=from_object):
                    self.machine_handler.put(from_file=from_object, to_file=to_object)
                else:  # isdir(from_object)
                    self.machine_handler.put_dir(from_dir=from_object, to_dir=to_object)

    def get_objects(self, from_objects=[]):

        client_home = self.client_machine.file_manager_root
        server_home = self.server_machine.file_manager_root

        if self.safe_mode:
            if not self.client_machine.is_dir(client_home):
                logger.error(f"{client_home} is not found.")
                raise FileNotFoundError
            if not self.server_machine.is_dir(server_home):
                logger.error(f"{server_home} is not found.")
                raise FileNotFoundError

        logger.info(f"client_dir_root={client_home}")
        logger.info(f"server_dir_root={server_home}")

        local_current_dir = os.path.abspath(os.getcwd())
        if not (
            self.client_machine.machine_type == "local"
            and self.server_machine.machine_type == "local"
        ):
            if client_home not in local_current_dir:
                logger.error(f"client_home = {client_home}")
                logger.error(
                    "server-client_manager.py works only in the client_home dir."
                )
                raise ValueError
            else:
                client_dir = local_current_dir.replace(client_home, client_home)
                server_dir = local_current_dir.replace(client_home, server_home)

                if len(from_objects) == 0:
                    logger.info("from objects is not specified")
                    logger.info(
                        f"All files and dirs in remote dir. will be rsynced to the corresponding local dir."
                    )
                    self.machine_handler.get_dir(from_dir=server_dir, to_dir=client_dir)

                else:
                    logger.info("remote_objects_list is specified")
                    logger.info(
                        "The files and dirs in the remote_objects_list will be rsynced from the corresponding remote dir."
                    )

                    for object in from_objects:
                        from_object = os.path.join(server_dir, object)
                        if not self.server_machine.exist(object_name=from_object):
                            logger.error(
                                f"{from_object} does not exist on server_machine"
                            )
                            raise FileNotFoundError
                        to_object = from_object.replace(server_home, client_home)
                        if self.server_machine.is_file(file_name=from_object):
                            self.machine_handler.get(
                                from_file=from_object, to_file=to_object
                            )
                        else:  # mysftp.is_dir(remote_dir=from_object):
                            self.machine_handler.get_dir(
                                from_dir=from_object, to_dir=to_object
                            )


if __name__ == "__main__":
    from logging import getLogger

    log_level = "DEBUG"
    logger = getLogger("file-manager")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    handler_format = Formatter(
        "Module-%(name)s, LogLevel-%(levelname)s, Line-%(lineno)d %(message)s"
    )
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    """
    server=Machine(machine="amd")
    logger.info(server.username)
    logger.info(server.get_job_list())

    server=Machine(machine="mac")
    logger.info(server.get_job_list())
    """

    data_transfer_test_dir = os.path.join(
        file_manager_test_dir, "data_transfer_manager"
    )
    os.chdir(data_transfer_test_dir)

    # trasfer_mac_amd=Data_transfer(local_machine_name="localhost", client_machine_name="localhost", server_machine_name="amd")
    # trasfer_mac_amd.put_objects(dryrun_flag=False)
    # trasfer_mac_amd.get_objects(dryrun_flag=False, delete_flag=False)

    trasfer_amd_kagayaki = Data_transfer(
        local_machine_name="localhost",
        client_machine_name="kagayaki",
        server_machine_name="amd",
    )
    trasfer_amd_kagayaki.put_objects(dryrun_flag=False)
