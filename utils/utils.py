import logging
from datetime import datetime


def logging_set_up(save_log_file: bool = True, logging_filename_prefix: str = None):
    # config logging to be compatible with the pytorch
    if (logging_filename_prefix is not None) & (not logging_filename_prefix.endswith("_")):
        logging_filename_prefix = f"{logging_filename_prefix}_"
    else:
        logging_filename_prefix = ""

    if save_log_file:
        filename = f"{logging_filename_prefix}loggings_on_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    else:
        filename = None
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=filename,
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger