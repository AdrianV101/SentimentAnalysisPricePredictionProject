import logging
from colorlog import ColoredFormatter
import datetime

# setting up logging


def setup_logger(program_name):
    # Create a logger for your module
    logger = logging.getLogger(program_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler with dynamic file name
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_name = f'../Logs/{program_name}_{current_datetime}.log'
    file_handler = logging.FileHandler(log_file_name)

    # Set the log message format
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # get colors in console
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
    # add console handler
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter("%(log_color)s%(asctime)s [%(levelname)s] - %(message)s",
                                         datefmt='%Y-%m-%d %H:%M:%S', log_colors=log_colors)
    console_handler.setFormatter(console_formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
