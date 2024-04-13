import logging
import os

def setup_logger(log_dir, log_file=None, mod_name = None):
	logger = logging.getLogger(mod_name)
	logger.setLevel(logging.INFO)

	# Create logs directory if it doesn't exist
	os.makedirs(log_dir, exist_ok=True)

	# Set up file handler
	file_handler = logging.FileHandler(os.path.join(log_dir, log_file))

	# Create a formatter to define the log message format
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
	file_handler.setFormatter(formatter)
	
	# Add the FileHandler to the logger
	logger.addHandler(file_handler)

	return logger, file_handler