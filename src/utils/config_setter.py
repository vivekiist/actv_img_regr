import argparse
import os
import sys
import json
# import logging
from utils.logger_utils import setup_logger

def load_config(timestamp):
	def load_json(args, json_logger):
		json_file_dict = args.json_file
		json_logger.info("Using file %s to setup configuration" % json_file_dict)

		try:
			with open(json_file_dict) as f:
				json_dict = json.load(f)
		except FileNotFoundError:
			json_logger.error('--json_load : file not found %s ' % json_file_dict)
			sys.exit(1)
		except json.JSONDecodeError:
			json_logger.error('--json_load: invalid JSON format in file %s' % json_file_dict)
			sys.exit(1)

		for key, value in json_dict.items():
			setattr(args, key, value)

	def parse_args(config_parser): 
		parser = argparse.ArgumentParser(parents=[config_parser])

		parser.add_argument("--no_cuda", action="store_true", default=False,
							help="disables CUDA training (default: False)")
		parser.add_argument("--data_parallel", action="store_true", default=False,
							help="enable data parallelism (default: start_epoch)")
		parser.add_argument("--seed", type=int, default=1,
							help="random seed (default: 1)")
		parser.add_argument("--root_dir_train", type=str,
							help="root of the training dataset")
		parser.add_argument("--root_dir_test", type=str,
							help="root of the test dataset")
		parser.add_argument("--param_file_train", type=str,
							help="file containing view params of the training dataset")
		parser.add_argument("--param_file_test", type=str,
							help="file containing view params of the test dataset")
		parser.add_argument("--root_out_dir", type=str,
							help="folder where the outputs of this run will be stored")				
		parser.add_argument("--resume", action="store_true", default=False,
							help="whether to resume training or not (default: False)")
		parser.add_argument("--chkpt", type=str, default="",
							help="path to the latest checkpoint (default: none)")
		parser.add_argument("--dvp", type=int, default=3,
							help="dimensions of the view parameters (default: 3)")
		parser.add_argument("--dvpe", type=int, default=512,
							help="dimensions of the view parameters' encode (default: 512)")
		parser.add_argument("--ch", type=int, default=64,
							help="channel multiplier (default: 64)")
		parser.add_argument("--sn", action="store_true", default=False,
							help="enable spectral normalization (default: False)")
		parser.add_argument("--use_mse_loss", action="store_true", default=False,
							help="enable MSE loss")
		parser.add_argument("--use_vgg_loss", action="store_true", default=False,
							help="enable perceptual loss computed by VGG19 model (default: False)")
		parser.add_argument("--use_gan_loss", action="store_true", default=False,
							help="gan loss (default: False)")
		parser.add_argument("--gan_loss_weight", type=float, default=1.0,
							help="weight of the GAN loss (default: 1.0)")
		parser.add_argument("--vgg_loss_weight", type=float, default=1.0,
							help="weight of the perceptual loss (default: 1.0)")
		parser.add_argument("--mse_loss_weight", type=float, default=1.0,
							help="weight of the MSE loss (default: 1.0)")
		parser.add_argument("--lr", type=float, default=1e-3,
							help="learning rate (default: 1e-3)")
		parser.add_argument("--d_lr", type=float, default=1e-3,
							help="learning rate of the discriminator (default: 1e-3)")
		parser.add_argument("--beta1", type=float, default=0.9,
							help="beta1 of Adam (default: 0.9)")
		parser.add_argument("--beta2", type=float, default=0.999,
							help="beta2 of Adam (default: 0.999)")
		parser.add_argument("--batch_size", type=int, default=80,
							help="batch size for training (default: 80)")
		parser.add_argument("--start_epoch", type=int, default=0,
							help="start epoch number (default: 0)")
		parser.add_argument("--epochs", type=int, default=10,
							help="number of epochs to train (default: 10)")
		parser.add_argument("--log_every", type=int, default=10,
							help="log training status every given number of batches (default: 10)")
		parser.add_argument("--check_every", type=int, default=20,
							help="save checkpoint every given number of epochs (default: 20)")
		parser.add_argument("--no_active_learning", action="store_true", default=False,
							help="enable active learning (default: False)")
		parser.add_argument("--data_gen_script", type=str, default="./gen_img.py",
							help="path of the script to generate new data")
		parser.add_argument("--raw_inp_file", type=str, default="../data/Isabel_pressure_raw/Pf25.binLE.raw_corrected_2_subsampled.vti",
							help="Path of the raw input data")
		parser.add_argument("--varname", type=str, default="ImageScalars",
							help="Name of the input variable for which new images are required (default: ImageScalars)")
		parser.add_argument("--num_new_samples", type=int, default=100,
							help="Number of samples to generate for Active Learning (default: 100)")
		parser.add_argument("--query_strategy", type=str, default="MSELoss",
							help="Query strategy for Active learning (MSELoss/VGG) (default: MSELoss)")
		parser.add_argument("--sampling_budget", type=int, default="5000",
							help="No of samples to train the model on (default: 5000)")
		return parser


	LOG_DIR = '../logs'
	LOG_DIR = os.path.join(LOG_DIR, timestamp)
	json_log_file = 'config_logger.log'
	json_logger, json_file_handler = setup_logger(LOG_DIR, json_log_file, mod_name = __name__)
	json_logger.info('Starting the load_config...')

	program_name = os.path.basename(sys.argv[0])
	config_parser = argparse.ArgumentParser(prog=program_name, description="InSituNet", add_help=False)
	config_parser.add_argument('--json_load', action="store_true", default=False, help='Enable loading from JSON file')
	config_parser.add_argument('--json_file', action="store", default=None, help='Configuration JSON file')

	args, left_argv = config_parser.parse_known_args()
	parser = parse_args(config_parser)

	if args.json_load:
		load_json(args, json_logger)

	args = parser.parse_args(left_argv + ['--json_file', str(args.json_file)], args)

	json_logger.info('The current run configuration is as follows: \n %s' , json.dumps(vars(args), indent=4))
	json_logger.info('The changes from %s configuration after CLI  are as follows: \n %s', args.json_file,  left_argv)

	json_logger.info('Exiting the load_config.')
	json_file_handler.close()
	return args