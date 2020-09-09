import argparse

from config import config as cfg
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel

from tensorpack.predict import PredictConfig
from tensorpack.tfutils import SmartInit
from tensorpack.tfutils.export import ModelExporter

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--load', help='load a model for evaluation.', required=True)
	parser.add_argument('--output-pb', help='Save a model to .pb')
	args = parser.parse_args()

	MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

	predcfg = PredictConfig(
		model=MODEL,
		session_init=SmartInit(args.load),
		input_names=MODEL.get_inference_tensor_names()[0],
		output_names=MODEL.get_inference_tensor_names()[1])

	ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
