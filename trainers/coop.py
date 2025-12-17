"""
Stub module to satisfy optional import of trainers.coop.
This file is intentionally minimal. If you need the original coop trainer,
replace this stub with the real implementation.
"""
import torch

from clip import clip


def load_clip_to_cpu(cfg):
	"""Load a CLIP model to CPU using the same logic as other trainers.

	This duplicates the helper used elsewhere in the repo so imports like
	`from trainers.coop import load_clip_to_cpu` succeed.
	"""
	backbone_name = cfg.MODEL.BACKBONE.NAME
	url = clip._MODELS[backbone_name]
	model_path = clip._download(url)

	try:
		# loading JIT archive
		model = torch.jit.load(model_path, map_location="cpu").eval()
		state_dict = None

	except RuntimeError:
		state_dict = torch.load(model_path, map_location="cpu")

	model = clip.build_model(state_dict or model.state_dict())

	return model
