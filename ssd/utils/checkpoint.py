'''

The comments are created by Xiaoyu Zhu at 26 April.
*This checkpoint.py code has been modified by Xiaoyu Zhu for TDT4265 final project.

*Additional Support:
1. Added the support for continuing training on TDT4265 Dataset after pre-trained on Waymo Dataset.
    * when cfg.MODEL.BACKBONE.AFTER_TRAINED = True, the parameters of models pre-trained on Waymo Data can be loaded.
    details: Line69-75
   **There is an additional argument 'cfg' added in Line 23.
   

'''
import logging
import os
import torch
from ssd.utils.model_zoo import cache_url


class CheckPointer:
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self,
                 cfg,
                 model,
                 optimizer=None,
                 scheduler=None,
                 save_dir="",
                 save_to_disk=None,
                 logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.after_trained = cfg.MODEL.BACKBONE.AFTER_TRAINED
        self.after_file_dir = cfg.MODEL.BACKBONE.AFTER_TRAINED_File
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f  and not self.after_trained:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        #modeifed by Xiaoyu to load the pre-trained parameters on waymo-dataset. 7th April 2020
        elif not f and self.after_trained:
            self.logger.info("Loading pretrained on waymo from {}".format(self.after_file_dir))
            model = self.model
            checkpoint=torch.load(self.after_file_dir, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint.pop("model"))
            return {}
        
        
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        model = self.model

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer and not self.after_trained:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler and not self.after_trained:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        return torch.load(f, map_location=torch.device("cpu"))
