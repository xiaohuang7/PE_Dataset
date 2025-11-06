from tianshou.utils import BasicLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Tuple, Union, Callable, Optional
from tensorboard.backend.event_processing import event_accumulator

class MyLogger(BasicLogger):
    def __init__(self, writer : SummaryWriter, valid_res: list):
        super().__init__(writer)
        self.valid_res = valid_res
    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["len"] = collect_result["lens"].mean()
            if step - self.last_log_train_step >= self.train_interval:
                self.write("train", step, {"n/ep": collect_result["n/ep"]})
                self.write("train", step, {"rew": collect_result["rew"]})
                self.write("train", step, {"len": collect_result["len"]})
                self.write("train", step, {"ntp": len(self.valid_res)})
                self.last_log_train_step = step
    
    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step