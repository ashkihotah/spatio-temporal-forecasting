import os

import hydra
from omegaconf import DictConfig

from experiments import ROOT_DIR
from dvc_mlops.run import Orchestrator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
os.environ["ABSL_LOG_LEVEL"] = "3" # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL

if __name__ == "__main__":
    orchestrator = Orchestrator(
        root_dir=ROOT_DIR,
        config_path=str(ROOT_DIR / "conf"),
        config_name="experiment"
    )
    orchestrator.main()
    # main()