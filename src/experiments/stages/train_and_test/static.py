import os

from dvc_mlops import StageContextManager, dvc_stage

from experiments.stages.train.static import main as train
from experiments.stages.test.static import main as test
from experiments import ROOT_DIR

@dvc_stage(root_dir=ROOT_DIR)
def main(ctx: StageContextManager):
    train(ctx=ctx)
    test(ctx=ctx)

    model_path = ctx.cfg.paths.model_path
    os.remove(model_path)

if __name__ == "__main__":
    main()