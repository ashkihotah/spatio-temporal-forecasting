# Find the site-packages directory
SP_DIR=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

# Export the library path so TF can find the .so files
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SP_DIR/nvidia/cuda_rt/lib:$SP_DIR/nvidia/cudnn/lib:$SP_DIR/nvidia/cublas/lib:$SP_DIR/nvidia/cufft/lib:$SP_DIR/nvidia/curand/lib:$SP_DIR/nvidia/cusolver/lib:$SP_DIR/nvidia/cusparse/lib:$SP_DIR/nvidia/nccl/lib:$SP_DIR/nvidia/nvjitlink/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvrtc/lib')")


# # Export the PATH so TF can find the ptxas compiler binary
# export PATH=$PATH:$SP_DIR/nvidia/cuda_nvcc/bin

# export TF_GPU_ALLOCATOR=cuda_malloc_async

# export TF_ENABLE_ONEDNN_OPTS=0
# export LD_PRELOAD=/usr/lib/wsl/lib/libcuda.so.1

models = [
    '$(cat overrides/hidden-state-ar.txt) $(cat overrides/altieri-ml.txt) "model/cell/cell@model.encoder_cell.cell=GRU"',
    '$(cat overrides/hidden-state-ar.txt) $(cat overrides/altieri-ml.txt) $(cat overrides/altieri-GAP-LSTM.txt) "model/cell/cell@model.encoder_cell.cell=GAP-LSTM"'
    '$(cat overrides/hidden-state-ar.txt) $(cat overrides/altieri-ml.txt) $(cat overrides/altieri-MGAP-LSTM.txt) "model/cell/cell@model.encoder_cell.cell=MGAP-LSTM"'
]