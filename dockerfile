USER ${NB_USER}
WORKDIR /home/jovyan

# 1) 先只做 env update，並打開 debug
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} env update \
             -p ${KERNEL_PYTHON_PREFIX} \
             --file environment.yml \
             --debug'

# 2) 環境清理
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} clean --all -f -y'

# 3) 列出安裝後的套件
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} list -p ${KERNEL_PYTHON_PREFIX}'
