# 1. 指定 base image（含 Jupyter 與 conda/mamba）
FROM jupyter/minimal-notebook:python-3.6

# 2. 切換到 root 權限，才能安裝套件
USER root

# 3. 複製 environment.yml 到映像裡
COPY environment.yml /home/jovyan/environment.yml

# 4. 切回 jovyan，後面的 RUN 都以 jovyan 身份執行
USER ${NB_USER}
WORKDIR /home/jovyan

# 5. （Step 1）只做 env update，並開 debug 模式
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} env update \
             -p ${KERNEL_PYTHON_PREFIX} \
             --file environment.yml \
             --debug'

# 6. （Step 2）環境清理
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} clean --all -f -y'

# 7. （Step 3）列出安裝後的套件
RUN TIMEFORMAT='time: %3R' \
    bash -c 'time ${MAMBA_EXE} list -p ${KERNEL_PYTHON_PREFIX}'
