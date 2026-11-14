# 官方格式的评测镜像：逐行读 stdin，逐行写 stdout，纯前向。
#
# 设计要点
# --------
# 1. **离线可构建**：numpy 用 vendor/wheels 里的本地 wheel，`--no-index` 不联网。
# 2. **镜像里没有数据集**：数据由平台挂载/管道喂入，镜像不带任何样本，
#    这样"没挂数据就跑出结果"的情况不可能发生（旧镜像踩过这个坑）。
# 3. **推理期零状态**：不拟合、不校准、不重训。见 docs/harness_自测.md 第三节。
#
# 构建：
#   docker build -f Dockerfile.harness -t zhixun-harness:1.0 .
# 运行（从文件喂）：
#   docker run --rm --network none -i zhixun-harness:1.0 < inputs.jsonl > outputs.jsonl
FROM python:3.11-slim

COPY vendor/wheels/ /wheels/
RUN pip install --no-index --find-links=/wheels numpy==1.26.4 \
 && rm -rf /wheels

COPY harness/ /app/harness/

WORKDIR /app
# 用绝对路径，避免平台设了别的 WORKDIR 就找不到脚本
ENTRYPOINT ["python", "/app/harness/predict.py"]
# 默认提交用的是我们十轮训出来的权重（A 0.7210 / C 50.9320）。
# 官方原版仍在 /app/harness/checkpoints/，传 --checkpoints 指过去即可复现基线 0.7140/52.8686。
CMD ["--checkpoints", "/app/harness/checkpoints_trained"]
