# Adsorption-ensemble-Pipline：全功能使用示例

这份示例把仓库的主要能力串起来：
- surface 检测与报告导出
- site primitives 生成
- pose sampling + postprocess（可选）
- pose → basin（relax + 异常过滤 + 能窗筛选 + 去重）
- basin → node（canonicalize + node_id）
- conformer_md（“MD → 特征 → 选择 → relax → 输出”）流程示例
- 用 plot.py 对 MACE 推理结果做 collate/plot

## 0. 快速可运行示例（不依赖 XTB，不强制 MACE）

在仓库根目录运行：

```bash
pip install -e .
python tools/full_repo_example.py --out-root artifacts/full_repo_example
```

输出：
- `artifacts/full_repo_example/adsorption_api/`：`pose_pool.extxyz / basins.extxyz / basins.json / nodes.json`
- `artifacts/full_repo_example/conformer_md/example/`：`ensemble.extxyz / summary.json / summary.txt`
- `artifacts/full_repo_example/example_summary.json`：汇总信息

也可以直接打开并运行 notebook：`examples/full_usage.ipynb`

如果你希望在这个“低层 API 示例”里也开启 MACE 特征去重（需要在 WSL 环境中能加载 MACE 模型）：

```bash
wsl -d Rocky9 bash -lc 'set -e
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
cd /mnt/d/Download/trae-research-code/Adsorption-ensemble-Pipline
python tools/full_repo_example.py \
  --out-root artifacts/full_repo_example_mace \
  --use-mace-dedup \
  --mace-model-path /root/.cache/mace/mace-omat-0-small.model \
  --mace-device cuda \
  --mace-dtype float64 \
  --skip-conformer-md'
```

## 1) 生成吸附系综（高层 sweep 工作流）

最常用入口是 sweep，它会把 surface/site/pose/postprocess/ensemble 全链路跑完，并生成报告：

```bash
python tools/run_pose_sampling_sweep.py ^
  --out-root artifacts/pose_sampling ^
  --slab-name fcc211 ^
  --max-molecules 1 ^
  --max-slabs 1 ^
  --max-combinations 1 ^
  --mace-model-path /root/.cache/mace/mace-omat-0-small.model ^
  --mace-desc-device cuda ^
  --mace-desc-dtype float64 ^
  --postprocess-enabled ^
  --ensemble-enabled ^
  --ensemble-relax-backend mace_batch ^
  --ensemble-dedup-metric mace_node_l2 ^
  --ensemble-mace-node-l2-threshold 0.20
```

每个 case（`<run_dir>/<slab>/<mol>/`）会产出：
- `pose_pool.extxyz`：原始姿态池
- `pose_final.extxyz` + 若干 PCA/直方图：postprocess 结果（如果启用）
- `basins.extxyz / basins.json / nodes.json`：吸附系综结果（如果启用 ensemble）

如果你只是做无 MACE 的 smoke run，可以把 `--ensemble-dedup-metric` 改回 `rmsd`，并去掉 `--mace-model-path / --mace-desc-*`。

## 2) 在 WSL + MACE 模型下跑 GPU（推荐跑法）

如果你在 WSL 里有 `mamba/conda` 环境 `mace_les`，并且模型在：
`/root/.cache/mace/mace-omat-0-small.model`

可以直接跑 demo：

```bash
wsl -d Rocky9 bash -lc 'set -e
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
cd /mnt/d/Download/trae-research-code/Adsorption-ensemble-Pipline
python tools/wsl_run_fcc211_co_demo.py \
  --out-root artifacts/wsl_demo_pose \
  --model-path /root/.cache/mace/mace-omat-0-small.model \
  --strict'
```

该脚本默认启用：
- postprocess（batch relax）
- ensemble（batch relax + basin/node 输出）
- 二级去重使用 `mace_node_l2`（逐原子特征 L2 求和）

## 3) Conformer-MD（分子构象系综）工作流

如果你有 XTB（`xtb` 可执行文件可用），可以用这些脚本作为入口：
- `tools/run_c6_conformer_search.py`
- `tools/run_conformer_param_sweep.py`
- `tools/run_conformer_final_report_600k.py`

其中 `tools/run_c6_conformer_search.py` 现在默认直接读取
`examples/C6H14.gjf`，并走 production-oriented 的
`selection_profile="adsorption_seed_broad"` 语义；如果你想做更严格的孤立态
筛选，可加 `--profile isolated_strict`。

如果没有 XTB，也可以参考 `tools/full_repo_example.py` 里用 FakeMDRunner 的方式把整个 pipeline（descriptor/selection/relax/output）跑通，用于调参/调流程。

## 4) 用 plot.py 批量推理与作图（MACE 数据集评估/可视化）

plot.py 支持三种 mode：
- `run`：读取 extxyz 分块 → MACE 推理 → 输出每块 npz
- `collate`：合并多个块的 npz
- `plot`：parity / PCA / 分布等图

示例（以 `mode=run` 为例）：

```bash
python plot.py --mode run \
  --input <your.extxyz> \
  --output-prefix artifacts/plot_run/out \
  --model /root/.cache/mace/mace-omat-0-small.model \
  --device cuda \
  --dtype float64
```
