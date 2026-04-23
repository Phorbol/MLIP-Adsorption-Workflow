# Adsorption Ensemble Pipeline

`Adsorption Ensemble Pipeline` 是一个面向表面反应自动探索的吸附结构生成库。  
它接收一个 `slab` 和一个 `adsorbate`，输出一组可去冗余、可落盘、可继续用于下游反应网络搜索的吸附态系综。

与传统“给定锚点原子 + 给定 top/bridge/hollow 模板”的工作流不同，这个项目的主线是：

- 从任意 slab 自动识别外表面
- 枚举 surface primitives，而不是只依赖命名位点
- 在 primitive 局部坐标系上做 anchor-free 的 SE(3) pose sampling
- 通过 batch relax 将初始 pose 映射到最终 basin
- 将 basin 规范化为 reaction-ready node

当前仓库已经具备完整的高层 API、可配置的多阶段筛选流程、可落盘的中间结果和最终结果，并带有一套面向论文实验的 artifact 生成脚本。

## Contents

- [What This Repository Provides](#what-this-repository-provides)
- [Workflow Overview](#workflow-overview)
- [Core Concepts](#core-concepts)
- [Installation](#installation)
- [How To Use This Repo](#how-to-use-this-repo)
- [Quick Start](#quick-start)
- [Recommended Production Usage](#recommended-production-usage)
- [Advanced Heterogeneous Support Recipe](#advanced-heterogeneous-support-recipe)
- [Current Default Workflow](#current-default-workflow)
- [Experimental Final Merge](#experimental-final-merge)
- [Visual Walkthrough](#visual-walkthrough)
- [Conformer Search Notes](#conformer-search-notes)
- [MACE Head And Device Semantics](#mace-head-and-device-semantics)
- [Surface and Site Dictionaries](#surface-and-site-dictionaries)
- [Basin and Node Example](#basin-and-node-example)
- [Output Files](#output-files)
- [Example Output Directory](#example-output-directory)
- [Representative Gallery](#representative-gallery)
- [Examples and Scripts](#examples-and-scripts)
- [Validation and Artifacts](#validation-and-artifacts)
- [Project Scope](#project-scope)

## What This Repository Provides

本仓库目前提供以下模块：

- `surface`：slab 分类、surface atom 检测、surface report 导出
- `site`：primitive 枚举、局部坐标系构建、equivalent-site basis 压缩
- `pose`：anchor-free / anchor-aware pose 采样
- `selection`：FPS、iterative FPS、energy + RMSD window、hierarchical / fuzzy 选择
- `relax`：identity / MACE / MACE batch relax backend
- `basin`：吸附 basin 去重、异常过滤、位点成键模式分析
- `node`：canonical node 导出，供下游 reaction exploration 使用
- `conformer_md`：柔性吸附质构象采样、筛选、松弛
- `workflows`：从 `slab + adsorbate` 一键生成最终吸附系综的高层接口

## Workflow Overview

当前高层工作流可概括为：

1. `slab -> SurfaceContext`
2. `SurfaceContext -> raw primitives`
3. `raw primitives -> inequivalent basis primitives`
4. `basis primitives x adsorbate -> pose pool`
5. `pose pool -> pre-relax selection`
6. `selected poses -> loose relax`
7. `relaxed structures -> post-relax selection`
8. `selected relaxed structures -> final basins`
9. `basins -> nodes`

当前最常用的入口函数是：

- [`generate_adsorption_ensemble`](adsorption_ensemble/workflows/api.py)
- [`make_sampling_schedule`](adsorption_ensemble/workflows/api.py)
- [`run_adsorption_workflow`](adsorption_ensemble/workflows/adsorption.py)

## Core Concepts

为了避免 README 和代码术语脱节，这里直接使用项目内部术语。

- `primitive`
  - 从 slab 几何/拓扑导出的原始吸附原语，不等同于最终的 named adsorption site
- `basis primitive`
  - 经过等价位点压缩后的 representative primitive
- `pose`
  - 在某个 primitive 条件下，对 adsorbate 进行一次刚体放置得到的候选构型
- `basin`
  - 经过松弛后收敛到的最终吸附能谷，是最终去重后的主要输出对象
- `ensemble`
  - 在给定 workflow 与筛选条件下保留下来的 basin 集合
- `node`
  - 面向下游反应网络搜索的规范化 basin 表示，包含 canonical atom order、binding graph、node id 等

这个项目的一个核心设计选择是：

- 不要求用户给出锚点原子
- 不强制用户把位点预命名成 top / bridge / hollow
- 通过 primitive + local frame + anchor-free SE(3) sampling 来搜索吸附姿态

## Installation

### Minimal

仓库根目录下：

```bash
pip install -e .
```

安装后可以直接：

```python
import adsorption_ensemble
```

### Optional Backends

`pyproject.toml` 中的基础依赖较轻，只包含：

- `numpy`
- `ase`
- `scipy`
- `matplotlib`

如果你需要真正的 batch relax / MACE 推理，请额外准备：

- 可用的 MACE 安装
- 一个可读取的 MACE 模型文件
- 建议使用 CUDA 环境

当前论文实验与较严肃的物理结果，默认使用：

- conda / mamba 环境：`mace_les`
- 模型：`mace-omat-0-small`
- device：`cuda`
- basin MACE descriptor dtype：`float64`
- basin MACE descriptor cuEq：`off`

如果你在做 batch relax，仍然可以把 relax backend 设成 `float32 + cuEq` 来换吞吐；但默认的 basin 去重已经切到 `mace_node_l2`，并使用 `fp64 + cueq off` 做更稳的特征比较。

## Conformer Search Notes

关于柔性吸附质的 `conformer_md` 设计、当前实现细节、预算语义、以及与
DockOnSurf / CREST / Molclus-like 工作流的对照说明，见：

- [`docs/theory/conformer_md_selection_and_budget.md`](docs/theory/conformer_md_selection_and_budget.md)
- [`docs/theory/conformer_md_production_design_zh.md`](docs/theory/conformer_md_production_design_zh.md)
- [`docs/theory/flexible_adsorption_case_design_zh.md`](docs/theory/flexible_adsorption_case_design_zh.md)

如果你想直接用仓库里的真实输入做一次生产式构象搜索，推荐从
[`examples/C6H14.gjf`](examples/C6H14.gjf) 开始，并用：

```bash
python tools/run_c6_conformer_search.py \
  --input examples/C6H14.gjf \
  --profile adsorption_seed_broad \
  --work-root artifacts/c6h14_profile_compare/conformer_md \
  --save-all-frames
```

这条入口默认采用当前推荐的 adsorption-coupled 语义：

- `selection_profile="adsorption_seed_broad"`
- `target_final_k=8`
- `preselect_k=max(96, 6 * target_final_k)`
- `metric_backend="mace"`
- MACE feature compare: `float64 + cueq off`
- MD seeds: `seed_mode="increment_per_run"`

如果你的目标是更严格的孤立态构象搜索，可以把 profile 切成：

```bash
python tools/run_c6_conformer_search.py \
  --input examples/C6H14.gjf \
  --profile isolated_strict
```

两种 profile 的差别不是“谁保留更多最终构象”，而是：

- `isolated_strict`
  - 更窄的能窗
  - 更大的最终预算，默认 `target_final_k=12`
- `adsorption_seed_broad`
  - 更宽的能窗，避免过早丢掉可能对表面吸附有利的气相次优构象
  - 更小的下游 handoff 预算，默认 `target_final_k=8`

当前 `conformer_md` 也支持把 raw 构象生成 backend 切成 `rdkit_embed`。例如：

```bash
python -m adsorption_ensemble.conformer_md.cli examples/C6H14.gjf \
  --generator-backend rdkit_embed \
  --selection-profile manual \
  --target-final-k 8 \
  --descriptor-backend geometry \
  --relax-backend identity \
  --rdkit-num-confs 96 \
  --rdkit-prune-rms-thresh 0.15 \
  --rdkit-optimize-forcefield mmff
```

如果你想直接比较 `xtb_md` 和 `rdkit_embed`，可以运行：

```bash
python tools/run_conformer_backend_benchmark.py examples/C6H14.gjf \
  --out-root artifacts/conformer_backend_benchmark_c6h14 \
  --selection-profile manual \
  --target-final-k 8 \
  --descriptor-backend geometry \
  --relax-backend identity
```

也可以通过环境变量设置模型路径：

```bash
export AE_MACE_MODEL_PATH=/path/to/mace-omat-0-small.model
```

如果你想直接跑一组面向柔性吸附质的生产式 case，可以使用：

```bash
python tools/run_flexible_adsorption_suite.py \
  --out-root artifacts/flexible_adsorption_suite \
  --mace-model-path /root/.cache/mace/mace-mh-1.model \
  --mace-device cuda
```

这个 suite 默认覆盖三类 DockOnSurf-style 关注点：

- `TiO2_110 + CH3COOH`
  - O-rich、可多锚定、对表面化学环境敏感
- `CuNi_fcc111_alloy + CH3CONH2`
  - 柔性酰胺在异质金属表面上的构象/取向/局域成分耦合
- `Pt_fcc111 + CH3CH2OH`
  - 小分子单转动自由度吸附质的构象与取向联动

如果当前机器拿不到 GPU，suite 会把 `MACE batch relax` 显式切到 `cpu`
执行，并在 runtime manifest 与 case summary 中记录这一点。

## MACE Head And Device Semantics

当前仓库里，`MACE head` 和 `device` 的生产语义已经显式化，不再依赖隐式假设。

### Head name

- 对 `conformer_md` 的 MACE inference 来说：
  - 如果你显式传了非空、且不等于 `Default` 的 `head_name`，代码会优先使用它
  - 不再静默退回到 model 自带的默认 head
- metadata 现在会写出：
  - `head_name`
  - `available_heads`

因此，当你在 CLI 或 Python 配置里写：

```bash
--mace-head-name omol
```

最终运行记录里就应该看到 `head_name = "omol"`；如果你没显式指定，才会退回到
model 暴露的首个可用 head。

### Device

- `conformer_md` 里的 MACE inference 会区分：
  - `requested_device`
  - `runtime_device`
- 如果你请求的是 `cuda`，但当前会话里 `torch.cuda.is_available()` 为 `False`：
  - inference 会自动回退到 `cpu`
  - metadata 会保留 `requested_device="cuda"` 与 `runtime_device="cpu"`

这意味着：

- 你可以从 artifact 中明确区分“配置本来想跑在哪”和“这次实际上跑在哪”
- 不会再出现 `torch.load(..., map_location="cuda")` 直接因无 GPU 崩掉的情况

对于 adsorption workflow 的 batch relax：

- 如果你要求“拿不到 CUDA 就直接失败”，用：
  - `MaceRelaxConfig(strict=True, device="cuda")`
- 如果你允许当前机器降级到 CPU 继续产出 artifact：
  - 显式把 relax backend 设到 `device="cpu"`
  - 或在你自己的 wrapper 里先解析 runtime device 再构造 backend

推荐实践：

- 结构比较 / basin dedup：
  - `float64 + cueq off`
- batch relax：
  - `float32 + cueq on` 仅在真正有 CUDA 时作为吞吐优化
  - CPU 路径不要把它误当成和 CUDA 同一性能档位

## How To Use This Repo

如果你只想知道“我到底该调用什么”，可以按下面三档来选：

### 1. 先验证 API 和输出结构

适用场景：

- 你刚装好环境，想确认 workflow 能跑通
- 你先关心目录结构、JSON 产物、可视化是否齐全
- 你暂时不关心物理可信度

入口：

- `generate_adsorption_ensemble(...)`
- `IdentityRelaxBackend()`

### 2. 常规生产使用

适用场景：

- 常规金属 slab 或较规则的表面
- 你希望直接拿到 `site -> pose -> basin -> node` 全链路结果
- 你有可用的 MACE 模型，且最好有 CUDA

入口：

- `generate_adsorption_ensemble(...)`
- `MACEBatchRelaxBackend(...)`
- `make_sampling_schedule("multistage_default")`

这是当前最推荐的用户入口。大多数用户不需要直接自己组装 `AdsorptionWorkflowConfig`。

### 3. 困难 case / 异质支撑 / 需要审计全过程

适用场景：

- 粗糙表面、台阶面、cluster-on-slab、异质金属支撑
- 你需要手动放大 pose coverage
- 你需要 case-scoped 地调整 primitive cap、pre-relax budget、异常过滤开关

入口：

- `make_adsorption_workflow_config(...)`
- `run_adsorption_workflow(...)`
- 显式的 slab/support pre-relax

这一档适合你已经知道为什么要偏离默认值，并且会审查 `raw_site_dictionary.json -> pose_pool.extxyz -> basins.json` 的全过程。

## Quick Start

下面是最小的一键示例。它的目标是验证 API 和输出目录，而不是给出最终可发表的吸附能谷。

```python
from pathlib import Path

from ase.build import fcc111, molecule

from adsorption_ensemble.workflows import generate_adsorption_ensemble, make_sampling_schedule
from adsorption_ensemble.relax import IdentityRelaxBackend

slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
adsorbate = molecule("NH3")

result = generate_adsorption_ensemble(
    slab=slab,
    adsorbate=adsorbate,
    work_dir=Path("artifacts/readme_quickstart/fcc111_nh3"),
    placement_mode="anchor_free",
    schedule=make_sampling_schedule("multistage_default"),
    # smoke / no-MACE path: explicitly pin the cheaper geometric dedup.
    dedup_metric="binding_surface_distance",
    signature_mode="provenance",
    basin_relax_backend=IdentityRelaxBackend(),
)

print(result.summary)
print(result.files)
```

这里故意把 `dedup_metric` 显式钉到 `binding_surface_distance`，是为了保留一个“不依赖 MACE 也能先把 API 跑通”的 smoke 路径。

如果你要按当前生产默认跑，不要沿用这一行；直接使用下一节的 `MACEBatchRelaxBackend(...)` 配方，并让 basin dedup 走默认的 `mace_node_l2`（或显式传 `dedup_metric="mace_node_l2"`）。

跑完后你最该先看：

- `result.summary`
- `result.files["site_dictionary_json"]`
- `result.files["pose_pool_extxyz"]`
- `result.files["basins_json"]`
- `result.files["nodes_json"]`

如果这个 smoke example 没问题，再切换到下面的 MACE 生产用法。

## Recommended Production Usage

对于真正的吸附结构生成，推荐直接使用 `generate_adsorption_ensemble + MACEBatchRelaxBackend`。这是当前 README 最推荐的成熟用户用法。

```python
from pathlib import Path

from ase.build import fcc211, molecule

from adsorption_ensemble.relax import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    DEFAULT_MACE_HEAD_NAME,
    generate_adsorption_ensemble,
    make_sampling_schedule,
)

slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
adsorbate = molecule("CO")

relax_backend = MACEBatchRelaxBackend(
    MaceRelaxConfig(
        model_path="/root/.cache/mace/mace-omat-0-small.model",
        device="cuda",
        dtype="float32",
        head_name=DEFAULT_MACE_HEAD_NAME,
        enable_cueq=True,
        strict=True,
        max_edges_per_batch=100000,
    )
)

result = generate_adsorption_ensemble(
    slab=slab,
    adsorbate=adsorbate,
    work_dir=Path("artifacts/readme_mace_example/pt211_co"),
    placement_mode="anchor_free",
    schedule=make_sampling_schedule("multistage_default"),
    dedup_metric="mace_node_l2",
    signature_mode="provenance",
    basin_overrides={
        "dedup_cluster_method": "greedy",
        "mace_dtype": "float64",
        "mace_enable_cueq": False,
        "desorption_min_bonds": 1,
        "energy_window_ev": 2.5,
    },
    basin_relax_backend=relax_backend,
)

print("n_pose_frames =", result.summary["n_pose_frames"])
print("n_selected_for_basin =", result.summary["n_pose_frames_selected_for_basin"])
print("n_basins =", result.summary["n_basins"])
print("work_dir =", result.files["work_dir"])
```

这个配方对应的用户心智模型很简单：

1. `generate_adsorption_ensemble` 负责把高层 workflow 串起来
2. `make_sampling_schedule("multistage_default")` 提供稳定默认筛选策略
3. `MACEBatchRelaxBackend` 负责真正有物理意义的 batch relax
4. basin dedup 默认使用 `mace_node_l2`（adsorbate-only、逐原子特征 L2）

建议你把下面这些文件当成标准检查点：

- `site_dictionary.json`
  - 看识别到了哪些 primitive / basis sites
- `pose_pool.extxyz`
  - 看采样到底放出了多少初猜
- `pose_pool_selected.extxyz`
  - 看哪些 pose 进入了真正的 basin relax
- `basins.json`
  - 看最后保留了哪些 basin，以及哪些 relaxed candidates 被拒绝
- `nodes.json`
  - 看下游 reaction-ready 的 canonical node

如果你只想做大多数常规 case，这一节就够用了。

## Advanced Heterogeneous Support Recipe

对于 `Pt(211)+Ag4+adsorbate` 这类困难体系，我们已经在 repo 内用正式 MACE 跑通过 `C6H6` 和 `CO` 两个 case。它们表明：对异质支撑体系，常常需要显式 pre-relax support，并且把 coverage 扩到默认值之上。

推荐把下面这套思路理解为“困难 case 配方”，而不是 repo 全局默认值：

```python
from pathlib import Path

from ase.build import molecule

from adsorption_ensemble.relax import MACEBatchRelaxBackend, MaceRelaxConfig
from adsorption_ensemble.workflows import (
    make_adsorption_workflow_config,
    make_default_surface_preprocessor,
    make_sampling_schedule,
    run_adsorption_workflow,
)

# Replace this with your own slab / cluster-on-slab builder.
slab = build_your_cluster_on_slab_structure()
adsorbate = molecule("CO")

relax_backend = MACEBatchRelaxBackend(
    MaceRelaxConfig(
        model_path="/root/.cache/mace/mace-omat-0-small.model",
        device="cuda",
        dtype="float32",
        head_name="omat_pbe",
        enable_cueq=True,
        strict=True,
        max_edges_per_batch=100000,
    )
)

# Important: pre-relax the bare support first.
relaxed_frames, _, _ = relax_backend.relax(
    frames=[slab.copy()],
    maxf=0.05,
    steps=200,
    work_dir=Path("artifacts/advanced_case/slab_relax"),
)
slab_relaxed = relaxed_frames[0]

schedule = make_sampling_schedule("multistage_default")
schedule.pre_relax_selection.max_candidates = 64

cfg = make_adsorption_workflow_config(
    work_dir=Path("artifacts/advanced_case/workflow"),
    placement_mode="anchor_free",
    single_atom=False,
    exhaustive_pose_sampling=True,
    dedup_metric="mace_node_l2",
    signature_mode="provenance",
    pose_overrides={
        "adaptive_height_fallback": True,
        "adaptive_height_fallback_step": 0.20,
        "adaptive_height_fallback_max_extra": 1.60,
        "adaptive_height_fallback_contact_slack": 0.60,
    },
    basin_overrides={
        "dedup_cluster_method": "greedy",
        "mace_dtype": "float64",
        "mace_enable_cueq": False,
        "desorption_min_bonds": 1,
        "surface_reconstruction_enabled": False,
        "energy_window_ev": 2.5,
    },
)
cfg.surface_preprocessor = make_default_surface_preprocessor(
    target_count_mode="off",
    target_surface_fraction=0.25,
)
cfg.pre_relax_selection = schedule.pre_relax_selection
cfg.basin_config.post_relax_selection = schedule.post_relax_selection
cfg.max_selected_primitives = None

result = run_adsorption_workflow(
    slab=slab_relaxed,
    adsorbate=adsorbate,
    config=cfg,
    basin_relax_backend=relax_backend,
)

print(result.summary)
```

这套配方对应的经验规则是：

- support 先 pre-relax，再做 adsorption workflow
- `exhaustive_pose_sampling=True` 用来放大 pose coverage
- `schedule.pre_relax_selection.max_candidates=64` 用来避免 pre-relax 阶段过早丢掉稀有 motif
- `cfg.max_selected_primitives=None` 用来避免 basis primitive 被固定上限截断
- `adaptive_height_fallback=True` 只建议在困难 case 里开启
- `surface_reconstruction_enabled=False` 只建议在你明确要做敏感性对比时 case-scoped 使用，不建议当作 repo 默认

目前 repo 内已有两组正式 run 可作为参考：

- `Pt(211)+Ag4+C6H6`
  - `352` poses -> `64` selected -> `37` basins
- `Pt(211)+Ag4+CO`
  - `414` poses -> `64` selected -> `36` basins

这两组结果说明，对于异质 cluster-on-slab case，成熟用法不是“只开默认 preset”，而是“默认 preset 为骨架，困难 case 再做 case-scoped coverage 扩展”。

## Current Default Workflow

当前最推荐的默认流程是：

- placement mode：`anchor_free`
- schedule：`multistage_default`
- surface target mode：`adaptive`
- pre-relax selection：fixed FPS
- pre-relax FPS budget：`k = 24`
- post-relax selection：energy window + RMSD window
- final basin dedup：`mace_node_l2 + hierarchical`

这里的 surface target mode 三个常见选项含义是：

- `off`
  - 不再强行把 surface atom 数量截断到某个目标值；保留 exposure / rescue 之后的全部暴露原子
- `fixed`
  - 按目标表面分数强制截断；四层 slab 的默认直觉等价于“保留约 1/4 总原子数作为顶面”
- `adaptive`
  - 先看次级暴露层是否只是弱尾巴
  - 如果像 `Fe_bcc111` 这样第二层 exposure 很弱，就自动退化到 `fixed`
  - 如果像 `Pt_fcc211`、`Ru_hcp10m10`、`TiO2(110)` 这样多层暴露本身有物理意义，就保持 `off`

对应的关键默认值来自当前 repo 内的受控实验：

- pre-relax budget sweep 结论：
  - [`pre_relax_budget_sweep_v1_report.md`](artifacts/autoresearch/paper_positioning/pre_relax_budget_sweep_v1_report.md)
- final dedup 结论：
  - [`polyatomic_final_dedup_suite_v1_report.md`](artifacts/autoresearch/paper_positioning/polyatomic_final_dedup_suite_v1_report.md)

`multistage_default` 当前对应的预筛选和后筛选定义见：

- [`adsorption_ensemble/workflows/api.py`](adsorption_ensemble/workflows/api.py)

`iterative_fps`、site occupancy convergence、PCA-grid convergence 也已经接入，但目前仍属于实验性 preset，不建议作为默认值。

## Experimental Final Merge

对于已经完成 `mace_node_l2` 去重、但仍怀疑存在“reference-equivalent basin false split”的体系，可以额外启用实验性的 final merge：

- `final_basin_merge_metric = "auto_ref_canonical_mace"`
- `final_basin_merge_node_l2_threshold = 0.02`

它的行为是：

- 用 `slab_ref` 而不是 relaxed slab 来定义 final-merge 的 canonical site equivalence
- 在 reference-canonical 分组内执行 MACE node-feature L2 merge
- 对单原子吸附质自动跳过，避免把 `H` 这类体系过度合并

示例：

```python
result = generate_adsorption_ensemble(
    slab=slab,
    adsorbate=adsorbate,
    work_dir=Path("artifacts/readme_ref_canonical_merge"),
    placement_mode="anchor_free",
    schedule=make_sampling_schedule("multistage_default"),
    dedup_metric="mace_node_l2",
    signature_mode="provenance",
    basin_overrides={
        "final_basin_merge_metric": "auto_ref_canonical_mace",
        "final_basin_merge_node_l2_threshold": 0.02,
    },
    basin_relax_backend=relax_backend,
)
```

截至当前 repo 内的 full-matrix post-hoc 结果，这个选项适合作为实验策略，不建议直接替代全局默认值。对应总结见：

- [`ref_canonical_merge_assessment.md`](artifacts/autoresearch/ref_canonical_merge_assessment_20260405/ref_canonical_merge_assessment.md)

## Visual Walkthrough

下面这些图都来自当前受控默认流程的真实输出，不是手工示意图。

### Example A: `fcc111 + NH3`

Case directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3)

| Surface primitives | Inequivalent sites |
| --- | --- |
| ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3/sites.png) | ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3/sites_inequivalent.png) |

| Site centers only | Site embedding PCA |
| --- | --- |
| ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3/sites_only.png) | ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3/site_embedding_pca.png) |

这个例子对应最熟悉的低指数金属表面，你可以直接检查：

- `ontop / bridge / fcc / hcp` 是否都被识别
- `site_dictionary.json` 是否和图中的 site 一一对应
- `pose_pool.extxyz -> pose_pool_selected.extxyz -> basins.extxyz` 的缩并路径是否合理

### Example B: `cu321 + C6H6`

Case directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6)

| Surface primitives | Inequivalent sites |
| --- | --- |
| ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6/sites.png) | ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6/sites_inequivalent.png) |

| Site centers only | Site embedding PCA |
| --- | --- |
| ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6/sites_only.png) | ![](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6/site_embedding_pca.png) |

这个例子更适合检查：

- rough / vicinal surface 上的 primitive 枚举是否足够丰富
- equivalent-site basis 压缩是否过度
- 芳香平面分子在复杂表面上是否保留了足够 basin 覆盖

## Surface and Site Dictionaries

和 `AutoAdsorbate` README 中最重合、也最值得直接对照审阅的部分，是 surface site 的结构化表示。  
我们当前的主输出不是 dataframe，而是一个更适合程序消费的 JSON 字典：

```json
{
  "meta": {...},
  "sites": {...},
  "kinds": {...},
  "basis_groups": {...}
}
```

### Top-level layout

| key | meaning |
| --- | --- |
| `meta` | site 字典摘要，例如 `n_sites`、不同 `kind` 的数量、basis group 数量 |
| `sites` | 以 `site_id -> site record` 组织的主 site 字典 |
| `kinds` | 按 `1c / 2c / 3c / 4c` 分组的 site id 列表 |
| `basis_groups` | 按 `basis_id` 分组的等价位点列表 |

### Site record fields

为了和 `AutoAdsorbate` 更容易对照，`site record` 现在同时保留项目内部字段和一组 AutoAdsorbate 风格别名字段。

| field | type | meaning |
| --- | --- | --- |
| `site_id` | `str` | 稳定 site 标识 |
| `kind` | `str` | primitive connectivity class，例如 `1c`、`2c`、`3c` |
| `atom_ids` | `list[int]` | 构成该 site 的 surface atom index |
| `center` | `list[float]` | site center 坐标 |
| `normal` | `list[float]` | site outward normal |
| `t1` | `list[float]` | local tangent basis 向量 1 |
| `t2` | `list[float]` | local tangent basis 向量 2 |
| `topo_hash` | `str` | 内部拓扑摘要 |
| `basis_id` | `int` | 等价位点 basis id |
| `site_label` | `str \| null` | 若有 ASE 命名位点信息，则记录如 `ontop`、`bridge`、`fcc`、`hcp` |
| `embedding` | `list[float] \| null` | primitive embedding |
| `coordinates` | `list[float]` | `AutoAdsorbate` 风格别名，等价于 `center` |
| `connectivity` | `int` | `AutoAdsorbate` 风格别名，等价于 `len(atom_ids)` |
| `topology` | `str` | `AutoAdsorbate` 风格别名；优先使用 `site_label`，否则回退到 `kind` |
| `n_vector` | `list[float]` | `AutoAdsorbate` 风格别名，等价于 `normal` |
| `h_vector` | `list[float]` | `AutoAdsorbate` 风格别名，当前对应 local frame 的 `t1` |
| `site_formula` | `str` | 由 site 关联的 surface atom 元素组成的局部化学式，例如 `Pt`、`Pt2`、`Pt3` |

### Example: `fcc111 + NH3`

下面这个例子来自当前默认 workflow 的：

- [`fcc111/NH3/site_dictionary.json`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3/site_dictionary.json)

| site_id | topology | connectivity | coordinates | n_vector | h_vector | site_formula |
| --- | --- | ---: | --- | --- | --- | --- |
| `site_00000` | `ontop` | 1 | `[0.000, 0.000, 16.790]` | `[0.000, 0.000, 1.000]` | `[1.000, 0.000, 0.000]` | `Pt` |
| `site_00001` | `bridge` | 2 | `[1.386, 0.000, 16.790]` | `[0.000, 0.000, 1.000]` | `[1.000, 0.000, 0.000]` | `Pt2` |
| `site_00002` | `fcc` | 3 | `[1.386, 0.800, 16.790]` | `[0.000, 0.000, 1.000]` | `[1.000, 0.000, 0.000]` | `Pt3` |
| `site_00003` | `hcp` | 3 | `[2.772, 1.600, 16.790]` | `[0.000, 0.000, 1.000]` | `[1.000, 0.000, 0.000]` | `Pt3` |

对应的真实 JSON 片段大致如下：

```json
{
  "site_id": "site_00002",
  "kind": "3c",
  "atom_ids": [48, 49, 52],
  "center": [1.3859, 0.8002, 16.7896],
  "normal": [0.0, 0.0, 1.0],
  "t1": [1.0, 0.0, 0.0],
  "t2": [0.0, 1.0, 0.0],
  "topo_hash": "3c|n=3|deg=6,6,6",
  "basis_id": 2,
  "site_label": "fcc",
  "coordinates": [1.3859, 0.8002, 16.7896],
  "connectivity": 3,
  "topology": "fcc",
  "n_vector": [0.0, 0.0, 1.0],
  "h_vector": [1.0, 0.0, 0.0],
  "site_formula": "Pt3"
}
```

如果你想同时审阅：

- site 可视化 PNG
- inequivalent site 分组
- site dictionary
- pose pool
- basin 和 node 输出

可以直接看这个索引：

- [`best_default_workflow_review_index_20260403.md`](artifacts/autoresearch/paper_positioning/best_default_workflow_review_index_20260403.md)

## Basin and Node Example

和 site dictionary 一样，`basins.json` 与 `nodes.json` 也是当前 workflow 的核心输出。  
它们分别回答两个不同问题：

- `basins.json`
  - 最终保留了哪些 relaxed adsorption states
  - 每个 basin 是由哪些候选 pose 支撑出来的
  - basin 的成键对、齿数、能量和 provenance 是什么
- `nodes.json`
  - 把 basin 转成下游 reaction exploration 可消费的规范化对象
  - 包含 canonical atom ordering、internal graph、binding graph、relative energy

### `basins.json` top-level layout

| key | meaning |
| --- | --- |
| `summary` | basin 层面的总体摘要 |
| `relax_backend` | 实际使用的 relax backend 信息 |
| `basins` | 最终保留 basin 的列表 |
| `rejected` | 被过滤掉的 relaxed candidate 及原因 |

### Basin record fields

| field | meaning |
| --- | --- |
| `basin_id` | basin 的唯一 id |
| `energy_ev` | basin 总能 |
| `denticity` | 最终识别到的齿数 |
| `signature` | basin 级签名 |
| `member_candidate_ids` | 支撑该 basin 的 pose candidate ids |
| `binding_pairs` | `adsorbate_atom_index -> slab_atom_index` 成键对 |
| `binding_adsorbate_indices` | 成键吸附质原子索引 |
| `binding_adsorbate_symbols` | 成键吸附质原子元素 |
| `binding_surface_atom_ids` | 成键表面原子索引 |
| `binding_surface_symbols` | 成键表面原子元素 |
| `member_site_labels` | basin 成员来自哪些 site label |
| `member_basis_ids` | basin 成员来自哪些 basis ids |
| `member_primitive_indices` | basin 成员来自哪些 primitive indices |
| `member_conformer_ids` | basin 成员来自哪些 conformer ids |
| `member_placement_modes` | basin 成员使用了哪些放置模式 |

### Node record fields

| field | meaning |
| --- | --- |
| `node_id` | canonical node id |
| `basin_id` | 对应的 basin id |
| `canonical_order` | 吸附质内部 canonical atom ordering |
| `atomic_numbers` | canonical ordering 下的原子序数 |
| `internal_bonds` | 吸附质内部键图 |
| `binding_pairs` | 吸附质-表面成键图 |
| `denticity` | 节点齿数 |
| `relative_energy_ev` | 相对最低能 basin 的能量 |
| `provenance` | 来源 basin signature 与 member candidates |

### Example: `fcc111 + NH3`

Case directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3)

Representative basin / node rows:

| basin_id | energy_ev | denticity | binding_pairs | member_site_labels | node_id | relative_energy_ev |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `0` | `-390.938629` | `1` | `[[0, 48]]` | `["ontop"]` | `18cfc1678ddc8c7678d9` | `0.000000` |
| `1` | `-390.859161` | `1` | `[[0, 48]]` | `["ontop"]` | `18cfc1678ddc8c7678d9` | `0.079468` |
| `2` | `-390.841797` | `1` | `[[0, 49]]` | `["ontop"]` | `ab5d07c79cfe014e9864` | `0.096832` |

这个例子有一个很重要的点：

- `basin_id` 是最终 relaxed state 的唯一身份
- `node_id` 是 canonicalized graph identity

因此多个 basin 可以共享同一个 `node_id`，如果它们在规范化后的吸附质内部图和吸附成键图上等价，但几何或能量上仍可区分。

### Example: `fcc111 + C6H6`

Case directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/C6H6`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/C6H6)

Representative basins:

| basin_id | energy_ev | denticity | binding_surface_atom_ids | binding_adsorbate_symbols | relative_energy_ev |
| --- | ---: | ---: | --- | --- | ---: |
| `0` | `-446.633972` | `5` | `[48, 49, 52, 61]` | `["C", "C", "C", "C", "C"]` | `0.000000` |
| `1` | `-446.395782` | `6` | `[48, 49, 60, 61]` | `["C", "C", "C", "C", "C", "C"]` | `0.238190` |
| `2` | `-446.391205` | `6` | `[49, 51, 52, 55, 60, 61]` | `["C", "C", "C", "C", "C", "C"]` | `0.242767` |

这个例子更像论文里会放的 basin diversity 图：

- 同一个 adsorbate 在同一 terrace 上可以形成多个高齿数 basin
- basin 间既有能量差，也有成键图差异
- basin 与 node 的关系比“找一个最低能吸附构型”更丰富

## Output Files

每个 case 的输出目录通常会包含如下文件：

| file | meaning |
| --- | --- |
| `sites.png` | primitive / basis site 的 2D 可视化 |
| `sites_inequivalent.png` | inequivalent site 可视化 |
| `sites_only.png` | site center 可视化 |
| `site_embedding_pca.png` | site embedding PCA 图 |
| `raw_site_dictionary.json` | raw primitive 字典 |
| `selected_site_dictionary.json` | 筛选后 primitive 字典 |
| `site_dictionary.json` | 主 site 字典，包含 `coordinates / connectivity / topology / n_vector / h_vector / site_formula` 等别名字段 |
| `pose_pool.extxyz` | 原始 pose pool |
| `pose_pool_selected.extxyz` | pre-relax selection 后的 pose pool |
| `pre_relax_selection.json` | pre-relax 选择诊断 |
| `basins.extxyz` | 最终 basin 结构 |
| `basins.json` | basin 级元数据与成键信息 |
| `basin_dictionary.json` | 更便于程序消费的 basin 字典 |
| `nodes.json` | 下游 reaction-ready nodes |
| `workflow_summary.json` | 工作流层面的摘要 |

如果你正在审阅当前最佳默认流程的实际输出，建议直接看这个索引：

- [`best_default_workflow_review_index_20260403.md`](artifacts/autoresearch/paper_positioning/best_default_workflow_review_index_20260403.md)

## Example Output Directory

下面这个目录树片段展示了“一个 case 跑完之后，你能拿到哪些东西”。

Example:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3)

```text
fcc111/NH3/
├── raw_site_dictionary.json
├── selected_site_dictionary.json
├── site_dictionary.json
├── sites.png
├── sites_only.png
├── sites_inequivalent.png
├── site_embedding_pca.png
├── pose_pool.extxyz
├── pose_pool_selected.extxyz
├── pre_relax_selection.json
├── basins.extxyz
├── basins.json
├── basin_dictionary.json
├── nodes.json
├── workflow_summary.json
├── final_dedup_suite_summary.json
└── final_dedup_suite_ablation.json
```

### Directory view: `basins.extxyz` vs `nodes.json`

如果你把这个 case 目录当成“文件夹截图”来看，最值得并排审的其实是下面这 4 个文件：

```text
fcc111/NH3/
├── basins.extxyz          <- 几何优先: 最终 relaxed basin 结构，一帧一个 basin
├── basins.json            <- basin 元数据: basin_id / energy / binding_pairs / provenance
├── basin_dictionary.json  <- 更适合程序后处理的 basin 字典视图
└── nodes.json             <- 图优先: canonical graph / canonical order / node_id
```

它们回答的是不同层面的问题：

| file | what it answers | when to open first |
| --- | --- | --- |
| `basins.extxyz` | “最终保留了哪些几何结构？” | 你要先审 relaxed adsorption geometry 是否合理 |
| `basins.json` | “这些结构各自的能量、成键对、member pose 是什么？” | 你要把几何结构和 basin 元数据一一对应 |
| `basin_dictionary.json` | “如果我要程序化消费 basin，最顺手的结构化字典是什么？” | 你要做统计、筛选或下游脚本 |
| `nodes.json` | “这些 basin 规范化后属于什么 node / graph identity？” | 你要看 canonical ordering、图等价性和 reaction-ready node |

对 `fcc111 + NH3` 这个最简单例子，推荐这样对照：

1. 先打开 `basins.extxyz`，逐帧看最终 3 个 relaxed basin。
2. 再打开 `basins.json`，确认每一帧对应的 `basin_id / energy_ev / binding_pairs`。
3. 最后看 `nodes.json`，确认哪些 basin 共享同一个 `node_id`。

这里有一个很重要的点：

- `basins.extxyz` / `basins.json` 区分的是几何态和能量态
- `nodes.json` 区分的是规范化后的图身份

所以多个 basin 可以对应同一个 `node_id`。  
在当前 `fcc111 + NH3` 的真实产物里，`basin_id = 0` 和 `basin_id = 1` 就共享同一个 `node_id`，但能量不同。

你可以把这些文件按三层来理解：

| layer | files | purpose |
| --- | --- | --- |
| `surface/site` | `site_dictionary.json`, `sites*.png` | 检查位点识别与等价位点压缩是否合理 |
| `pose` | `pose_pool.extxyz`, `pose_pool_selected.extxyz`, `pre_relax_selection.json` | 检查 pre-relax 采样和筛选 |
| `basin/node` | `basins.*`, `basin_dictionary.json`, `nodes.json` | 检查最终吸附态、去重与下游节点输出 |

### Review checklist

如果你要像审稿人一样快速审这个 case，推荐顺序：

1. `sites.png`
2. `sites_inequivalent.png`
3. `site_dictionary.json`
4. `pose_pool_selected.extxyz`
5. `pre_relax_selection.json`
6. `basins.extxyz`
7. `basins.json`
8. `nodes.json`

## Representative Gallery

下面这 3 个 case 是当前默认 workflow 最值得优先审阅的代表样例。  
可以把它们理解成 README 内的 miniature supplementary gallery。

### Case 1: `fcc111 + NH3`

Directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/NH3)

| metric | value |
| --- | ---: |
| surface atoms | `16` |
| basis primitives | `4` |
| raw pose frames | `16` |
| selected for basin | `16` |
| final basins | `3` |
| final nodes | `3` |

Why this case matters:

- 最简单、最容易人工审阅的 terrace + 单齿小分子体系
- 适合检查位点等价性、pose 去重和 basin/node 映射

### Case 2: `fcc111 + C6H6`

Directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/C6H6`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/fcc111/C6H6)

| metric | value |
| --- | ---: |
| surface atoms | `16` |
| basis primitives | `4` |
| raw pose frames | `16` |
| selected for basin | `16` |
| final basins | `13` |
| final nodes | `13` |

Why this case matters:

- 平面芳香分子在规则 terrace 上的 basin 多样性很高
- 适合检查高齿数吸附、平面分子姿态遍历、final dedup 的分辨能力

### Case 3: `cu321 + C6H6`

Directory:

- [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2/cu321/C6H6)

| metric | value |
| --- | ---: |
| surface atoms | `38` |
| basis primitives | `31` |
| sampled primitives used | `24` |
| raw pose frames | `74` |
| selected for basin | `24` |
| final basins | `14` |
| final nodes | `14` |

Why this case matters:

- 这是更接近“arbitrary slab”论点的 rough / vicinal surface 例子
- 能直接暴露 primitive 枚举、basis 压缩、pre-relax FPS budget 是否合理

### At-a-glance summary

| case | surface class | adsorbate class | main thing to inspect |
| --- | --- | --- | --- |
| `fcc111 + NH3` | low-index terrace | small monodentate | site equivalence and node canonicalization |
| `fcc111 + C6H6` | low-index terrace | flat aromatic | basin diversity on simple surface |
| `cu321 + C6H6` | rough / vicinal | flat aromatic | robustness on complex surface geometry |

## Examples and Scripts

推荐从下面这些文件开始：

- 更完整的使用说明：
  - [`examples/FULL_USAGE.md`](examples/FULL_USAGE.md)
- 端到端示例脚本：
  - [`tools/full_repo_example.py`](tools/full_repo_example.py)

与当前默认流程和 benchmark 最相关的 artifact 目录：

- controlled default workflow:
  - [`artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2`](artifacts/autoresearch/polyatomic_final_dedup_suite_default_current_v2)
- pre-relax budget sweep:
  - [`artifacts/autoresearch/polyatomic_pre_relax_budget_sweep_v1`](artifacts/autoresearch/polyatomic_pre_relax_budget_sweep_v1)
- paper-positioning reports:
  - [`artifacts/autoresearch/paper_positioning`](artifacts/autoresearch/paper_positioning)

## Validation and Artifacts

这个仓库目前同时维护两类验证：

### 1. 代码级回归

与当前 workflow 和 selection 最相关的测试包括：

- `tests/test_stage_selection.py`
- `tests/test_workflow_api.py`
- `tests/test_workflow_presets.py`
- `tests/test_fps_selector.py`

### 2. 物理与工程 artifact

当前较重要的内部结论包括：

- 低指数金属表面与主流库的一致性审计
- `mace_node_l2 + hierarchical` 作为 final basin dedup 默认值的受控支持
- fixed `k=24` 作为 pre-relax FPS 默认预算的受控支持
- 当前 cross-library positioning 的内部报告

对应的主要报告文件：

- [`competitive_positioning_report.md`](artifacts/autoresearch/paper_positioning/competitive_positioning_report.md)
- [`competitive_positioning_summary.json`](artifacts/autoresearch/paper_positioning/competitive_positioning_summary.json)
- [`crosslib_final_basin_report_multistage_default_v3_heavy_only_binding.md`](artifacts/autoresearch/paper_positioning/crosslib_final_basin_report_multistage_default_v3_heavy_only_binding.md)
- [`progress_vs_plan_20260403.md`](artifacts/autoresearch/paper_positioning/progress_vs_plan_20260403.md)

## Project Scope

当前这个项目最适合解决的问题是：

- 单吸附质、低覆盖条件下的吸附构型生成
- 任意 slab 上的吸附 primitive 发现与 pose sampling
- basin-level 去冗余与 node 导出
- 柔性吸附质在表面搜索前的构象采样串联

当前还在持续推进、但不应在 README 里夸大成“已完全解决”的部分包括：

- alloy / oxide / defect / cluster-interface 上的大规模最终 workflow benchmark
- flexible adsorbate 的自动 trigger policy 与预算自适应
- 更完整的 cross-library final-basin 对标矩阵

换句话说，这个仓库已经是一个可用的工程化吸附系综工作流，但仍处于快速迭代阶段，相关默认值与基准将继续随着实验更新。

## Acknowledgement

这个项目在 README 组织方式和问题设定上，受到了下列生态中相关工作的启发：

- `ASE`
- `AutoAdsorbate`
- `DockOnSurf`
- `MACE`

但本项目的重点不是 marked-SMILES 或固定模板放置，而是：

- arbitrary slab
- primitive-to-basin 映射
- anchor-free site-conditioned SE(3) sampling
- reaction-ready adsorption ensemble export
