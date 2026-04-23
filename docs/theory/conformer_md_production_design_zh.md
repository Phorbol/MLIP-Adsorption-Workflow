# Conformer-MD 生产设计说明（中文）

## 1. 文档目的

这份文档面向仓库维护者和生产用户，回答四个具体问题：

1. `conformer_md` 目前在代码里是如何实现的。
2. 这个 repo 里说的 `production` 到底指什么。
3. 目前已经实现了哪些 production-oriented 能力。
4. 下一步为了“最终生产级鲁棒性”还应该继续推进什么。

它不是理想化的未来设计稿，而是以当前可验证源码为准的实现说明。
对应代码主路径：

- `adsorption_ensemble/conformer_md/config.py`
- `adsorption_ensemble/conformer_md/pipeline.py`
- `adsorption_ensemble/conformer_md/xtb.py`
- `adsorption_ensemble/conformer_md/selectors.py`
- `adsorption_ensemble/conformer_md/descriptors.py`
- `adsorption_ensemble/workflows/flex_sampling.py`

英文版背景说明见：

- `docs/theory/conformer_md_selection_and_budget.md`

## 2. `production` 在本 repo 中的含义

这里的 `production` 不是“理论上最先进的构象搜索算法”，而是更实际的
工程标准：

- 用途明确：服务于 adsorption workflow 的上游构象种子生成，而不是只做孤立态最低能构象分析。
- 成本可控：下游 site/pose/basin 计算量会被构象数线性放大，因此必须有显式最终预算。
- 相似性判据稳定：结构比较尽量走 chemistry-aware 的 MACE 特征，而不是仅依赖纯几何向量。
- 可复现：多次 MD 运行必须有明确 seed policy，metadata 必须记录 resolved 配置。
- 可审计：每个阶段都要有可落盘 artifact，便于追溯“原始帧 -> 预选 -> loose -> refine -> final”全过程。
- 高层可自动使用：柔性吸附质进入 adsorption workflow 时，应能由预算器自动决定是否跑构象搜索，以及大致跑多重。

因此，这里的 production 更接近：

> “在 adsorption ensemble 生产链路里，以尽量少的用户调参，稳定地产生可审计、可复现、预算受控的构象代表集。”

## 3. 当前完整流程

当前 `conformer_md` 的实现流程如下。

### 3.1 输入读取

入口是 `read_molecule_any(...)`。

- 优先用 ASE 直接读取。
- 对 `.gjf/.com` 走 Gaussian 输入文件几何解析 fallback。

当前状态：

- 已支持 `examples/C6H14.gjf` 这种输入。
- 只解析原子与坐标。
- 暂未把 `charge / multiplicity` 贯通到后续 MD。

### 3.2 xTB GFN-FF MD

`XTBMDRunner` 会：

1. 写 `input.xyz`
2. 写 `md.inp`
3. 执行：

```bash
xtb input.xyz --gfnff --md --input md.inp --omd
```

4. 读取 `xtb.trj`

特点：

- 支持 `n_runs`
- 支持 `seed_mode`
  - `fixed`
  - `increment_per_run`
  - `hashed`
- 若第一次 MD 输入失败，会尝试一个 fallback `md_fallback.inp`
- 若返回码非零但 `xtb.trj` 可读且非空，可接受“部分成功”

### 3.3 原始帧 descriptor 提取

当前支持两类 descriptor：

- `geometry`
  - 全分子 pair-distance vector
- `mace`
  - 全局池化的 MACE invariant descriptor

原始帧 preselect 会先对全部 MD frame 提 descriptor。

### 3.4 raw preselect

`ConformerSelector._preselect(...)` 当前支持三种模式：

- `fps`
- `fps_pca_kmeans`
- `kmeans`

其中：

- `fps` 适合纯多样性优先
- `fps_pca_kmeans` 是当前默认
- `kmeans` 更偏聚类后选低能代表

当前实现里，raw preselect 已经可以利用 descriptor 提供的原始能量，而不是一律用零能量。

### 3.5 loose stage

preselected frames 会进入 `relax_loose`。

backend 可选：

- `identity`
- `mace_energy`
- `mace_relax`

区别：

- `identity` 不改结构，只给零能量
- `mace_energy` 只做打分，不改坐标
- `mace_relax` 才真正松弛坐标

### 3.6 loose filter

loose 输出经过第一轮后处理筛选。

支持策略：

- `none`
- `energy`
- `rmsd`
- `dual`

这里的 `rmsd_threshold` 在 `conformer_md` 里并不是 Kabsch 对齐后的笛卡尔 RMSD，
而是“descriptor 空间 L2 阈值”。因此代码已提供语义更准确的别名：

- `structure_metric_threshold`

### 3.7 refine stage

通过 loose filter 的结构进入 `relax_refine`，使用更严格的 `maxf/steps`。

### 3.8 final filter

refine 输出再做一次筛选，逻辑与 loose filter 同类，但可以使用独立阈值。

### 3.9 final budget thinning

如果 final filter 之后数量仍然大于 `target_final_k`，当前实现会执行显式预算裁剪：

1. 先按能量排序
2. 先保留最低能结构
3. 后续每一步选择“与已选集合最远、同时能量仍合理”的代表

这一步的目标是：

- 保留低能结构
- 同时保留结构多样性
- 控制最终 handoff 数量

### 3.10 结果输出

当前默认可输出的 artifact 包括：

- `all_frames.extxyz`
- `preselected.extxyz`
- `loose_relaxed.extxyz`
- `loose_filtered.extxyz`
- `refined.extxyz`
- `ensemble.extxyz`
- `metadata.json`
- `summary.json`
- `summary.txt`
- `stage_metrics.json`
- `stage_metrics.csv`

## 4. 已实现的 production-oriented 能力

以下能力已经不只是设计想法，而是已落在代码中。

### 4.1 显式最终预算

当前已支持：

- `target_final_k`

这意味着最终交给 adsorption workflow 的构象数不再完全由阈值偶然决定。

### 4.2 profile 化默认值

当前已实现两个 profile：

- `isolated_strict`
- `adsorption_seed_broad`

它们由 `resolve_selection_profile(...)` 解析。

### 4.3 默认走 MACE 特征比较

当 profile 解析为 production-oriented 模式时，结构比较默认会走：

- `metric_backend = "mace"`
- `descriptor.mace.dtype = "float64"`
- `descriptor.mace.enable_cueq = False`

这条路径的意图很明确：

- 特征比较优先稳定性，而不是吞吐
- batch relax 仍可单独使用更激进的吞吐配置

### 4.4 多次 MD 的 seed diversification

`seed_mode="increment_per_run"` 已经实际生效，不再默默重复相同随机种子。

### 4.5 raw preselect 能量语义修正

当前 raw preselect 已可以使用 descriptor 提供的原始能量，而不是固定零数组。

### 4.6 pair-energy-gap 去重补偿

当前已支持：

- `pair_energy_gap_ev`

这使得两种“结构很近但能量差显著”的构象可以同时保留，而不是被机械去重掉。

### 4.7 adsorption 高层自动预算

`plan_flex_sampling_budget(...)` 已能根据吸附质和表面复杂度决定：

- 是否启用 conformer search
- `md_time_ps`
- `md_runs`
- `preselect_k`
- `target_final_k`
- `selection_profile`
- `fps_rounds`
- `fps_round_size`

因此 conformer search 已经接入高层 adsorption workflow，而不是一个孤立玩具模块。

### 4.8 `head / device` 语义已从“隐式”变成“显式”

这是本轮实际落地、且对生产审计非常重要的一项改动。

当前 `conformer_md` 的 MACE inference 已具备以下语义：

- `head_name`
  - 若用户显式传入非空、且不等于 `Default` 的 head，则优先使用该 head
  - 不再静默退回 model 自带默认 head
  - metadata 会记录：
    - `head_name`
    - `available_heads`

- `device`
  - metadata 会区分：
    - `requested_device`
    - `runtime_device`
  - 若请求 `cuda`，但运行环境里 `torch.cuda.is_available()` 为 `False`
    - 推理阶段会自动退回 `cpu`
    - metadata 会保留“本来想跑在哪”和“这次实际上跑在哪”

这件事之所以重要，是因为过去最容易出现两类不可审计行为：

- 用户明明传了 `--mace-head-name omol`，结果实际还是跑了 model 默认 head
- 用户明明传了 `device=cuda`，结果因为当前 session 没有 GPU，直接在 `torch.load(...)` 处崩掉

现在这两类问题都已经被显式化处理：

- head 由用户配置优先决定
- device 的降级会被记录，而不是隐藏

## 5. 当前仍未完全闭环的点

以下是当前必须明确承认的“未完全 production-safe”部分。

### 5.1 `charge / spin` 尚未贯通

当前：

- `gjf` 输入可以被读取
- 但 `charge / multiplicity` 没有继续传入 xTB MD

这意味着：

- 中性闭壳层小分子没有问题
- 离子、自由基、开壳层体系仍不能称为 production-safe

这项能力后续仍需补齐，但不在当前这轮改造范围内。

### 5.2 energy 语义需要显式化

MACE 推理天然更容易给出 per-atom energy。
而 conformer ranking 的用户心智模型通常是 total energy。

当前改造方向是：

- 用 metadata 明确记录当前 energy 语义
- 在 conformer 筛选中默认使用 total-energy 语义
- 允许用户显式切回 per-atom 语义用于兼容或诊断

### 5.3 一部分重要参数尚未完全暴露到 CLI

配置层已经存在的能力，并不等于用户入口层已经完整暴露。

当前应优先面向 CLI 暴露的参数包括：

- `pair_energy_gap_ev`
- `energy_semantics`
- `loose_filter / final_filter`
- stage-specific window / metric threshold

### 5.4 GPU 不可用时，吞吐档位与语义档位要分开看

当前 `conformer_md` inference 已支持：

- 请求 `cuda`
- 实际退回 `cpu`
- metadata 审计

但这并不意味着：

- `cpu fallback` 与真正的 `cuda` 生产吞吐处在同一性能档位

更准确的理解应当是：

- 语义档位：
  - 产物仍可用于方法学验证、artifact 追溯、case 审查
- 吞吐档位：
  - CPU 路径不能被误当成与 CUDA 同级的正式批量生产路径

因此对 README 和 suite 脚本的推荐表述应该始终区分：

- `requested_device`
- `runtime_device`
- 以及是否只是一次 CPU 降级执行

## 6. 当前两个 profile 的实际语义

### 6.1 `isolated_strict`

意图：

- 做更紧的孤立态构象筛选

当前默认解析为：

- `target_final_k = 12`
- `preselect_k = max(64, 4 * target_final_k)`
- `energy_window_ev = 0.20`
- `pair_energy_gap_ev = 0.02`
- `metric_backend = mace`
- MACE compare 使用 `float64 + cueq off`

### 6.2 `adsorption_seed_broad`

意图：

- 作为 adsorption workflow 的上游种子集合

当前默认解析为：

- `target_final_k = 8`
- `preselect_k = max(96, 6 * target_final_k)`
- `energy_window_ev = 0.60`
- `pair_energy_gap_ev = 0.01`
- `metric_backend = mace`
- MACE compare 使用 `float64 + cueq off`

重要说明：

- `adsorption_seed_broad` 不是“最后保留更多构象”
- 它的含义是“前面放宽保留，最后 handoff 预算更紧”

## 7. 参数 -> 算法阶段 -> 输出 artifact 对应表

下表只列 production 里最重要的参数，而不是 dataclass 中的每一个字段。

| 参数 | 所属阶段 | 作用 | 直接影响的输出 / artifact |
| --- | --- | --- | --- |
| `temperature_k` | MD | 控制 xTB MD 温度 | `md_run_*/md.inp`, `xtb.trj`, `metadata.json` |
| `time_ps` | MD | 控制每条 MD 长度 | `xtb.trj`, `n_raw_frames`, `stage_metrics.json` |
| `dump_fs` | MD | 控制采样间隔 | `xtb.trj`, `n_raw_frames` |
| `n_runs` | MD | 控制独立 MD 条数 | `md_run_*`, `md_runs`, `per_run_seeds` |
| `seed_mode` | MD | 控制多条 MD 的种子策略 | `metadata.json` 中的 `md_runs` 与 `per_run_seeds` |
| `descriptor.backend` | raw descriptor | 决定 raw frame 特征来源 | `descriptor_inference`, raw preselect 行为 |
| `selection.mode` | raw preselect | 决定 `fps / fps_pca_kmeans / kmeans` | `preselected.extxyz`, `preselected_ids_in_raw` |
| `preselect_k` | raw preselect | 控制 preselect 上限 | `n_preselected`, `preselected.extxyz` |
| `metric_backend` | loose/final compare | 决定 post-preselect 用什么特征做去重 | `stage_metrics.diversity`, final 保留集合 |
| `structure_metric_threshold` | loose/final compare | 控制 descriptor 距离去重阈值 | `n_after_loose_filter`, `n_selected` |
| `energy_window_ev` | loose/final compare | 控制全局能窗 | `n_after_loose_filter`, `n_selected`, `summary.json` |
| `pair_energy_gap_ev` | loose/final compare | 允许近似结构因能量差而共存 | `n_selected`, `ensemble.extxyz` |
| `loose_filter` | loose filter | 控制 loose 阶段筛选策略 | `loose_filtered.extxyz`, `n_after_loose_filter` |
| `final_filter` | final filter | 控制 final 阶段筛选策略 | `ensemble.extxyz`, `n_selected` |
| `target_final_k` | final budget | 控制最终 handoff 上限 | `n_selected`, `ensemble.extxyz`, `summary.json` |
| `relax.backend` | loose/refine | 决定只打分还是实际松弛 | `loose_relaxed.extxyz`, `refined.extxyz`, `relax_inference` |
| `loose.maxf / loose.steps` | loose relax | 控制 loose 阶段松弛强度 | `relax_loose/*`, `stage_metrics.relax_shift.pre_to_loose` |
| `refine.maxf / refine.steps` | refine relax | 控制 refine 阶段松弛强度 | `relax_refine/*`, `stage_metrics.relax_shift.loose_filtered_to_refined` |
| `use_total_energy` | energy semantics | 决定排序和汇总按 total 还是 per-atom energy | `summary.json`, `summary.txt`, `metadata.json` |
| `selection_profile` | profile resolver | 决定一组生产默认值 | `metadata.json`, `resolved_*` 字段 |
| `output.save_all_frames` | I/O | 控制是否写完整中间帧 | `all_frames.extxyz` 等中间产物 |

## 8. 当前对用户暴露的参数层级

### 8.1 Python 配置层

当前最完整的参数面仍然是 `ConformerMDSamplerConfig`。

适用场景：

- 你要细调某个 case
- 你要把 conformer search 嵌到 adsorption workflow
- 你要控制 profile 之外的细节

### 8.2 CLI 层

CLI 已适合常规生产调用，但仍是“高价值参数优先暴露”，不是 config 逐字段镜像。

适用场景：

- 单分子或单 case 生产运行
- 用户想直接从 `gjf` 或 `xyz` 输入开始
- 想在 shell 里快速切换 profile、budget 和后处理阈值

### 8.3 adsorption 高层预算层

高层 workflow 通过 `plan_flex_sampling_budget(...)` 自动决定是否跑 conformer search，
以及大致投入多少预算。

适用场景：

- 用户主要目标是最终 adsorption ensemble
- 不希望单独手工调 conformer 参数

## 9. 面向最终生产的鲁棒性推进顺序

当前建议的推进顺序如下。

### P0

- 明确并落地 total-energy 语义
- 让 metadata 记录 resolved profile、energy semantics、per-run seeds
- 把关键筛选参数暴露到 CLI

### P1

- 扩展 stage-specific 参数的 CLI 可控性
- 在高层 workflow summary 中更显式地记录 conformer stage 的 resolved 配置
- 增加“参数是否被 profile 覆盖”的审计字段

### P2

- 贯通 `charge / spin`
- 针对更复杂、开壳层、离子体系做专门 case suite

## 10. 实用结论

一句话总结当前状态：

> 我们已经有了一条“可接入 adsorption workflow、默认走 MACE 比较、带显式最终预算、带 profile、可审计”的 conformer-MD 生产链路。

但如果从“最终生产级”来审视，仍需继续收紧：

- energy 语义
- 用户入口暴露
- metadata 完整度
- 以及后续的 `charge / spin` 贯通

而在已经完成的部分里，最值得强调的是：

- `head_name` 不再被静默覆盖
- `requested_device / runtime_device` 已进入 metadata
- 这使得生产 artifact 至少在“我到底用什么 head、到底跑在什么设备上”这两个问题上不再含糊

在不讨论电子态的前提下，本轮最值得继续推进的就是：

- 让 total-energy 语义真正生效
- 让运行记录更完整
- 让已经实现的关键参数真正对用户可用
