# 柔性吸附生产案例设计说明（中文）

## 1. 文档目的

这份文档记录两件事：

1. 我们如何参考 DockOnSurf 一类工作的设计原则，构造本 repo 的柔性吸附测试。
2. 为什么当前默认 suite 选择了几组具体 case，以及这些 case 分别在检验什么。

它不是对 DockOnSurf 的复刻说明，而是把外部方法学启发映射到本仓库现有实现上。

## 2. 外部方法学启发

DockOnSurf 的核心启发不是“某一个具体输入文件格式”，而是把柔性吸附搜索拆成若干
物理上有意义的维度：

- isolated conformers
- anchoring points
- surface sites
- orientations
- collision control
- optional proton dissociation

对应到本仓库，最直接的映射是：

| DockOnSurf-style 维度 | 本仓库中的对应模块 |
| --- | --- |
| conformers | `conformer_md` |
| anchoring points | 不要求用户显式给 anchor；由 `primitive + local frame + pose` 隐式覆盖 |
| sites | `surface + site` primitive/basis 枚举 |
| orientations | `pose sampler` 的 `n_rotations / n_azimuth / n_shifts` |
| collision control | `adaptive_height_fallback` |
| proton-transfer / dissociation sensitivity | 当前主要体现在 basin anomaly / binding 诊断；不是前端枚举主轴 |

因此，我们当前的 suite 设计原则是：

- 不只测“一个稳定吸附构型能不能找到”
- 而要测“构象、局部表面 motif、姿态采样、碰撞避免、后端 basin 去重”是否能连成闭环

## 3. 本仓库采用的生产语义

对柔性吸附质，我们当前采用的生产语义是：

- 先由 `plan_flex_sampling_budget(...)` 判断是否需要 isolated conformer search
- 若需要，则在 adsorption workflow 前显式跑 `conformer_md`
- 采用 `selection_profile="adsorption_seed_broad"`
- 下游 adsorption 继续走：
  - `anchor_free`
  - broad pose coverage
  - `adaptive_height_fallback=True`
  - basin dedup 使用 `mace_node_l2`
  - basin MACE compare 使用 `float64 + cueq off`

这一套语义的重点不是“把 isolated conformers 筛得极窄”，而是：

> 给 adsorption workflow 一个预算受控、但不至于过早丢掉 surface-competent conformer 的代表集。

## 4. 默认 suite 的 case 设计

当前默认 suite 放三类柔性案例。

### 4.1 `TiO2_110 + CH3COOH`

它主要检验：

- O-rich adsorbate 在 oxide surface 上的多锚定竞争
- 构象与吸附取向的联动
- 碰撞避免与 basin 过滤对极性体系是否稳健

选择这个 case 的原因是：

- carboxylic acid 是典型的表面吸附强官能团
- oxide surface 能放大“局部配位环境”和“姿态选择”的影响
- 即使当前 workflow 不显式前端枚举质子解离，这个 case 仍然能有效压力测试多 O 吸附几何

### 4.2 `CuNi_fcc111_alloy + CH3CONH2`

它主要检验：

- amide 这类多杂原子、小到中等柔性分子的构象搜索是否真正进入 adsorption 主链
- 合金表面的局部成分差异是否会放大 basin 多样性
- `head/device` 语义和 MACE descriptor 比较在 heterogeneous local environment 下是否可审计

这个 case 比纯金属 + 单官能团小分子更接近“真实催化表面上中等复杂有机分子”的工程情境。

### 4.3 `Pt_fcc111 + CH3CH2OH`

它主要检验：

- 小分子、单主要转动自由度体系上，conformer stage 是否仍然会被合理触发
- isolated conformer 与 adsorption orientation 是否会共同影响 basin 分布
- 在比 `CNB` 更便宜的 case 上，整条生产链能否稳定跑完

它不是“最复杂”的柔性 case，但很适合作为 production smoke-to-real bridge。

## 5. 可选重型案例

除默认 suite 外，我们还建议保留一个更重的芳香多官能团案例作为扩展示例：

- `Pt_fcc211 + oCNB`

它检验：

- 芳香环平躺/倾斜吸附与官能团导向吸附的竞争
- stepped surface 上 site 丰富度对 basin 分布的放大
- 更大的构象与姿态组合空间下，去重是否仍然稳健

这个 case 很适合作为论文展示和高压测试，但在纯 CPU 路径下成本明显更高，因此当前不放在默认 suite 中。

## 6. 运行入口

当前推荐入口是：

```bash
python tools/run_flexible_adsorption_suite.py \
  --out-root artifacts/flexible_adsorption_suite \
  --mace-model-path /root/.cache/mace/mace-mh-1.model \
  --mace-device cuda
```

默认会写出：

- `runtime_manifest.json`
- `flexible_adsorption_suite_summary.json`
- `flexible_adsorption_suite_summary.csv`
- `flexible_adsorption_suite_summary.md`
- 每个 case 自己的 `workflow_summary.json`
- 若启用了 conformer search，则还有 `conformer_metadata.json`

## 7. 关于 GPU 不可用时的语义

当前 suite 明确区分：

- `mace_device_requested`
- `mace_device_effective`

如果请求 `cuda` 但当前没有可用 GPU：

- `conformer_md` 的 MACE inference 会自动回退到 `cpu`
- adsorption workflow 的 relax backend 会按 `cpu` 构造
- summary 与 runtime manifest 会记录这是一次 CPU 降级执行

这很重要，因为：

- 产物仍然可以用于功能验证和方法学审计
- 但它不应被误认为与 CUDA 生产吞吐属于同一性能档位

## 8. 当前 suite 想回答的问题

默认 suite 并不打算回答“哪个表面-分子组合的吸附能最准确”，而是回答：

1. 柔性吸附质是否会被高层预算器稳定识别并触发 conformer search。
2. conformer -> pose -> basin 的链路是否能在不同表面类别上稳定闭环。
3. `adaptive_height_fallback` 是否能减少初猜碰撞导致的大量无意义异常。
4. basin dedup 在多姿态、多局域表面环境的情况下是否仍然可控。
5. 运行记录是否足够清楚地区分：
   - requested head / effective head
   - requested device / runtime device
   - isolated conformer budget / downstream adsorption budget

一句话总结：

> 这组 suite 是用来检验“柔性吸附的生产链是否真的成熟”，而不是只检验单个算例能不能给出若干 basin。
