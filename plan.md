# 第一篇方法旗舰稿：从 Arbitrary Slab 到 Reaction-Network-Ready Adsorption Ensembles（计划稿）

## 0. 目标与边界（写在纸面上的“不可变约束”）

### 0.1 一句话问题定义（主文反复复用）

输入任意 slab 与任意 adsorbate（低覆盖、单吸附质、真空/近真空条件），输出一个面向下游自动反应探索的、可规范化与去冗余的低能吸附态系综（reaction-network-ready adsorption ensemble）。

### 0.2 输出对象与术语不变量

- primitive：由 slab 几何/拓扑导出的初始原语（不是最终 adsorption site）
- pose：在 primitive 条件下的 SE(3) 放置提案（SO(3) 旋转 + 切平面位移 + 高度）
- basin：经过松弛后落入的吸附能谷（最终输出单位）
- ensemble：在给定能窗内的去重 basins 集合
- node：可被图论反应搜索直接消费的吸附态节点对象（basin 的规范化表示）

node 输出最小字段集（第一篇硬约束）：
- basin_id / node_id（稳定可复现的身份）
- canonicalized atom ordering（吸附质内部可规范化的原子顺序）
- adsorbate internal graph（分子内部键图）
- adsorbate–surface binding graph（吸附成键图 + binding atom set）
- denticity / motif label（单齿/双齿/多齿等）
- relative energy（同体系内相对能）
- provenance（primitive ancestry / 支撑该 basin 的 pose 计数与来源）

### 0.3 明确不做什么（第一篇必须收紧）

- 不做 coverage / adsorbate–adsorbate interaction / net distribution（可留作后续）
- 不把全文重心写成“找到单个 global minimum 的工具”
- 不把 generative / agentic 当主 baseline（放在 related work 与 discussion，主比较走可控 baseline）

### 0.4 本文定位（必须写进摘要与 Fig.1）

- 这是一篇 node paper：解决 automated surface reaction exploration 的 state-generation bottleneck（node completeness）。
- CCQN 在本文中只作为 fixed downstream probe：不 benchmark CCQN 本身，不和其他 TS 算法正面打擂台；只用它读出“同一 downstream engine 下，不同 seed-state policy 导致不同 verified reaction discovery”。

---

## 1. 主线叙事与贡献点（主文结构骨架）

### 1.1 主线叙事（从“拼接流水线”升级为映射问题）

我们把任意 slab 转成一组 exposed-surface site primitives；用 chemistry-aware primitive basis 定义 adsorption-equivalent minimal basis；在 primitives 上进行 anchor-free、site-conditioned 的 SE(3) pose sampling，并通过分层 MLIP 松弛将多样初猜映射到少量非冗余吸附盆地（basins）；最终把 basins 规范化为可被下游图论反应搜索直接消费的 node objects，输出 reaction-network-ready adsorption ensemble。

核心升级：primitive-to-basin mapper（多对多映射）+ node canonicalization，而非“初猜生成器”。

### 1.2 四个贡献点（建议以四条 bullet 作为摘要贡献）

1) Exposed-surface primitives on arbitrary slabs：从外表面可达性与 surface graph 构造 1c/2c/3c/4c primitives（在规整表面退化为经典 site motifs）  
2) Chemistry-aware adsorption-equivalent minimal basis：用 MACE-derived atom features（+少量显式几何通道）定义 primitive embeddings，降低 symmetry-only / topology-only 的错误合并（以 false-merge rate 量化）  
3) Anchor-free, site-conditioned basin generation：在 primitive 局部参考系上进行 uniform SO(3) + 切平面 shifts + 自适应高度求解，靠松弛发现 realized binding pattern，形成 primitive-to-basin 多对多映射  
4) Reaction-network-ready node output：将 basins 规范化为 node objects（internal graph + binding graph + canonical ID + provenance），可被下游 CCQN 直接消费，形成可控的 seed-state policy

---

## 2. 方法设计：模块化到“可直接动手实现/迭代”的工程结构

下面每个模块都给出：输入/输出、关键不变量（必须硬校验）、最小可行版本（v1）、以及对 benchmark 的对应验证点。

### 模块 M0：Surface preprocessing（slab -> SurfaceContext）

- 输入：slab（Atoms）
- 输出：SurfaceContext（surface atoms、surface graph、grid meta、可选 SDF、可选 MACE node feats）

关键不变量（硬校验）：
- 真空侧方向一致：normal 必须指向 exterior（可通过 exterior flood-fill / 最大空腔连通域来定义）
- surface atom 识别可解释：precision/recall 可在 B0 上量化

v1 建议实现：
- occupancy grid + exterior flood-fill -> exposed atoms
- exposed atoms 子图 -> exposed-surface graph（cutoff 或 Voronoi 邻接）
- 可选：提取 MACE invariant node features（先不追 edge latent）

### 模块 M1：Primitive enumeration（SurfaceContext -> raw primitives）

- 输入：surface graph + surface atoms
- 输出：1c/2c/3c/4c primitives（不命名 atop/bridge/hollow）

关键不变量：
- 枚举是“拓扑可复现”的：同一 slab + cutoff 规则应给出稳定的 primitive 集
- 对称等价不强行合并（交给 M2 的 embedding/basis）

v1 枚举规则：
- 1c：每个 exposed atom
- 2c：graph edges
- 3c：3-cycle / 3-clique
- 4c：4-cycle 或紧凑 4-node motif（需要定义“紧凑”的一致规则）

### 模块 M2：Site embedding + adsorption-equivalent clustering（raw -> minimal basis）

- 输入：raw primitives + per-atom features（MACE node feats）+ 少量几何辅助量
- 输出：basis primitives（每类 representative + basis_id）

关键不变量：
- 压缩不应引入“低能 basin 漏检”的系统偏置（用 PrimitiveCoverage 与 EnsembleRecall 在 B1/B2 验证）

v1 embedding（强约束版，避免过拟合工程复杂度）：
- primitive pooling：mean/max over involved atom node feats
- concat：pool + one_hot(kind) + geom_aux_features
- MLP -> z_site
- 聚类：先按 topo_hash 分桶，再桶内用 embedding 距离 + 几何容差聚类

输出记录：
- raw_count、basis_count、compression_ratio
- cluster purity（可用“同一 reference basin 命中是否集中”作为 proxy）

### 模块 M3：Local frame construction（primitive -> (c,n,t1,t2)）

关键不变量：
- n 方向固定到真空侧
- t1/t2 与 n 正交、右手系、数值稳定（对称退化时定义 tie-break）

v1：
- center：involved atoms 质心
- normal：优先 SDF 梯度；fallback 局部平面拟合
- t1/t2：局部 patch covariance 主方向 + Gram-Schmidt

### 模块 M4：Anchor-free site-conditioned SE(3) pose sampler

采样变量：
- R ∈ SO(3)
- (u,v)：tangent-plane shift
- h：沿 normal 的高度（clash-controlled）

关键不变量：
- 采样可复现：显式 RNG seed；uniform SO(3) 采样用标准算法（如 quaternion method）
- 初始化不产生大量硬碰撞（用 valid pose rate 与 pre-relax 失败率度量）

v1 参数建议（可写入 config，后续 ablation）：
- n_rot：按 adsorbate 尺寸分档（64/128/192）
- n_shift：3/5/9
- taus：0.95/1.05/1.15（共价半径缩放）

高度求解 v1：
- 目标：min_ij d_ij / (r_cov_i + r_cov_j) ≈ tau

---

## 3. 采样阶段性能优化计划（在不改变物理/功能约束前提下最大化吞吐与 scaling）

本节目标：在保持当前“高度求解 + 碰撞判定 + 去重”语义一致的前提下，把采样阶段做到：
- 单进程：CPU 利用率尽可能高（避免 Python 循环与小函数开销主导）
- 单核速度：尽可能快（vectorize + batch MIC）
- 多核 scaling：按 primitives / rotations / slabs 可并行，且可复现
- 物理一致性：PBC/MIC 定义与当前一致；结果集（pose 分布、过滤/去重行为）不发生系统性偏移

### 3.1 当前瓶颈与性能不变量

已知热点（以实际 cProfile 为准）：
- 高度求解 `_solve_height → _check_height_constraints → (距离计算)` 占采样绝大多数时间
- ASE MIC 距离计算路径 `get_distances/find_mic/general_find_mic/minkowski_reduce` 是主耗时

性能不变量（必须硬保证）：
- MIC/PBC 语义一致：对 pbc 轴，采用 minimum image；对非 pbc 轴不 wrap
- 阈值一致：仍使用共价半径与 tau 规则（含 site_contact_tolerance 等额外容差）
- 数值稳定：对几何退化/极端 cell 不崩溃；有可解释的 fallback
- 可复现：同 seed 同输入得到相同 pose 列表（顺序允许在并行模式下通过稳定排序恢复）

### 3.2 Profiling 与资源监控（CPU/RSS/峰值内存）

1) 基线采集（必须每次优化前后跑同一组缩小参数）
- cProfile：记录 top cumulative 与关键函数调用次数
- 细粒度计时：记录 slab_features / sampling / height_checks / clash / prune 时间占比

2) 资源监控（低开销、跨平台优先）
- 默认实现：`tracemalloc` 记录 Python 堆峰值；`resource`/`/proc/self/statm`（WSL）记录 RSS；`time.process_time` 记录 CPU time
- 如果环境有 `psutil`：可选采样 CPU%/RSS，并输出为 CSV（采样周期 0.2–1.0s，可配置）
- 输出：每个 run 生成 `perf_monitor.csv` + `perf_summary.json`（包含峰值 RSS、峰值 tracemalloc、CPU time/Wall time）

验收标准：
- profiling 文件可复现生成；监控开销 < 3%

### 3.3 距离计算内核重构路线图（优先级从“低风险高收益”到“高收益高工程量”）

#### 路线 A（优先做）：纯 NumPy batch + 一次 MIC 归约（避免重复 Minkowski）

核心思想：
- 不再逐对调用 `Atoms.get_distance/get_distances`
- 对每次高度检查，直接在 NumPy 中构造 ads–surface 的位移向量批（shape ≈ (Na, Ns, 3)）
- 将 MIC 操作批量化：优先使用 `ase.geometry.find_mic` 对展平后的 vectors 一次处理（保证与 ASE 语义一致），并缓存 cell 的 Minkowski 预处理结果（如可行）

适用场景：
- 通用 cell（含非正交）；需要严格与 ASE 对齐时

预期收益：
- 显著减少 Python 调用开销；把 MIC 归约从“多次调用”变为“少量大批次调用”

#### 路线 B：正交/近正交 cell 的快速 MIC（fractional rounding）

核心思想：
- 对常见 slab（正交 + 真空轴），在 fractional 坐标下做 `delta -= round(delta)`（仅对 pbc 轴），再乘回 cell 得到最短向量
- 该路径在正交 cell 上与 minimum image 等价，且可完全 NumPy 向量化

正确性策略：
- 自动检测 cell 是否正交（angles≈90° 且 off-diagonal 小于阈值）
- 不满足条件时回退路线 A（ASE find_mic batch）

预期收益：
- 大幅降低 MIC 成本，提高 CPU 计算密度与缓存友好性

#### 路线 C：大 surface 的邻域裁剪（SciPy KDTree / 自实现邻域网格）

核心思想：
- 当 Ns 很大（例如 > 5e4），Na×Ns 批量距离会吃内存/带宽
- 用 KDTree / cell-list 先筛候选邻居，再对候选做精确 MIC 距离

实现约束：
- 只对“正交 cell + pbc 轴明确”的情形启用 KDTree（更容易保证正确性）
- triclinic 或不确定时回退路线 A

预期收益：
- Ns 极大时显著改善 scaling；避免 Na×Ns 的 O(N) 全扫

#### 路线 E：候选邻域裁剪的“物理安全版”（保守上界 + 精确 MIC 复核）

动机：
- 当前纯 NumPy batch MIC 在 Ns 很大时仍是 O(Na×Ns) 全扫，最终会被内存带宽限制
- 想再快 1 个数量级，必须把每次高度检查的候选 surface 原子数从 Ns 降到 K（K≪Ns）

基本原则（不改变物理意义）：
- 任何裁剪都必须是“保守的”：宁可多保留一些候选，也不能漏掉真正的最近邻/碰撞对
- 裁剪只做粗筛，最终距离仍用精确 MIC 内核计算（正交用 fractional rounding，非正交用 Minkowski-reduced MIC）

可选实现：
1) **正交/近正交 slab（推荐先做）**
   - 使用 2D cell-list（xy 网格）或 SciPy cKDTree（xy 维度）做候选
   - 候选半径使用上界：`r_cut = tau*(r_ads_max + r_surf_max) + padding`，并考虑 `site_contact_tolerance` 等额外容差
2) **非正交 slab**
   - 在 Minkowski-reduced cell 的 fractional 坐标里对 pbc 轴 wrap 后做 2D 网格
   - 对 triclinic 的 xy shear，使用“包围盒上界”来构建保守候选集合，再用精确 MIC 复核

验收与回归：
- 与全扫版本在同一 seed 下输出统计一致（pose_n、height 分布、tilt 分布），并通过随机 spot-check 的距离/碰撞一致性
- 在 Ns≥O(1e5) 的 slab 上 wall time 随 Ns 的增长显著优于线性（接近常数或弱增长，受候选密度控制）

#### 路线 F：GPU 距离核（可选，作为极端大体系的吞吐方案）

动机：
- 对 Na×Ns 级别的大矩阵距离，GPU 更擅长吞吐；尤其当 Ns 很大而 CPU 带宽成为瓶颈时

约束：
- 不引入强依赖：优先复用已存在的 torch（项目已在用），JAX 仅作为可选项
- 物理一致性必须可证明：正交 cell 的 fractional rounding 在 GPU 上最容易严格对齐；非正交优先走 CPU Minkowski 版本或做严格等价实现后再启用

实现方向：
- torch CPU/GPU 实现 `dist2 = ||(wrap(frac_surf - frac_ads) * lengths)||^2`（只对 pbc 轴 wrap）
- clash/min_ratio 等判据用 `torch.min`/`torch.any` 做归约，避免 Python 循环
- 需要显式控制 dtype（float64/float32）并做误差评估：边界处（接近阈值）可能需要 float64 或 safety margin

验收：
- 与 CPU 版本在误差容忍范围内一致；对阈值边界给出 deterministic tie-break（例如 epsilon buffer）
- 在 GPU 可用时对大 Ns 显著加速；GPU 不可用时自动回退 CPU

#### 路线 G：并行化把 CPU 吃满（在单核/单机内核稳定后）

目标：
- “CPU 持续高占用直到任务结束”：当 sampling 内核足够快后，把整个 sweep 在多核上并行

策略：
- 粗粒度：slab×molecule×primitive 分块，多进程并行（进程间只传 NumPy 数组与轻量元数据）
- 可复现：块级 seed 派生 + 汇总后稳定排序
- I/O：每块写入独立子目录，最后合并 summary，避免锁争用

#### 路线 D：JIT/NUMBA/JAX（可选）

原则：
- 不假设环境一定有 numba/jax；必须做可选依赖与纯 NumPy fallback
- 优先 JIT 纯数组 kernel（距离与阈值比较），不要 JIT Atoms/IO

验收标准：
- 有无 JIT 的结果一致（允许 1e-12 量级数值差）

### 3.4 并行化与 scaling（在单核优化完成后再做）

并行策略优先级：
1) primitives-level 并行（最独立）：每个 primitive 一组姿态生成与过滤
2) rotation/azimuth block 并行：适合大采样预算

实现要点：
- 并行后仍保持可复现：每个任务块独立 RNG seed（由全局 seed + block_id 生成），结果汇总后按 (height, primitive_id, rotation_id, azimuth_id, shift_id) 稳定排序
- 避免大量 Atoms 复制：并行块只传 NumPy positions/cell/pbc 与必要元数据

验收标准：
- 2/4/8 进程下 wall time 接近线性下降（受内存带宽限制时允许亚线性）
- 输出 pose 集合与单进程一致（或在允许误差内一致）

### 3.5 验证与回归（物理一致性）

必须做的回归：
- MIC 一致性：随机采样一批 (ads_i, surf_j) 对，比较新 kernel 与 ASE `get_distance(mic=True)` 的距离误差分布（max/percentile）
- 采样输出一致性：同一 slab+mol+seed 下，pose_n、height 分布、tilt 分布与旧实现对齐（允许极小浮点差导致的边界位点差异，但不能系统偏移）
- 极端 case：非正交 cell、pbc=partial、超大 surface_n、Na>12 的分子
- 可用一维搜索或解析近似（保证速度与稳健）

### 模块 M5：Hierarchical relax（poses -> relaxed candidates）

关键不变量：
- 同一套 relax backend 被所有方法与 baseline 共享（保证公平）
- 失败结构必须可追踪（原因分类：碰撞、发散、脱附、解离等）

v1 staged：
1) pre-relax：短、带约束（仅去 clash）
2) MLIP loose relax：fmax ~ 0.10 eV/Å
3) MLIP tight relax：仅 top-M promotion，fmax ~ 0.03 eV/Å

### 模块 M6：Anomaly filter + Dedup + Basin construction

关键不变量：
- “输出是 Basin，不是 RelaxedStructure”
- 去重是两层：contact graph hash -> RMSD+energy clustering

异常过滤（v1 最少集合）：
- desorption（adsorbate 离开表面阈值）
- severe reconstruction（表面原子位移/配位突变阈值）
- unintended dissociation（键断裂/分子碎裂）
- subsurface burial（吸附质进入表面下方）

### 模块 M7：Node canonicalization（basin -> reaction-searchable node object）

目标：把几何结构翻译成稳定、可对齐、可复现的“节点对象”，作为 downstream 图论反应搜索的唯一入口。

关键不变量：
- node identity 稳定：同一体系同一 basin 在数值扰动/不同 primitive provenance 下应收敛到同一 node_id（允许极小浮点差，但不允许系统性分裂）
- basin 去重与 node 去重一致：同一个 node 不能在输出里以多个 basin/object 重复出现
- node 输出信息完整：internal graph / binding graph / denticity / energy / provenance 必须齐全，且可反向审计

v1 实现建议：
- internal graph：基于已知 adsorbate connectivity（或以可控阈值从几何恢复），输出 bond list
- binding graph：以几何阈值 + 元素/共价半径规则定义 adsorbate–surface 接触边，输出 binding atom set 与 surface atom set
- canonicalization：对 adsorbate internal graph 做 canonical ordering（例如基于图同构的 canonical label），并在 node_id 中纳入 binding graph 信息与 slab/primitive ancestry 哈希

### 模块 M8：柔性吸附质构象 → 吸附 workflow 一体化（conformer ensemble -> adsorption ensemble）

目标：支持“柔性吸附质先做 conformer MD 得到一批构象（可视作 gas-phase basins/representatives），再对每个构象走完整吸附 workflow（pose→relax→basin→node）”，并将 provenance 完整写入 summary/report，形成可复现的多构象吸附系综。

#### M8.1 输入与来源（v1 只做 provided conformers.extxyz）

1) **provided conformers（v1 唯一入口）**  
   - 输入：`conformers.extxyz`（每帧是一个 adsorbate Atoms，frames 可带 `info`：charge/spin/label）  
   - 优点：不强制 xtb、可与外部 RDKit/CREST/DFT 产生的构象无缝对接  
   - 约束：每帧必须是“纯吸附质”（无 slab），原子数一致；可选：允许多组分（后续）

2) **conformer_md 内置生成（v2，后续）**  
   - 将 `ConformerMDSampler` 的 `ensemble.extxyz` 作为 M8 的输入之一键衔接（需要 xtb 环境与额外 CLI/API 设计）

3) **hybrid（v3，后续）**  
   - 在吸附前对 provided conformers 做一次气相 quick relax/去重（需要定义物理不变量与审计字段）

#### M8.2 新的“库级”API 设计（不破坏现有 sweep）

原则：不复制现有 `run_pose_sampling_sweep` 的大量逻辑，只在 adsorbate 来源处插入“conformer 维度”，其余流程复用既有模块（surface/site/pose/postprocess/ensemble）。

- 新增（建议）API：`run_pose_sampling_sweep_with_adsorbates(...)` 或在现有 `run_pose_sampling_sweep(...)` 中新增可选参数
  - `adsorbates: dict[str, list[Atoms]] | None`  
    - key：molecule name（例如 "C2H5OH"）  
    - value：一组 conformer frames（顺序稳定，提供 `conformer_id`）
  - 或更 IO 友好：`adsorbate_extxyz_map: dict[str, Path] | None`（内部读取）
  - 与现有参数的关系：当提供 adsorbates 时，跳过 `list_supported_molecules()` 的默认枚举

- PoseSweepConfig 扩展（建议集中在一个小子配置里，避免字段爆炸）：
  - `adsorbate_source: str = "ase"`（"ase" | "extxyz" | "conformer_md"）
  - `adsorbate_extxyz: str = ""`（单分子输入）或 `adsorbate_extxyz_dir: str = ""`（多分子映射）
  - `max_conformers_per_molecule: int = 32`（控制 combinatorics）
  - `conformer_selection: str = "topk_energy_or_fps"`（后续扩展）
  - `conformer_id_field: str = "conformer_id"`（写入输出的字段名）

#### M8.3 输出目录与输出契约（避免破坏现有下游脚本）

必须保持：
- 现有 per-case 输出文件名不变：`pose_pool.extxyz`、`pose_final.extxyz`、`basins.extxyz`、`nodes.json`、`pose_sampling_report.md`、summary csv/json。

新增字段（兼容旧消费者）：
- summary row 新增：
  - `conformer_id`（int）
  - `conformer_label`（str，可选，来自 extxyz `info`）
  - `conformer_source`（"extxyz" | "conformer_md" | "ase"）
  - `conformer_energy_ev`（可选；如果来源能提供气相能）
  - `adsorbate_hash`（对 adsorbate 原子序列与坐标做稳定 hash，用于审计）

输出目录策略（v1 推荐：**不嵌套目录、用后缀编码 conformer_id**）：
- 原 `<run_dir>/<slab>/<mol>/...` 改为 `<run_dir>/<slab>/<mol>__conf0007/...`
- 同时在 report 里按 molecule 聚合展示（避免目录变长影响可读性）

#### M8.4 关键不变量与硬校验（避免“看起来能跑”但不可复现）

- **构象输入一致性**：每个 molecule 的 conformer frames 必须：
  - 原子数相同、元素序列相同；否则拒绝（或拆成不同 molecule key）
  - 坐标应已中心化或可平移（pose 采样只依赖相对几何），在导入时统一做 `center_of_mass` 平移到原点（写入 provenance）

- **可复现**：
  - `conformer_id` 的排序规则固定（extxyz 顺序优先；若有能量则可选 stable sort）
  - random seed 派生固定：`seed_case = hash(slab_case, mol, conformer_id, global_seed)`，并写入 metadata

- **计算可控**：
  - 添加预算控制：`max_conformers_per_molecule` 与 `max_combinations` 共同限制总工作量
  - report 中必须输出：总 conformer 数、实际使用数、丢弃原因统计（超预算/重复/无效）

#### M8.5 与现有 ensemble/basin/node 的集成点（最少改动）

集成策略：
- 对每个 (slab_case, molecule, conformer_id) 生成一个独立 case，复用既有：
  - surface preprocessing
  - primitive basis
  - pose sampling + postprocess
  - ensemble generation（basin/node）
- provenance 扩展：
  - basin/node 的 provenance 里新增 `conformer_id`、`conformer_hash`、`conformer_source`
  - nodes.json 顶层每个 node 也携带这三个字段（便于下游合并/过滤）

去重策略（可选扩展，v2/v3）：
- 先在同一 molecule 内做“气相构象去重”（conformer_md 已做 selection，可直接复用）
- 吸附后仍以 basins/node 去重为准；允许不同构象落到同一个 basin（这是预期行为），此时 provenance 记录“来自哪些 conformer”

#### M8.6 CLI 与 Notebook（面向用户的一键入口，v1 只支持 extxyz）

CLI（v1）：
- 扩展 `tools/run_pose_sampling_sweep.py`：
  - `--adsorbate-extxyz <path>`：单分子 conformer 输入（与 `--slab-name` 等搭配）
  - `--max-conformers-per-molecule <int>`
  - `--conformer-source extxyz`
  - 保留现有 `--max-molecules` 语义：当提供 extxyz 时，`max-molecules` 仅限制 slab 数/组合数，不再枚举 molecule list

Notebook（v1）：
- 在 `examples/full_usage.ipynb` 增加一节：
  - 读取/构造一个 `conformers.extxyz`（最少 2–3 帧）
  - 调用新 API/CLI 对 `conformers.extxyz` 做吸附 workflow（验证 conformer_id 传播与输出目录规则）

#### M8.7 测试与验收（必须覆盖的回归）

1) 单元测试：
- extxyz conformer 读入的硬校验（元素序列不一致、原子数不一致等）
- `conformer_id` 传播到 summary/nodes/provenance

2) 端到端最小测试（不依赖 xtb/mace）：
- 使用 FakeMDRunner 或直接构造 3 个构象（轻微扰动）写 extxyz
- 对一个小 slab（如 fcc211 的小尺寸）跑 `max_conformers_per_molecule=2`，验证：
  - 生成 2 个 case 输出目录
  - 每个 case 都产出 `basins.extxyz/nodes.json`
  - summary csv/json 中 conformer 字段齐全

3) 可选集成测试（WSL + MACE）：
- 复用现有 `wsl_run_fcc211_co_demo.py` 增加“conformer 输入路径”的分支（只在环境可用时跑）

---

## 3. Benchmark 设计（既能发 paper，也能指导改代码）

总体原则：
- 分层 benchmark：B0（几何/primitive 单测）-> B1（旗舰刚性/半刚性三 case）-> B2（可选柔性扩展）-> B3（hard-case fallback，少量昂贵上限）-> B4（可选社区对齐）
- 所有 benchmark 固定随机种子、固定 candidate budget、固定 relax backend、固定去重与 canonicalization 规则
- 指标按三层输出组织：primitive/basis 层、basin/node 层、downstream probe 层

### B0：几何与 primitive 单元测试套件（工程验收，不做主结果）

系统：bare slabs（Pt(111)/(100)/(211)、Cu(211)、vacancy/missing-row、PdAg(111)、可选 TiO2(101)）

指标：
- surface atom detection precision/recall
- primitive raw count 与 reduced basis count
- normal direction stability / frame consistency

验收标准（建议）：
- surface atom recall 在规整金属面接近 1；复杂面允许下降但应可解释
- frame consistency 在同一 slab 旋转/平移下保持等价（数值容差）

### B1：刚性/半刚性 adsorption core benchmark（主文核心）

建议先把主文的“可解释因果链”做干净，只保留 3 个 flagship cases（每个 case 以 adsorbate family 为单位组织）：
- Case A（方法英雄）：Rh(211)（对照 Rh(111)）+ C2 oxygenates family（CH2CO*, CH3CO*, CH3CHO*, CH3CH2O*）
- Case B（node-to-edge 接口英雄）：Pt(111) + C2Hx ethane NDH family（CH3CH2*, CH2CH2**, CH3CH**, CH2CH**）
- Case C（hetero-surface + motif switch）：rutile TiO2(110) + HCOOH*/HCOO*（可选 hydrated/defected 作为扩展条件）

指标（主文 Fig.6 主报）：
- Success@0.1 eV（至少命中一个参考低能 basin）
- EnsembleRecall@0.2 eV（低能系综召回）
- Efficiency（valid unique basins / candidate budget；duplicate relax rate）

额外必须记录（服务 Fig.2–5 与方法学论证）：
- primitive richness：raw primitives count、basis count、compression_ratio
- false-merge rate（site embedding ablation）：不同 adsorption outcome 的 primitives 被错误合并比例
- primitive-to-basin mapping complexity：many-to-one 与 one-to-many 比例
- node statistics：distinct canonical nodes 数量、basin→node 合并率、node 去重稳定性

### B2：柔性 adsorption benchmark（可选扩展，避免稀释主线）

建议先不作为主文核心，只在 SI 或 discussion 证明框架可扩展：
- 选 1–2 个柔性分子在金属面（例如 Rh(111)/(211) 或 Cu(111)/(211)）做小规模对照即可

重点指标：
- conformer branch 对 Success 与 Recall 的增益
- unique basin yield 与去重压缩率

### B3：hard-case fallback benchmark（只在少数系统上比较“更贵上限”）

建议 2–3 个系统足够：
- CH2CO/Rh(211)（复杂 PES 上限参考）
- （可选）TiO2(110)+HCOOH* hydrated 条件（验证 motif reweighting 的稳健性）

baseline：GlobalOpt fallback（on-the-fly MLIP global optimization / minima hopping / annealing）

### B4：可选 community-compatibility benchmark（小规模 OC20-Dense 子集）

目标：报告 success@0.1 eV、DFT speedup、valid structure rate，便于审稿人定位象限  
注意：不是主战场，工作量可控时再做

---

## 4. Baseline 设置（4 个足够，且可控/可复现）

所有 baseline 共享同一 relax + anomaly + dedup backend。

- Baseline A：GraphHeur（classical sites、固定 height、固定 orientation grid、无 embedding clustering、无 lateral shift、无 conformer）
- Baseline B：AnchorAware（自动 anchor 检测，site×anchor×orientation 枚举；柔性先 conformer search；其余同后端）
- Baseline C：Rand+MLIP（random COM + random orientation + 简单 height；同后端）
- Baseline D：GlobalOpt fallback（仅 hard cases；作为昂贵参考上限）

---

## 5. Reference basins 构造协议（EnsembleRecall 能否站住的关键）

对每个 system 构造 dense reference pool：
- union(GraphHeur_dense, AnchorAware_dense, Rand+MLIP_dense, Ours_dense, GlobalOpt_dense[hard-only])

统一流程：
1) 同一 MLIP relax（含 staged）
2) anomaly filter
3) dedup（graph hash -> RMSD+energy）
4) top-N 做 DFT single-point
5) top-M 做 DFT relax（必要时）
6) 再次 dedup 得到 reference basins

记录不变量：
- reference 构造必须可追溯（每一步输入输出、种子、预算、阈值）
- reference basins 的能量基准一致（同一 DFT 设置/同一 slab cell）

---

## 6. 主文 7 图与实验映射（写作与实验绑定，减少跑偏）

- Fig.1：问题定义与定位（node paper）：primitives → basis → anchor-free basin generation → canonicalized nodes → downstream CCQN probe
- Fig.2：Arbitrary slab → exposed-surface primitives（3 类表面：low-index / step-high-index / alloy/defect 或 oxide），强调“在规整表面退化为经典 site motifs”
- Fig.3：Chemistry-aware primitive basis（embedding 构造 + UMAP/t-SNE + compression_ratio + false-merge rate ablation）
- Fig.4：Anchor-free, site-conditioned SE(3) basin generation（local frame + shifts + height solve；primitive-to-basin bipartite map 展示 many-to-one 与 one-to-many）
- Fig.5：Reaction-ready node canonicalization（relaxed basin → internal graph + binding graph → canonical node_id + metadata card）
- Fig.6：主 benchmark（Success@0.1 eV、Recall@0.2 eV、efficiency；按三类困难轴分组：surface complexity / binding ambiguity / hetero-surface chemistry）
- Fig.7：Hard cases + downstream probe（上：Rh(211) basin ladder 与 TiO2 motif ladder；下：Pt(111) fixed-CCQN 的 seed-policy 对比：verified TS/IRC、unique modes、low-barrier coverage）

---

## 7. 工程落地路线图（建议的代码组织与验收节奏）

即使当前仓库尚未整理，也建议按“对象模型 + 纯函数模块”组织，保证可测与可替换：

- core 数据结构：SurfaceContext / SitePrimitive / AdsorbateConformer / Pose / Basin
- surface：voxelize、exterior flood-fill、surface graph
- primitive：enumerate、topo_hash、frame
- embedding：feature extract、pool、cluster
- pose：so3 sampler、shift sampler、height solve、placement validity
- relax：staged relax runners（统一接口，便于替换 MLIP/DFT）
- post：anomaly filter、contact graph、RMSD、basin clustering、ranking
- bench：B0-B3 runner、metrics、日志与结果导出

验收顺序（减少“跑全链条才发现几何错”）：
1) 先跑 B0（几何/拓扑/frames）全通过
2) 再跑 B1 的 2–3 个系统做 smoke test
3) 再扩展 B1 全套 + Fig.6
4) 再上 B2（柔性）与 conformer 分支
5) 最后做 B3（昂贵 fallback）与 Fig.7

---

## 8. 让 Codex 直接开始改代码的实施顺序（按批次提交）

不要一开始把所有 fancy 模块一起写；先把几何/拓扑与测试立住，再逐步接上 embedding、采样、松弛与 benchmark。

第 1 批 commit：feat(surface): voxel exposure + exposed-surface graph  
目标：
- detect_surface_atoms
- build_exposed_surface_graph
- bare slab 单元测试通过

第 2 批 commit：feat(site): 1c/2c/3c/4c primitive enumeration + local frame  
目标：
- primitive dataclass
- frame construction
- Pt(111) / Pt(100) / Pt(211) primitive sanity tests

第 3 批 commit：feat(mace): invariant node feature extraction + primitive embedding  
目标：
- 拿到每个 atom 的 invariant embedding
- primitive pooling
- topo-hash + clustering

第 4 批 commit：feat(sampler): anchor-free SE(3) pose sampler  
目标：
- uniform SO(3)
- tangent shifts
- clash-controlled height
- initial pose pruning

第 5 批 commit：feat(relax): pre-relax + loose relax + anomaly filter + dedup  
目标：
- 从 pose 到 basin
- basins 成为 first-class output

第 5.5 批 commit：feat(node): canonical node object + graph extraction  
目标：
- basin -> node（internal graph + binding graph + canonical id + provenance）
- node 去重与 basin 去重一致

第 6 批 commit：feat(flexible): CREST conformer branch  
目标：
- 只对自由 adsorbate 做 conformer ensemble
- 不要先写 full-slab xTBFF MD 版本

第 7 批 commit：feat(benchmark): benchmark runner + metrics + plotting  
目标：
- B0/B1/B2 套件
- success / recall / yield / cost plot

### 8.1 现有代码资产对照：adsorption.ipynb（已实现 vs 与论文主线的差距）

参考源码：[adsorption.ipynb](file:///c:/Users/user/xwechat_files/wxid_zzv1ec85xnn712_92f5/msg/file/2025-12/adsorption.ipynb)

按论文主线（primitive -> pose -> relax -> basin/ensemble）逐项对照：

- MACE 原子 invariant 特征（对应第 3 批的一部分，已具备原型）
  - 已实现：get_mace_atomic_features() 使用 MACECalculator.get_descriptors 提取 (N_atoms, F_dim) 特征。
  - 缺口：目前没有把 node features 明确限定为“invariants-only + 指定层输出 + 批量推理接口”；也缺少与 primitive pooling/embedding 的稳定接口约束。

- 候选位点/primitive 枚举（对应第 2 批的一部分，已具备原型，但不是 arbitrary-slab 版本）
  - 已实现：
    - find_all_candidate_sites_graph()：基于 surface_indices 子集的邻接图 clique 枚举 1/2/3/4-fold。
    - find_all_candidate_sites()：基于投影到 XY 的 2D Delaunay，提取 2c/3c，并由相邻三角形合成 4c。
  - 关键差距（需要在 plan 中明确为“要替换/升级”的点）：
    - 需要外部提供 surface_indices；没有 detect_surface_atoms（voxel exposure + exterior flood-fill）与 exposed-surface graph 的定义，因此不满足 arbitrary slab。
    - Delaunay 版本显式假设 slab 已对齐 Z 为法向量，且以 XY 投影做平面几何；对 high-index、缺陷、粗糙表面会有系统性失真。
    - 当前命名仍带 ontop/bridge/hollow_3/hollow_4；论文与代码需要统一成 1c/2c/3c/4c primitive 术语。

- “adsorption-equivalent” site reduction（对应第 3 批的一部分，已具备原型，但尚未形成 minimal basis + topo-hash + clustering 的完整闭环）
  - 已实现：
    - find_inequivalent_sites_by_feature()：canonical signature（lexsort + allclose）去重。
    - find_inequivalent_sites_by_feature_V4_4()：rounding + frozenset 哈希去重（更接近 topo-hash 的精神）。
  - 缺口：
    - 尚未显式引入 topo_hash（基于 exposed-surface topology）与“先拓扑分桶、再 embedding/geometry 聚类”的两阶段聚类策略。
    - 尚未产生“minimal basis”的数据结构输出（basis_id、代表元、cluster 统计）。

- Pose 生成（对应第 4 批，已具备“主轴对齐 + 面内旋转”的原型，但不是论文要强调的 anchor-free site-conditioned SE(3)）
  - 已实现：generate_adsorption_configurations_V4_PrincipalAxis()
    - 通过 get_principal_axes_and_type() 自动判别分子类型并选不等价主轴；
    - 将 principal_axis 对齐到 local_normal（由 get_local_normal 或 global normal 给出）；
    - 通过 in_plane_step 扫面内旋转；
    - 通过“动态最低点”选择 binding atom（dynamic_binding_atom_k）并用共价半径估计高度；
    - check_collision() 做表面-吸附质碰撞过滤。
  - 关键差距：
    - 不是 uniform SO(3)，也没有 (u,v) tangent-plane shifts，因此采样空间仍带结构性偏置；
    - “动态最低点绑定原子”本质上引入了一个隐式 anchor 选择规则，和论文叙事的“无预定义 binding atoms”并不完全一致（建议作为 AnchorAware baseline 的一部分或 ablation，而非主方法的最终形态）。
    - 高度初始化是启发式 (r_ads + r_site_avg) * height_factor，尚未实现 clash-controlled height solve（按最小距离比值逼近 tau）。

- Relax / 去重 / basin 输出（对应第 5 批，核心缺失）
  - 现状：adsorption.ipynb 主流程产出的是“初始构型列表”（Atoms），没有 staged MLIP relax、没有 anomaly filter、没有 adsorption contact graph hash、没有 basin 聚类与 provenance。
  - 已有可复用资产（偏工程工具层）：ConformerExplorer（GFNFF MD -> MACE 特征 FPS -> xTB pre-relax -> MACE batch relax -> energy+feature RMSD 去重）。
  - 计划调整建议：把 ConformerExplorer 明确定位为“自由分子 conformer generation 的可选实现”，而 adsorption 主链仍按第 5 批定义实现 basin-first 输出。

- Flexible 分支（对应第 6 批，部分具备，但实现路线与当前 plan 不同）
  - 已实现：ConformerExplorer 提供 free-adsorbate 的构象探索与去重（MD+FPS 方案）。
  - 缺口：尚未按论文主线把 conformers 作为 AdsorbateConformer 对象接入 site-conditioned sampler；也未实现 CREST 接口（若 CREST 作为主路线）。

对提交顺序的影响（把“已有原型”转成“最小侵入的可复现库能力”）：
- 第 1 批：当前 notebook 基本未覆盖（必须补齐 detect_surface_atoms 与 exposed-surface graph，才能称为 arbitrary slab）。
- 第 2 批：候选 1c/2c/3c/4c 枚举已有原型，但建议把 Delaunay/graph-clique 作为 fallback/对照实现，主实现迁移到 exposed-surface graph motifs。
- 第 3 批：MACE 原子特征与“特征签名去重”已有原型，可直接沉淀为 embedding + clustering 的骨架。
- 第 4 批：主轴对齐采样可作为 baseline 或 ablation；主方法仍需补齐 uniform SO(3) + tangent shifts + height solve + pose pruning。

### 8.2 结合 detect_surf_atoms.py 与“覆盖优先”选择策略的更新

参考源码：[detect_surf_atoms.py](file:///c:/Users/user/Downloads/detect_surf_atoms.py)

#### A) Slab 判定与法向识别（第 1 批新增硬约束）

- 主方案：先用 ASE dimensionality/isolation 做体系维度识别与 slab 轴候选（1D/2D/3D 组件判定），输出 `is_slab` 与 `normal_axis_candidates`。
- 回退方案：若 isolation 判定不稳定或多候选冲突，则使用几何统计回退（真空厚度最大轴 + 顶层/底层原子分布）确定法向。
- 工程不变量：
  - 若不是 slab（例如 3D bulk），主流程拒绝进入 adsorption primitive 枚举并给出显式错误；
  - 若是 slab，必须产出统一朝向到真空侧的 `global_normal` 与置信度评分。

#### B) 表面原子检测（吸收 detect_surf_atoms.py 经验，但升级为可配置策略）

- 你提供的实现核心可复用点：
  - `mic_displacements()`：在指定 pbc 下做 MIC 距离。
  - `probe_surface()`：沿法向扫描探针网格识别 first-hit 可达原子（暴露原子）。
- 计划中的升级与统一接口：
  - 新建 `SurfaceAtomDetector` 抽象接口，提供两种策略：
    - `ProbeScanDetector`（基于探针扫描，继承 detect_surf_atoms.py 思路）
    - `VoxelFloodDetector`（主计划方案：voxel occupancy + exterior flood-fill）
  - 两策略输出统一为：`surface_atom_ids`, `exposure_score`, `diagnostics`（命中率、扫描网格密度、候选法向）。
  - B0 中对两策略做一致性测试，允许差异但要求在规整表面（Pt(111)/(100)/(211)）达到高重合。

#### C) 选择策略面向对象封装（覆盖优先，不同场景不同原则）

你指出的关键是正确的：conformer / site / pose / basin 这四类“选子集”不应共用同一规则。计划改为策略对象化：

- `SelectionStrategy`（统一协议）
  - `fit(candidates, context)` / `select(k|budget)` / `explain()`。
- `ConformerSelectionStrategy`
  - 目标：保持构象多样性 + 低能覆盖；
  - 推荐默认：FPS(K-center) in descriptor space + energy window 双阈值。
- `SiteSelectionStrategy`
  - 目标：最大化 adsorption-equivalent 覆盖；
  - 推荐默认：topo-hash 分桶后，每桶 embedding 聚类代表元采样（不是按能量）。
- `PoseSelectionStrategy`
  - 目标：在预算内覆盖 SE(3) 空间并控制无效姿态；
  - 推荐默认：分层采样（SO(3) 均匀 + shifts + tau）后做初始硬筛。
- `BasinSelectionStrategy`
  - 目标：低能 + 非冗余 + 反应网络可用；
  - 推荐默认：graph-hash 去重后，按 energy-diversity 联合排序。

#### D) 双阈值筛选规范（从“是否加能量阈值”变成标准能力）

把能量阈值与结构阈值设为可组合过滤器，而非散落在流程里的 if：

- `EnergyWindowFilter(ΔE)`：保留 `E <= E_min + ΔE`。
- `RMSDFilter(r_cut)`：按 heavy-atom RMSD 或 descriptor-RMSD 去重。
- `CompositeFilter`：支持先能量后结构、先结构后能量两种顺序（两者都做 ablation）。

默认建议（按场景）：
- conformer：先能量窗再 FPS（避免高能噪声主导多样性）
- basin：先 graph hash，再 RMSD，再能量窗（避免同构重复挤占名额）

#### E) 系统测试矩阵（新增到第 7 批 benchmark runner）

- 单元测试：
  - slab 判定：slab / bulk / molecule 三类输入必须分类正确；
  - surface atom 检测：ProbeScanDetector 与 VoxelFloodDetector 在规整面高一致；
  - 选择策略：每个策略的可重复性（固定 seed）与预算约束正确性。
- 回归测试：
  - 关心“覆盖率不退化”：固定预算下，PrimitiveCoverage 与 EnsembleRecall 不低于基线；
  - 关心“有效产率”：valid unique basins / 100 poses 维持或提升。
- 对照实验：
  - 单阈值 vs 双阈值；
  - FPS vs 随机；
  - site clustering on/off；
  - pose pruning on/off。

---

## 9. 哪些地方先不要过度设计（减少返工）

- 不要一开始就强求 edge latent feature；版本 1 用 MACE invariant node features + 少量 geometry auxiliaries 足够。
- 不要把“site type”当成最终输出；primitive 和 basin 必须是两个对象。
- 不要一开始就写 full-slab xTB 高温 MD；版本 1 先写 free-adsorbate conformer generation，再统一走 anchor-free sampler。
- 不要过早做 crystallographic symmetry reduction；主 reduction 应该是 adsorption-equivalent clustering，不是群论对称性。
- 不要把 oxide 放进第一批主 benchmark，除非已确认 MLIP 很稳。

---

## 10. 三个 flagship cases 的分工与执行优先级（按“先跑通因果链”排序）

- 第一优先：Pt(111) + C2Hx（最容易把 adsorption ensemble 与 CCQN 接口跑通）
- 第二优先：Rh(211) + C2 oxygenates（最能做出方法学英雄图：primitive-to-basin mapping）
- 第三优先：TiO2(110) + formic acid/formate（最能把“最稳定节点≠最重要节点、环境重排系综”讲透）

最小 3×3 对照矩阵（优先跑通的九格）：
- 体系：Rh(211)+CH2CO*；Pt(111)+CH2CH2**/CH3CH**；TiO2(110)+HCOOH*
- 方法：可控 baseline（heuristic/random）；本方法（ensemble generator）；本方法 + downstream CCQN probe
- 输出：adsorption coverage；canonical node set；downstream verified reactions

---

## 11. 最后把整篇 paper 浓缩成一句话（全文主句）

This work formulates adsorption structure generation as the mapping from arbitrary-slab site primitives to nonredundant low-energy adsorption basins, enabling reaction-network-ready adsorption ensembles without predefined adsorbate binding atoms.

---

## 12. 可复现性与实验管理（审稿与后期维护的生命线）

- 配置：所有预算/阈值/随机种子写入单一 config（YAML/JSON 皆可），实验输出包含 config snapshot
- RNG：numpy/torch/random 统一 seed，记录在日志与结果文件
- 版本：记录 MLIP/MACE/ASE 版本、模型权重 hash、硬件信息
- 数据：benchmark system 列表、slab/adsorbate 的来源与预处理固定（不要每次跑都“重新清洗”）
- 输出：每个 system 输出 basins（结构 + 能量 + provenance：来自哪些 primitive/pose）

---

## 13. 风险点与可替代方案（避免被某个依赖卡死）

- MACE node features 暂不可用：先用传统局部环境描述符（CN、SOAP-lite、几何特征）做 v0，但必须保留接口，便于切回 MACE
- MLIP 对某些体系不稳：B0/B1 先限定金属表面；氧化物放 SI stress test
- CREST/xTB 不可用：柔性分支先用 RDKit conformer（作为工程替代）；若 xTB 可用但无 CREST，可用 adsorption.ipynb 里的 ConformerExplorer（GFNFF MD + FPS）作为替代实现路径，并在主文明确差异与局限
- 4c motif 定义不唯一：先实现 4-cycle 版本并在 SI 给出 motif 定义与敏感性分析

---

## 14. 需要你确认的 5 个关键信息（确认后可把计划细化到“可执行任务清单”）

1) 你希望 v1 采用的 relax backend 是什么（MACE-MP/CHGNet/NequIP/其它）？是否已有稳定模型与权重？  
2) DFT reference 的标准化设置是什么（软件、泛函、k 点、色散、slab 固定策略、真空厚度）？你能接受的 DFT 预算大概落在哪个量级？  
3) B1/B2 的 slab 与 adsorbate 输入格式你希望统一为哪种（ASE Atoms、POSCAR、CIF）？是否要求输出也固定某格式？  
4) 第一篇是否只做金属/合金表面（Pt/Cu/Rh/PdAg）？氧化物（如 TiO2）是否严格只放 SI？  
5) 代码交付形态你更偏好哪种：纯 Python API（可被外部 workflow 调用）、CLI（面向批跑）、还是两者都要（API 为主，CLI 为薄封装）？

---

## 15. 当前实现进展（TDD + 案例驱动）

已完成首轮可运行骨架与测试：

- 新增包结构（面向对象）：
  - `adsorption_ensemble.surface.classifier`：`SlabClassifier`
  - `adsorption_ensemble.surface.detectors`：`SurfaceAtomDetector` 抽象 + `ProbeScanDetector` + `VoxelFloodDetector`
  - `adsorption_ensemble.surface.graph`：`ExposedSurfaceGraphBuilder`
  - `adsorption_ensemble.surface.pipeline`：`SurfacePreprocessor`（分类 -> 检测 -> 建图）
  - `adsorption_ensemble.site.primitives`：`SitePrimitive` + `PrimitiveEnumerator` + `LocalFrameBuilder` + `PrimitiveBuilder`
  - `adsorption_ensemble.selection.strategies`：`EnergyWindowFilter`、`RMSDSelector`、`FarthestPointSamplingSelector`、`DualThresholdSelector`
  - `adsorption_ensemble.visualization.sites`：`plot_surface_primitives_2d` + `plot_surface_sites_from_groups`（PNG 可视化）
  - `adsorption_ensemble.surface.report`：`export_surface_detection_report`（surface 原子 CSV/XYZ/tagged slab 导出）
  - `adsorption_ensemble.site.delaunay`：Delaunay 位点枚举与 graph-vs-delaunay 对照统计
  - 位点中心/距离判定改为显式 MIC 距离策略（ASE `get_distance/get_distances`）
  - surface 预处理改为“双检测器并行评估 + 结果择优”（优先保留覆盖更完整且合理的检测结果）
  - 高指数晶面 side-filter 改为“按最大 z-gap 分簇”，避免用 median 误砍台阶层原子
  - 新增目标计数约束：`target_surface_fraction=0.25`（四层 slab 默认目标为总原子数的 1/4）
  - 缺陷计数规则：原子数 `mod 4 = 1/3` 时按 `ceil/floor(N*0.25)` 自动映射 adatom/vacancy（避免强制回到16）

- 已完成案例驱动测试（ASE build）：
  - slab/bulk/molecule 分类测试
  - fcc111/fcc211/surface 的表面原子检测测试
  - exposed-surface graph 连通边测试
  - Pt(111)/(100)/(211) primitive 枚举健全性测试
  - primitive local frame 正交归一性测试
  - primitive 位点可视化 PNG 生成测试
  - 位点最小间距去冗余与“高配位优先保留”测试
  - 多晶面/多参数 sweep 诊断脚本测试（graph/delaunay 对照、位点中心最小距离）
  - surface 检测报告导出测试
  - 双阈值与 FPS 选择策略测试

- 当前测试状态：
  - `py -3 -m pytest -q tests/test_pose_sampler.py tests/test_site_primitives.py tests/test_surface_pipeline.py`
  - 27 passed
  - 示例图：`artifacts/pt111_surface_sites_mic.png`
  - 检测报告：`artifacts/surface_reports/pt111_4x4/surface_atoms.csv`、`surface_atoms_only.xyz`、`slab_with_surface_tags.extxyz`
  - sweep 汇总：`artifacts/sweep/surface_sweep_summary.csv` 与 `surface_sweep_report.md`（含每组 `sites.png` + `sites_only.png` + `site_audit.csv`）
  - sweep 规模：26 个 case × 3 组参数 = 78 组实验（含缺陷 adatom/vacancy）
  - 新增位姿采样测试：`tests/test_pose_sampler.py`

- 与第 1 批目标的关系：
  - 已落地 `detect_surface_atoms` 的可扩展双策略框架与测试；
  - 已落地第 2 批核心骨架（1c/2c/3c/4c primitive + local frame）并通过 Pt 系列 sanity tests；
  - 已落地第 3 批核心骨架（primitive 均值池化 embedding + topo-hash 分桶聚类 + basis_id 输出）；
  - 已落地第 4 批核心骨架（anchor-free SE(3) pose sampler：uniform SO(3)、tangent shifts、clash-controlled height、initial pose pruning）；
  - 第 4 批调试结论：独位点姿态“看似单一”主要由 sweep 侧 `max_poses_per_site=4` 限幅与缺少显式方位角扫描导致；修复为 `n_azimuth` 显式 0~2π 采样、按 `(azimuth,rotation)` 分层保留，并把 sweep 默认上限提高到每位点 24；
  - 回归验证：`alloy_cuni_111_l4 + 2-butyne` 在新 run 中生成 254 个构型，`azimuth_unique_n=6`，显著提升位点附近方位覆盖；
  - 第 4 批进一步修复：初猜高度改为“基于位点构成原子 + 局域法向 + MIC 距离”的自适应 `height_taus` 求解，并增加 `site_contact_tolerance` 控制高度合理区间；
  - 采样输入改为“不等价位点最小子集”优先（`basis_primitives` 全量，`max_basis_sites` 可选限幅），并在构型元信息中记录 `site_atom_ids/site_normal`；
  - 新增姿态池后处理链：MACE特征预采样(FPS) → 粗弛豫 → RMSD+能量双阈值筛选 → 精弛豫 → 最终能量排序，输出阶段文件与 `pose_postprocess_metrics.json`；
  - 下一步聚焦第 5 批（pose -> basin 的 staged relax、异常过滤与去重）。

---

## 16. 下一步实施计划（把当前代码推进到 M5–M7 可用）

### 16.1 目标（v1 可验收）

- 给定 `slab + adsorbate + pose pool（来自 PoseSampler 或 pose_sweep 输出）`，产出：
  - `basins.extxyz`：按 basin 聚类去重后的低能结构集合（每条结构带 basin 元信息）
  - `basins.json`：每个 basin 的统计、异常过滤原因、去重归并信息、能量与 provenance
  - `nodes.json`：每个 basin 对应的 reaction-ready node（internal graph + binding graph + canonical node_id + denticity）
- 形成最小“固定下游消费者可读”的 node contract（先不绑定 CCQN，仅保证 node 字段齐全与稳定）。

### 16.2 文件与模块落地（不引入重依赖）

- `adsorption_ensemble/relax/`
  - `backends.py`：`RelaxBackend` 协议 + `IdentityRelaxBackend` + `MACERelaxBackend` + `MACEBatchRelaxBackend`（复用现有 MACE 配置规范化与 batch relax 逻辑）
- `adsorption_ensemble/basin/`
  - `types.py`：`BasinConfig`、`BasinCandidate`、`BasinResult`、`Basin`（basin-first 数据结构）
  - `anomaly.py`：`classify_anomaly()`（desorption/reconstruction/dissociation/burial 等）
  - `dedup.py`：contact/binding signature + RMSD 聚类（contact-hash → RMSD+energy 二级去重）
  - `pipeline.py`：`BasinBuilder.build()`：poses → relax → anomaly → dedup → basins（单入口）
- `adsorption_ensemble/node/`
  - `types.py`：`NodeConfig`、`ReactionNode`
  - `canonicalize.py`：`build_internal_graph()`、`build_binding_graph()`、`canonicalize_adsorbate_order()`、`make_node_id()`

### 16.3 关键不变量（强制硬校验）

- 可复现：固定 seed + 固定 relax backend + 固定阈值 → basins/node_id 稳定（允许极小浮点差但不允许系统性分裂）
- 一致性：basin dedup 与 node dedup 规则不冲突；同一 node 不得重复输出
- 物理合理性：异常分类可解释（至少给出 fail reason 与关键统计），并可通过阈值回归测试锁定行为

### 16.4 测试与验收（先小体系、后扩展）

- 新增单测：
  - `tests/test_basin_pipeline.py`：小 slab + 小分子，构造少量 pose，验证输出 basins 数量与去重稳定性、异常过滤可触发
  - `tests/test_node_canonicalization.py`：同一吸附态在不同原子排列/微扰下 `node_id` 不变；不同 denticity/binding graph 必须给不同 `node_id`
- 新增最小 smoke：
  - 在 `workflows/smoke.py` 加入可选 `basin_enabled` 路径（默认关闭），只跑 1–2 组合并检查输出文件存在且可读回

### 16.5 集成策略（最小侵入）

- v1 先以“独立 pipeline + 可选 workflow”形式落地，不强耦合现有 `pose/sweep.py` 的默认输出；
- 等 basin/node 稳定后，再把 `pose/sweep.py` 通过开关接入（不破坏现有测试与输出格式）。

