from __future__ import annotations

import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.io import write as ase_write
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase.stress import full_3x3_to_voigt_6_stress
from mace import data
from mace.tools import torch_geometric
from tqdm.auto import tqdm

logger = logging.getLogger("MACE_BatchRelax")


def setup_distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ.get("RANK", rank))
        world_size = int(os.environ.get("WORLD_SIZE", world_size))
    return rank, world_size, local_rank


def resolve_runtime_device(preferred_device: str) -> tuple[str, dict]:
    rank, world_size, local_rank = setup_distributed_env()
    device = preferred_device
    if preferred_device.startswith("cuda"):
        if not torch.cuda.is_available():
            device = "cpu"
        elif preferred_device == "cuda":
            device = f"cuda:{local_rank}"
    meta = {
        "rank": int(rank),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "device": device,
    }
    return device, meta


def _get_mace_config_and_data(atoms: Atoms, calculator, heads: List[str]) -> data.AtomicData:
    key_spec = data.KeySpecification(
        info_keys={},
        arrays_keys={calculator.charges_key: "Qs"} if hasattr(calculator, "charges_key") else {},
    )
    if isinstance(heads, list):
        head_name = heads[0] if len(heads) > 0 else "Default"
    else:
        head_name = str(heads)
    config = data.config_from_atoms(
        atoms,
        key_specification=key_spec,
        head_name=head_name,
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=calculator.z_table,
        cutoff=calculator.r_max,
        heads=heads,
    )
    return atomic_data


class RelaxBatch:
    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        fmax: float = 0.01,
        atoms_filter_cls=None,
        max_n_steps: int = 500,
        device: str = "cuda",
        optimizer_kwargs: Dict | None = None,
        target_heads: List[str] | None = None,
        compute_stress: Optional[bool] = None,
        data_builder: Optional[Callable[[Atoms], data.AtomicData]] = None,
    ):
        self.calc = calculator
        self.model = calculator.models[0]
        self.optimizer_cls = optimizer_cls
        self.fmax = fmax
        self.atoms_filter_cls = atoms_filter_cls
        self.max_n_steps = max_n_steps
        self.device = device
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.target_heads = target_heads if target_heads else ["Default"]
        self.compute_stress = compute_stress
        self.data_builder = (
            data_builder
            if data_builder is not None
            else (lambda atoms: _get_mace_config_and_data(atoms, self.calc, heads=self.target_heads))
        )
        self.opt_list: List[Optimizer] = []
        self.all_atoms: List[Atoms] = []
        self.edge_counts: List[int] = []
        self.opt_flags: List[bool] = []
        self.ids: List[Any] = []
        self.cached_data: List[Optional[data.AtomicData]] = []
        self.trajectories: List[Union[Trajectory, None]] = []
        self.total_edges: int = 0

    @property
    def num_active(self) -> int:
        return sum(self.opt_flags)

    def insert(
        self,
        atoms: Atoms,
        num_edges: int,
        idx: Any,
        logfile=None,
        traj_file=None,
        data_obj: Optional[data.AtomicData] = None,
    ) -> None:
        atoms.calc = SinglePointCalculator(atoms)
        if self.atoms_filter_cls:
            filtered_atoms = self.atoms_filter_cls(atoms)
        else:
            filtered_atoms = atoms
        final_kwargs = self.optimizer_kwargs.copy()
        if "opt_kwargs" in atoms.info and isinstance(atoms.info["opt_kwargs"], dict):
            final_kwargs.update(atoms.info["opt_kwargs"])
        opt = self.optimizer_cls(
            filtered_atoms,
            logfile=logfile,
            trajectory=None,
            **final_kwargs,
        )
        opt.fmax = self.fmax
        traj_handler = None
        if traj_file:
            traj_handler = Trajectory(traj_file, "w", atoms)
            traj_handler.write(atoms)
        self.opt_list.append(opt)
        self.all_atoms.append(atoms)
        self.edge_counts.append(num_edges)
        self.opt_flags.append(True)
        self.ids.append(idx)
        self.cached_data.append(data_obj)
        self.trajectories.append(traj_handler)
        self.total_edges += num_edges

    def pop_converged(self) -> List[Tuple[Any, Atoms]]:
        new_opt_list = []
        new_all_atoms = []
        new_edge_counts = []
        new_ids = []
        new_cached_data = []
        new_trajectories = []
        converged_items = []
        for i in range(len(self.opt_list)):
            if self.opt_flags[i]:
                new_opt_list.append(self.opt_list[i])
                new_all_atoms.append(self.all_atoms[i])
                new_edge_counts.append(self.edge_counts[i])
                new_ids.append(self.ids[i])
                new_cached_data.append(self.cached_data[i])
                new_trajectories.append(self.trajectories[i])
            else:
                converged_items.append((self.ids[i], self.all_atoms[i]))
                if self.trajectories[i] is not None:
                    self.trajectories[i].close()
        self.opt_list = new_opt_list
        self.all_atoms = new_all_atoms
        self.edge_counts = new_edge_counts
        self.ids = new_ids
        self.cached_data = new_cached_data
        self.trajectories = new_trajectories
        self.opt_flags = [True] * len(self.opt_list)
        self.total_edges = sum(self.edge_counts)
        return converged_items

    def step(self):
        if not self.opt_list:
            return
        real_atoms_list = []
        for opt in self.opt_list:
            if self.atoms_filter_cls:
                real_atoms_list.append(opt.atoms.atoms)
            else:
                real_atoms_list.append(opt.atoms)
        data_list: List[data.AtomicData] = []
        for i, atoms in enumerate(real_atoms_list):
            cached = self.cached_data[i]
            if cached is not None:
                data_list.append(cached)
                self.cached_data[i] = None
            else:
                data_list.append(self.data_builder(atoms))
        self.edge_counts = [int(d.edge_index.shape[1]) for d in data_list]
        self.total_edges = sum(self.edge_counts)
        batch = torch_geometric.Batch.from_data_list(data_list).to(self.device)
        use_compile = getattr(self.calc, "use_compile", False)
        batch["node_attrs"].requires_grad_(True)
        batch["positions"].requires_grad_(True)
        if self.compute_stress is not None:
            compute_stress = self.compute_stress
        else:
            compute_stress = (self.calc.model_type in ["MACE", "EnergyDipoleMACE"]) and (not use_compile)
        out = self.model(
            batch.to_dict(),
            compute_stress=compute_stress,
            training=use_compile,
        )
        energies = out["energy"].detach().cpu().numpy()
        node_forces = out["forces"].detach().cpu().numpy()
        stresses = out["stress"].detach().cpu().numpy() if compute_stress else None
        ptr = batch.ptr.detach().cpu().numpy()
        for i, opt in enumerate(self.opt_list):
            target_atoms = real_atoms_list[i]
            start = int(ptr[i])
            end = int(ptr[i + 1])
            e = float(energies[i]) * self.calc.energy_units_to_eV
            f = node_forces[start:end] * self.calc.energy_units_to_eV / self.calc.length_units_to_A
            s = None
            if stresses is not None:
                stress_i = stresses[i]
                if getattr(stress_i, "ndim", 0) == 3:
                    stress_i = stress_i[0]
                s = full_3x3_to_voigt_6_stress(
                    stress_i * self.calc.energy_units_to_eV / self.calc.length_units_to_A**3
                )
            target_atoms.calc = SinglePointCalculator(target_atoms, energy=e, forces=f, stress=s)
            # ASE Optimizer.converged expects per-atom force shape (N, 3),
            # not a flattened 1D vector.
            current_f = opt.atoms.get_forces()
            step_count = getattr(opt, "nsteps", 0)
            if opt.converged(current_f) or (step_count >= self.max_n_steps):
                self.opt_flags[i] = False
            else:
                opt.step()
                if self.trajectories[i] is not None:
                    self.trajectories[i].write(target_atoms)


class BatchRelaxer:
    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        max_edges_per_batch: int = 30000,
        relax_cell: bool = False,
        device: str = "cuda",
    ):
        self.calc = calculator
        self.optimizer_cls = optimizer_cls
        self.max_edges = max_edges_per_batch
        self.default_relax_cell = relax_cell
        self.device = device
        if len(calculator.models) != 1:
            raise ValueError("BatchRelaxer only supports single-model calculators.")

    def relax(
        self,
        atoms_list: List[Atoms],
        fmax: float = 0.02,
        relax_cell: Optional[bool] = None,
        head: Optional[str] = None,
        max_n_steps: int = 200,
        inplace: bool = True,
        compute_stress: Optional[bool] = None,
        trajectory_dir: Optional[str] = None,
        append_trajectory_file: Optional[str] = None,
        save_log_file: Optional[str] = None,
        verbose: bool = False,
        optimizer_kwargs: Dict | None = None,
    ) -> List[Atoms]:
        use_relax_cell = relax_cell if relax_cell is not None else self.default_relax_cell
        available_heads = getattr(self.calc, "available_heads", ["Default"])
        if available_heads is None:
            available_heads = ["Default"]
        target_heads_list = None
        if head is not None:
            if head not in available_heads:
                raise ValueError(f"Selected head '{head}' not in {available_heads}")
            target_heads_list = [head]
            logger.info(f"Using manually selected head: {head}")
        else:
            if len(available_heads) == 1:
                target_heads_list = available_heads
            elif len(available_heads) > 1:
                raise ValueError(f"Multiple heads {available_heads} found. Please specify 'head=...'.")
            else:
                target_heads_list = ["Default"]
        try:
            rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
        except Exception:
            rank = 0
        log_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(log_level)
        logger.propagate = False
        handlers: List[logging.Handler] = []
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(stream_handler)
        handlers.append(stream_handler)
        if save_log_file:
            file_handler = logging.FileHandler(save_log_file, mode="w")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(file_handler)
            handlers.append(file_handler)
        if not inplace:
            atoms_list = [at.copy() for at in atoms_list]
        queue = {i: at for i, at in enumerate(atoms_list)}
        relaxed_results = {}
        if trajectory_dir:
            os.makedirs(trajectory_dir, exist_ok=True)
        stream_obj = None
        if append_trajectory_file:
            stream_obj = open(append_trajectory_file, "w", encoding="utf-8")
        filter_cls = FrechetCellFilter if use_relax_cell else None
        get_data = lambda atoms: _get_mace_config_and_data(atoms, self.calc, heads=target_heads_list)
        worker = RelaxBatch(
            self.calc,
            optimizer_cls=self.optimizer_cls,
            fmax=fmax,
            atoms_filter_cls=filter_cls,
            max_n_steps=max_n_steps,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
            target_heads=target_heads_list,
            compute_stress=compute_stress,
            data_builder=get_data,
        )
        disable_pbar = bool(os.environ.get("MACE_BATCHRELAX_DISABLE_TQDM", "0") == "1") or (not sys.stderr.isatty())
        pbar = tqdm(total=len(atoms_list), desc=f"[Rank {rank}] Relaxing", unit="struct", disable=disable_pbar)
        try:
            while len(queue) > 0 or worker.num_active > 0:
                for idx in list(queue.keys()):
                    if worker.total_edges >= self.max_edges and worker.num_active > 0:
                        break
                    atoms = queue[idx]
                    try:
                        data_obj = get_data(atoms)
                        n_edges = data_obj.edge_index.shape[1]
                    except Exception as e:
                        logger.error(f"Failed to graph structure {idx}: {e}")
                        del queue[idx]
                        pbar.update(1)
                        continue
                    if n_edges > self.max_edges and worker.num_active > 0:
                        break
                    traj_path = None
                    if trajectory_dir:
                        name = atoms.info.get("name", atoms.info.get("ID", f"{idx}"))
                        traj_path = os.path.join(trajectory_dir, f"{name}.traj")
                    worker.insert(atoms, n_edges, idx, logfile=None, traj_file=traj_path, data_obj=data_obj)
                    del queue[idx]
                if worker.num_active > 0:
                    worker.step()
                converged = worker.pop_converged()
                if converged:
                    for idx, atoms in converged:
                        relaxed_results[idx] = atoms
                        pbar.update(1)
                        if stream_obj:
                            ase_write(stream_obj, atoms, format="extxyz")
                            stream_obj.flush()
                pbar.set_postfix(active=worker.num_active, edges=f"{worker.total_edges / 1000:.1f}k")
        finally:
            if stream_obj:
                stream_obj.close()
            pbar.close()
            for h in handlers:
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
        logger.info("Relaxation finished.")
        return [relaxed_results.get(i, None) for i in range(len(atoms_list))]
