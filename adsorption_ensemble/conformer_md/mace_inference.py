from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import os

import numpy as np
from ase import Atoms

from .config import MACEInferenceConfig


def _process_single_atom_config(image: Atoms, head_name: str):
    from mace import data

    try:
        info_map = {"total_charge": "charge", "total_spin": "spin"}
        keyspec = data.KeySpecification(info_keys=info_map, arrays_keys={"charges": "Qs"})
        config = data.config_from_atoms(image, key_specification=keyspec, head_name=[head_name])
        return config, None
    except Exception as exc:
        return None, str(exc)


@dataclass
class MACEBatchResult:
    descriptors: np.ndarray
    energies_per_atom_ev: np.ndarray
    metadata: dict


class MACEBatchInferencer:
    def __init__(self, config: MACEInferenceConfig):
        self.config = config
        self._model = None
        self._runtime_device = None
        self._runtime_meta = None

    def infer(self, frames: list[Atoms]) -> MACEBatchResult:
        if not frames:
            return MACEBatchResult(
                descriptors=np.empty((0, 0), dtype=float),
                energies_per_atom_ev=np.empty((0,), dtype=float),
                metadata={
                    "n_input_frames": 0,
                    "n_batches": 0,
                    "n_config_ok": 0,
                    "n_config_failed": 0,
                    "n_atomicdata_ok": 0,
                    "n_atomicdata_failed": 0,
                    "energy_fallback_batches": 0,
                    "edge_counts_per_batch": [],
                },
            )
        torch, extract_invariant, scatter_mean, has_scatter, tg, utils, data_mod, o3 = self._load_dependencies()
        runtime_device, runtime_meta = self._resolve_runtime_device(torch)
        model = self._get_model(torch, runtime_device=runtime_device)
        num_layers, num_inv_feats, l_max, feat_dim = self._resolve_feature_dims(model, o3)
        available_heads = self._discover_available_heads(model)
        head_name_use = self._resolve_head_name(model)
        batches_data, batches_img, build_stats = self._create_batches_parallel(
            frames=frames,
            max_edges=self.config.max_edges_per_batch,
            model=model,
            utils=utils,
            data_mod=data_mod,
            head_name=head_name_use,
            workers=self.config.num_workers,
        )
        all_desc: list[np.ndarray] = []
        all_epa: list[float] = []
        edge_counts_per_batch: list[int] = []
        energy_fallback_batches = 0
        for bd, bi in zip(batches_data, batches_img):
            edge_counts_per_batch.append(int(np.sum([d.edge_index.shape[1] for d in bd])))
            natoms = [len(a) for a in bi]
            loader = tg.dataloader.DataLoader(dataset=bd, batch_size=len(bd), shuffle=False)
            batch = next(iter(loader)).to(runtime_device)
            batch_dict = batch.to_dict()
            tdtype = torch.float64 if self.config.dtype == "float64" else torch.float32
            for k, v in list(batch_dict.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    batch_dict[k] = v.to(dtype=tdtype)
            out = model(batch_dict)
            ek = self.config.mlp_energy_key if self.config.mlp_energy_key else "energy"
            eout = out.get(ek)
            if eout is None and ek != "energy":
                eout = out.get("energy")
            if eout is None:
                energy_fallback_batches += 1
            energies = eout.detach().cpu().numpy() if eout is not None else np.zeros(len(natoms), dtype=float)
            all_epa.extend([float(e) / float(n) for e, n in zip(energies, natoms)])
            node_feats = out["node_feats"].detach()
            invs = extract_invariant(node_feats, num_layers=num_layers, num_features=num_inv_feats, l_max=l_max)
            if feat_dim is not None:
                invs = invs[:, :feat_dim]
            if has_scatter:
                pooled = scatter_mean(invs, batch.batch, dim=0).cpu().numpy()
            else:
                grp = batch.batch.cpu().numpy()
                invs_np = invs.cpu().numpy()
                pooled = np.array([invs_np[grp == k].mean(0) for k in range(len(bi))], dtype=float)
            all_desc.append(pooled)
        if all_desc:
            descriptors = np.concatenate(all_desc, axis=0)
        else:
            descriptors = np.empty((0, 0), dtype=float)
        metadata = {
            "model_path": self.config.model_path,
            "device": self.config.device,
            "requested_device": self.config.device,
            "runtime_device": runtime_meta["device"],
            "runtime_rank": runtime_meta["rank"],
            "runtime_world_size": runtime_meta["world_size"],
            "runtime_local_rank": runtime_meta["local_rank"],
            "dtype": self.config.dtype,
            "enable_cueq": bool(self.config.enable_cueq),
            "max_edges_per_batch": self.config.max_edges_per_batch,
            "layers_to_keep": self.config.layers_to_keep,
            "head_name": head_name_use,
            "available_heads": available_heads,
            "n_input_frames": len(frames),
            "n_batches": len(batches_data),
            "n_output_frames": len(all_epa),
            "n_config_ok": build_stats["n_config_ok"],
            "n_config_failed": build_stats["n_config_failed"],
            "n_atomicdata_ok": build_stats["n_atomicdata_ok"],
            "n_atomicdata_failed": build_stats["n_atomicdata_failed"],
            "energy_fallback_batches": energy_fallback_batches,
            "edge_counts_per_batch": edge_counts_per_batch,
        }
        return MACEBatchResult(descriptors=descriptors, energies_per_atom_ev=np.asarray(all_epa, dtype=float), metadata=metadata)

    def infer_node_descriptors(self, frames: list[Atoms]) -> tuple[list[np.ndarray | None], np.ndarray, dict]:
        if not frames:
            return [], np.empty((0,), dtype=float), {
                "n_input_frames": 0,
                "n_batches": 0,
                "n_config_ok": 0,
                "n_config_failed": 0,
                "n_atomicdata_ok": 0,
                "n_atomicdata_failed": 0,
                "energy_fallback_batches": 0,
                "edge_counts_per_batch": [],
                "head_name": None,
                "failed_indices": [],
            }
        torch, extract_invariant, _, _, tg, utils, data_mod, o3 = self._load_dependencies()
        runtime_device, runtime_meta = self._resolve_runtime_device(torch)
        model = self._get_model(torch, runtime_device=runtime_device)
        num_layers, num_inv_feats, l_max, feat_dim = self._resolve_feature_dims(model, o3)
        available_heads = self._discover_available_heads(model)
        head_name_use = self._resolve_head_name(model)
        batches_data, batches_img, batches_idx, build_stats = self._create_batches_parallel_aligned(
            frames=frames,
            max_edges=self.config.max_edges_per_batch,
            model=model,
            utils=utils,
            data_mod=data_mod,
            head_name=head_name_use,
        )
        node_desc: list[np.ndarray | None] = [None for _ in range(len(frames))]
        epa = np.full((len(frames),), np.nan, dtype=float)
        edge_counts_per_batch: list[int] = []
        energy_fallback_batches = 0
        for bd, bi, bix in zip(batches_data, batches_img, batches_idx):
            edge_counts_per_batch.append(int(np.sum([d.edge_index.shape[1] for d in bd])))
            natoms = [len(a) for a in bi]
            loader = tg.dataloader.DataLoader(dataset=bd, batch_size=len(bd), shuffle=False)
            batch = next(iter(loader)).to(runtime_device)
            batch_dict = batch.to_dict()
            tdtype = torch.float64 if self.config.dtype == "float64" else torch.float32
            for k, v in list(batch_dict.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    batch_dict[k] = v.to(dtype=tdtype)
            out = model(batch_dict)
            ek = self.config.mlp_energy_key if self.config.mlp_energy_key else "energy"
            eout = out.get(ek)
            if eout is None and ek != "energy":
                eout = out.get("energy")
            if eout is None:
                energy_fallback_batches += 1
            energies = eout.detach().cpu().numpy() if eout is not None else np.zeros(len(natoms), dtype=float)
            for ei, n, idx in zip(energies, natoms, bix):
                epa[int(idx)] = float(ei) / float(n)
            node_feats = out["node_feats"].detach()
            invs = extract_invariant(node_feats, num_layers=num_layers, num_features=num_inv_feats, l_max=l_max)
            if feat_dim is not None:
                invs = invs[:, :feat_dim]
            grp = batch.batch.detach().cpu().numpy()
            invs_np = invs.detach().cpu().numpy()
            for k, idx in enumerate(bix):
                node_desc[int(idx)] = np.asarray(invs_np[grp == k], dtype=float)
        failed_indices = [int(i) for i, v in enumerate(node_desc) if v is None]
        metadata = {
            "model_path": self.config.model_path,
            "device": self.config.device,
            "requested_device": self.config.device,
            "runtime_device": runtime_meta["device"],
            "runtime_rank": runtime_meta["rank"],
            "runtime_world_size": runtime_meta["world_size"],
            "runtime_local_rank": runtime_meta["local_rank"],
            "dtype": self.config.dtype,
            "enable_cueq": bool(self.config.enable_cueq),
            "max_edges_per_batch": self.config.max_edges_per_batch,
            "layers_to_keep": self.config.layers_to_keep,
            "head_name": head_name_use,
            "available_heads": available_heads,
            "n_input_frames": len(frames),
            "n_batches": len(batches_data),
            "n_config_ok": build_stats["n_config_ok"],
            "n_config_failed": build_stats["n_config_failed"],
            "n_atomicdata_ok": build_stats["n_atomicdata_ok"],
            "n_atomicdata_failed": build_stats["n_atomicdata_failed"],
            "energy_fallback_batches": energy_fallback_batches,
            "edge_counts_per_batch": edge_counts_per_batch,
            "failed_indices": failed_indices,
        }
        return node_desc, epa, metadata

    def _resolve_runtime_device(self, torch) -> tuple[str, dict]:
        preferred_device = str(self.config.device)
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ.get("RANK", rank))
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
        runtime_device = preferred_device
        if preferred_device.lower().startswith("cuda"):
            if not bool(torch.cuda.is_available()):
                runtime_device = "cpu"
            elif preferred_device == "cuda":
                runtime_device = f"cuda:{local_rank}"
        meta = {
            "rank": int(rank),
            "world_size": int(world_size),
            "local_rank": int(local_rank),
            "device": str(runtime_device),
        }
        self._runtime_device = str(runtime_device)
        self._runtime_meta = dict(meta)
        return str(runtime_device), meta

    def _get_model(self, torch, runtime_device: str | None = None):
        if self._model is not None:
            return self._model
        if not self.config.model_path:
            raise ValueError("MACE model_path is required for mace backend.")
        runtime_device_use = str(runtime_device or self._runtime_device or self.config.device)
        model = torch.load(self.config.model_path, map_location=runtime_device_use)
        model = model.float() if self.config.dtype == "float32" else model.double()
        model = model.to(runtime_device_use)
        if bool(self.config.enable_cueq) and runtime_device_use.lower().startswith("cuda"):
            try:
                from mace.calculators.mace import run_e3nn_to_cueq
            except Exception as exc:
                raise RuntimeError("enable_cueq=True but CuEq conversion is unavailable for MACE inference.") from exc
            model = run_e3nn_to_cueq(model, device=runtime_device_use).to(runtime_device_use)
        if runtime_device_use.lower().startswith("cuda"):
            model._enable_amp = False
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self._model = model
        return model

    def _resolve_feature_dims(self, model, o3):
        irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_inv_feats = irreps_out.dim // (l_max + 1) ** 2
        num_layers = int(model.num_interactions)
        keep = self.config.layers_to_keep if self.config.layers_to_keep != -1 else num_layers
        layer_dims = [irreps_out.dim for _ in range(num_layers)]
        layer_dims[-1] = num_inv_feats
        feat_dim = int(np.sum(layer_dims[:keep]))
        return num_layers, num_inv_feats, l_max, feat_dim

    def _create_batches_parallel(self, frames, max_edges, model, utils, data_mod, head_name, workers):
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        cutoff = float(model.r_max.cpu())
        configs_list = []
        n_config_failed = 0
        if workers > 1 and len(frames) > 50:
            func = partial(_process_single_atom_config, head_name=head_name)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(func, frames))
            for (conf, err), img in zip(results, frames):
                if conf is not None:
                    configs_list.append((conf, img))
                elif err is not None:
                    n_config_failed += 1
        else:
            for img in frames:
                conf, err = _process_single_atom_config(img, head_name=head_name)
                if conf is not None:
                    configs_list.append((conf, img))
                elif err is not None:
                    n_config_failed += 1
        data_list = []
        img_list = []
        edge_counts = []
        n_atomicdata_failed = 0
        for conf, img in configs_list:
            try:
                ad = data_mod.AtomicData.from_config(conf, z_table=z_table, cutoff=cutoff, heads=[head_name])
                data_list.append(ad)
                img_list.append(img)
                edge_counts.append(int(ad.edge_index.shape[1]))
            except Exception:
                n_atomicdata_failed += 1
        batches_data, batches_img = [], []
        curr_d, curr_i, curr_e = [], [], 0
        for ad, img, e in zip(data_list, img_list, edge_counts):
            if curr_e + e <= max_edges:
                curr_d.append(ad)
                curr_i.append(img)
                curr_e += e
            else:
                if curr_d:
                    batches_data.append(curr_d)
                    batches_img.append(curr_i)
                curr_d = [ad]
                curr_i = [img]
                curr_e = e
        if curr_d:
            batches_data.append(curr_d)
            batches_img.append(curr_i)
        stats = {
            "n_config_ok": len(configs_list),
            "n_config_failed": n_config_failed,
            "n_atomicdata_ok": len(data_list),
            "n_atomicdata_failed": n_atomicdata_failed,
        }
        return batches_data, batches_img, stats

    def _create_batches_parallel_aligned(self, frames, max_edges, model, utils, data_mod, head_name):
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        cutoff = float(model.r_max.cpu())
        data_list = []
        img_list = []
        idx_list = []
        edge_counts = []
        n_config_failed = 0
        n_atomicdata_failed = 0
        for idx, img in enumerate(frames):
            conf, err = _process_single_atom_config(img, head_name=head_name)
            if conf is None:
                n_config_failed += 1
                continue
            try:
                ad = data_mod.AtomicData.from_config(conf, z_table=z_table, cutoff=cutoff, heads=[head_name])
                data_list.append(ad)
                img_list.append(img)
                idx_list.append(int(idx))
                edge_counts.append(int(ad.edge_index.shape[1]))
            except Exception:
                n_atomicdata_failed += 1
        batches_data, batches_img, batches_idx = [], [], []
        curr_d, curr_i, curr_x, curr_e = [], [], [], 0
        for ad, img, ix, e in zip(data_list, img_list, idx_list, edge_counts):
            if curr_e + e <= max_edges:
                curr_d.append(ad)
                curr_i.append(img)
                curr_x.append(ix)
                curr_e += e
            else:
                if curr_d:
                    batches_data.append(curr_d)
                    batches_img.append(curr_i)
                    batches_idx.append(curr_x)
                curr_d = [ad]
                curr_i = [img]
                curr_x = [ix]
                curr_e = e
        if curr_d:
            batches_data.append(curr_d)
            batches_img.append(curr_i)
            batches_idx.append(curr_x)
        stats = {
            "n_config_ok": int(len(data_list) + n_atomicdata_failed),
            "n_config_failed": int(n_config_failed),
            "n_atomicdata_ok": int(len(data_list)),
            "n_atomicdata_failed": int(n_atomicdata_failed),
        }
        return batches_data, batches_img, batches_idx, stats

    @staticmethod
    def _normalize_head_name(value) -> str:
        return str(value).strip() if value is not None else ""

    @classmethod
    def _discover_available_heads(cls, model) -> list[str]:
        found: list[str] = []

        def add(value) -> None:
            name = cls._normalize_head_name(value)
            if name and name not in found:
                found.append(name)

        available_heads = getattr(model, "available_heads", None)
        if isinstance(available_heads, (list, tuple)):
            for item in available_heads:
                add(item)

        head_names = getattr(model, "head_names", None)
        if isinstance(head_names, (list, tuple)):
            for item in head_names:
                add(item)

        heads = getattr(model, "heads", None)
        if isinstance(heads, dict):
            for item in heads.keys():
                add(item)
        elif isinstance(heads, (list, tuple)):
            for item in heads:
                add(item)

        add(getattr(model, "head_name", None))
        add(getattr(getattr(model, "config", None), "head_name", None))
        return found

    def _resolve_head_name(self, model) -> str:
        preferred = self._normalize_head_name(self.config.head_name)
        available_heads = self._discover_available_heads(model)
        if preferred and preferred != "Default":
            if available_heads and preferred not in available_heads:
                raise ValueError(
                    f"Requested head '{preferred}' is not available in model heads {available_heads}."
                )
            return preferred
        if available_heads:
            return str(available_heads[0])
        return "Default"

    @staticmethod
    def _load_dependencies():
        import torch
        
        from mace import data
        from mace.modules.utils import extract_invariant
        from mace.tools import torch_geometric, utils
        from e3nn import o3

        try:
            from torch_scatter import scatter_mean

            has_scatter = True
        except ImportError:
            scatter_mean = None
            has_scatter = False
        return torch, extract_invariant, scatter_mean, has_scatter, torch_geometric, utils, data, o3
