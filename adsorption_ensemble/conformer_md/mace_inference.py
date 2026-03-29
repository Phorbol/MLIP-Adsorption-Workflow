from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

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
        model = self._get_model(torch)
        num_layers, num_inv_feats, l_max, feat_dim = self._resolve_feature_dims(model, o3)
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
            batch = next(iter(loader)).to(self.config.device)
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
            "dtype": self.config.dtype,
            "max_edges_per_batch": self.config.max_edges_per_batch,
            "layers_to_keep": self.config.layers_to_keep,
            "head_name": head_name_use,
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
        model = self._get_model(torch)
        num_layers, num_inv_feats, l_max, feat_dim = self._resolve_feature_dims(model, o3)
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
            batch = next(iter(loader)).to(self.config.device)
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
            "dtype": self.config.dtype,
            "max_edges_per_batch": self.config.max_edges_per_batch,
            "layers_to_keep": self.config.layers_to_keep,
            "head_name": head_name_use,
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

    def _get_model(self, torch):
        if self._model is not None:
            return self._model
        if not self.config.model_path:
            raise ValueError("MACE model_path is required for mace backend.")
        model = torch.load(self.config.model_path, map_location=self.config.device)
        model = model.float() if self.config.dtype == "float32" else model.double()
        model.to(self.config.device).eval()
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
    def _resolve_head_name(model) -> str:
        head = str(getattr(model, "head_name", "")).strip()
        if head:
            return head
        cfg = str(getattr(getattr(model, "config", None), "head_name", "")).strip()
        if cfg:
            return cfg
        heads = getattr(model, "heads", None)
        if isinstance(heads, dict) and heads:
            k0 = next(iter(heads.keys()))
            if isinstance(k0, str) and str(k0).strip():
                return str(k0).strip()
        if isinstance(heads, (list, tuple)) and heads:
            k0 = heads[0]
            if isinstance(k0, str) and str(k0).strip():
                return str(k0).strip()
        head_names = getattr(model, "head_names", None)
        if isinstance(head_names, (list, tuple)) and head_names:
            k0 = head_names[0]
            if isinstance(k0, str) and str(k0).strip():
                return str(k0).strip()
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
