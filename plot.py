#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified script for MACE model inference, analysis, and plotting.
VERSION: V6 ADAPTIVE (Fixes TypeError for models without native LES support)

Updates:
1. BUGFIX: Uses `inspect` to check model.forward signature.
2. ADAPTIVE LES: If model doesn't accept `compute_bec`, calls model.les manually.
3. ROBUSTNESS: Keeps all previous V5 features (Jitter KDE, Stress Components, etc.)
"""

import numpy as np
import matplotlib
# Use 'Agg' backend to prevent errors on headless servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from ase.io import iread
from ase import Atoms
from ase.data import chemical_symbols
import torch
from tqdm import tqdm
import argparse
import glob
import os
import sys
import yaml
import inspect  # <--- Added for signature inspection
from collections import defaultdict
from typing import List, Generator
from concurrent.futures import ProcessPoolExecutor
import functools
from ase.io import write, read
from kit.services.atoms_metadata import get_total_energy, suggest_energy_keys
# --- ML & Stats Imports ---
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# --- Imports from MACE ---

from mace.modules.models import ScaleShiftMACE
from e3nn import o3
from mace import data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, utils

# --- PyTorch Safe Loading ---


# --- Torch Scatter Import with Fallback ---
try:
    from torch_scatter import scatter_mean
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False
    print("[Warning] 'torch_scatter' not found. Using NumPy fallback (slower).")


# =============================================================================
# --- Helper Functions ---
# =============================================================================

def read_chunks(filename: str, chunk_size: int) -> Generator[List[Atoms], None, None]:
    """Reads ASE file in chunks."""
    chunk = []
    iterator = iread(filename)
    while True:
        try:
            for _ in range(chunk_size):
                chunk.append(next(iterator))
            yield chunk
            chunk = []
        except StopIteration:
            if chunk:
                yield chunk
            break
        except Exception as e:
            print(f"Error reading file: {e}")
            break

def _process_single_atom_config(image: Atoms, head_name: str):
    """Worker: Atoms -> Config."""
    try:
        # Map specific info keys if necessary
        info_map = {"total_charge": "charge", "total_spin": "spin"}
        keyspec = data.KeySpecification(
            info_keys=info_map, 
            arrays_keys={"charges": "Qs"} 
        )
        config = data.config_from_atoms(image, key_specification=keyspec, head_name=[head_name])
        return config, None
    except Exception as e:
        return None, str(e)

def create_batches_parallel(images: List[Atoms], max_edges: int, model, head_name="Default", workers=4):
    """Parallel preprocessing of Atoms into AtomicData batches."""
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max.cpu())
    
    configs_list = []
    if workers > 1 and len(images) > 50:
        func = functools.partial(_process_single_atom_config, head_name=head_name)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(func, images))
        for (conf, err), img in zip(results, images):
            if conf: configs_list.append((conf, img))
    else:
        for img in images:
            conf, err = _process_single_atom_config(img, head_name)
            if conf: configs_list.append((conf, img))

    data_list, img_list, edge_counts = [], [], []
    for conf, img in configs_list:
        try:
            ad = data.AtomicData.from_config(conf, z_table=z_table, cutoff=cutoff, heads=[head_name])
            data_list.append(ad); img_list.append(img); edge_counts.append(ad.edge_index.shape[1])
        except: pass

    batches_data, batches_img = [], []
    curr_d, curr_i, curr_e = [], [], 0
    for ad, img, e in zip(data_list, img_list, edge_counts):
        if curr_e + e <= max_edges:
            curr_d.append(ad); curr_i.append(img); curr_e += e
        else:
            if curr_d: batches_data.append(curr_d); batches_img.append(curr_i)
            curr_d = [ad]; curr_i = [img]; curr_e = e
    if curr_d: batches_data.append(curr_d); batches_img.append(curr_i)
    return batches_data, batches_img

def calculate_metrics(true, pred):
    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    
    t, p = true[mask], pred[mask]
    mae = np.mean(np.abs(t - p))
    rmse = np.sqrt(np.mean((t - p) ** 2))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else float('nan')
    return {"mae": mae, "rmse": rmse, "r2": r2}

def fit_e0_and_get_binding_energy(total_energies, atom_counts):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(atom_counts, total_energies)
    e0_values = reg.coef_
    e_ref = reg.predict(atom_counts)
    binding_energies = total_energies - e_ref
    return binding_energies, e0_values

# =============================================================================
# --- Main Script ---
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MACE Analysis Script (Unified V6)")
    parser.add_argument("--mode", required=True, choices=['run', 'plot', 'collate'])
    parser.add_argument("--input", type=str)
    parser.add_argument("--output-prefix", type=str, required=True)
    
    parser.add_argument('--model', type=str, default="mace.model")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--max-edges', type=int, default=15000)
    parser.add_argument('--chunk-size', type=int, default=50000)
    
    parser.add_argument('--skip-forces', action='store_true', help='Skip force calculation.')
    parser.add_argument('--compute-stress', action='store_true')
    parser.add_argument('--layers-to-keep', type=int, default=-1)
    
    # --- LES / BEC Arguments ---
    parser.add_argument("--compute-bec", action="store_true", help="Compute Born Effective Charges")
    parser.add_argument("--les-yaml", type=str, default=None, help="Path to LES config yaml")
    parser.add_argument("--eps-inf", type=float, default=None, help="Epsilon Infinity for LES")

    parser.add_argument('--compute-binding-energy', action='store_true', help='Fit E0 and plot binding energy.')
    parser.add_argument('--plot-pca', action='store_true', help='Plot PCA maps.')
    parser.add_argument('--plot-components', action='store_true', help='Plot separate force/stress components.')
    parser.add_argument('--use-quantiles', action='store_true', help='Filter outliers (0.1%%-99.9%%).')
    parser.add_argument('--energy-key', type=str, default='energy')
    parser.add_argument('--mlp-energy-key', type=str, default=None)
    parser.add_argument('--forces-key', type=str, default='forces')
    parser.add_argument('--stress-key', type=str, default='stress')
    parser.add_argument('--mlp-forces-key', type=str, default=None)
    parser.add_argument('--mlp-stress-key', type=str, default=None)
    parser.add_argument('--virials-key', type=str, default='virials')
    parser.add_argument('--mlp-virials-key', type=str, default=None)
    parser.add_argument('--fps-k', type=int, default=0)
    parser.add_argument('--source-xyz', type=str, default=None)
    parser.add_argument('--fps-allow-elements', type=str, default=None)
    parser.add_argument('--fps-dft-max-force-min', type=float, default=None)
    parser.add_argument('--fps-dft-max-force-max', type=float, default=None)
    parser.add_argument('--export-fake-label', action='store_true', help='Export predicted labels to extxyz dataset.')

    args = parser.parse_args()

    # =========================================================================
    # --- MODE: RUN ---
    # =========================================================================
    if args.mode == 'run':
        if not args.input: parser.error("Run mode needs --input")
        
        existing_chunks = glob.glob(f"{args.output_prefix}_chunk_*_data.npz")
        if existing_chunks:
            print(f"[Warning] Found {len(existing_chunks)} existing chunk files. Ensure no mixing!")
        
        def _find_latest_model_path():
            pats = ["**/*.model", "**/*.pt", "**/*.pth", "**/*.ckpt"]
            files = []
            for p in pats:
                files.extend(glob.glob(p, recursive=True))
            files = [f for f in files if os.path.isfile(f)]
            if not files:
                return None
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0]
        
        mdl_path = args.model
        if not mdl_path or not os.path.isfile(mdl_path):
            cand_latest = _find_latest_model_path()
            if cand_latest and os.path.isfile(cand_latest):
                mdl_path = cand_latest
            else:
                cand_models = [
                    os.path.join(os.sep, "models", "mace-omat-0-small.model"),
                    os.path.join(os.sep, "models", "MACE-omol-0-extra-large-1024.model"),
                ]
                mdl_path = next((p for p in cand_models if os.path.isfile(p)), None)
        if not mdl_path or not os.path.isfile(mdl_path):
            raise FileNotFoundError("No model found. Please provide --model, ensure a trained model exists, or place a pretrained model under /models/")
        print(f"Loading model: {mdl_path}")
        try:
            with open(f"{args.output_prefix}_model.txt", "w", encoding="utf-8") as fp:
                fp.write(str(mdl_path))
        except Exception:
            pass
        model = torch.load(mdl_path, map_location=args.device)
        model = model.float() if args.dtype == "float32" else model.double()
        model.to(args.device).eval()
        
        # --- SETUP LES IF REQUESTED ---
        les_enabled = False
        if args.compute_bec or args.les_yaml is not None or args.eps_inf is not None:
            try:
                from les import Les
                les_args = {}
                if args.les_yaml is not None and os.path.exists(args.les_yaml):
                    with open(args.les_yaml, "r", encoding="utf-8") as f:
                        les_args = yaml.safe_load(f) or {}
                if args.eps_inf is not None:
                    les_args["epsilon_inf"] = float(args.eps_inf)
                if args.compute_bec:
                    les_args["compute_bec"] = True
                
                model.les = Les(les_arguments=les_args)
                les_enabled = True
                if args.compute_bec:
                    setattr(model, "compute_bec", True)
                    setattr(model, "bec_output_index", les_args.get("bec_output_index", None))
                print(">>> LES/BEC Module Attached.")
            except ImportError:
                print("[Error] requested LES/BEC but 'les' module not found.")
            except Exception as e:
                print(f"[Error] Failed to attach LES: {e}")

        # --- INSPECT MODEL SIGNATURE ---
        # This prevents TypeError if the model.forward() doesn't accept compute_bec/les
        forward_params = inspect.signature(model.forward).parameters
        supports_les_kwarg = "compute_les" in forward_params
        supports_bec_kwarg = "compute_bec" in forward_params
        supports_kwargs = "kwargs" in forward_params
        
        print(f"Model Signature Check: LES arg={supports_les_kwarg}, BEC arg={supports_bec_kwarg}, Kwargs={supports_kwargs}")

        model_z_table = [int(z) for z in model.atomic_numbers]
        
        # Auto-detect descriptor size
        try:
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_inv_feats = irreps_out.dim // (l_max + 1) ** 2
            num_layers = int(model.num_interactions)
            keep = args.layers_to_keep if args.layers_to_keep != -1 else num_layers
            layer_dims = [irreps_out.dim for _ in range(num_layers)]
            layer_dims[-1] = num_inv_feats 
            feat_dim = np.sum(layer_dims[:keep])
        except:
            feat_dim = None
            l_max = 0; num_layers = int(model.num_interactions); num_inv_feats = 0

        chunks = read_chunks(args.input, args.chunk_size)
        first_fake_write = True
        energy_warned = False
        for i, chunk in enumerate(chunks):
            print(f"Processing Chunk {i} ({len(chunk)} atoms)...")
            batches_d, batches_i = create_batches_parallel(chunk, args.max_edges, model, "Default", args.num_workers)
            
            res = {k: [] for k in ["dft_e", "mlp_e", "dft_f", "mlp_f", "dft_s", "mlp_s", 
                                   "desc", "counts", "natoms", 
                                   "charges", "vs_norms", "bec_norms", "all_z"]}
            
            comp_force = not args.skip_forces
            comp_stress = args.compute_stress
            comp_bec = args.compute_bec
            
            satellite_scheme = getattr(model, "satellite_scheme", "none")

            pbar = tqdm(total=len(batches_d), desc=f"Infer Chunk {i}", unit="batch")
            chunk_fake_imgs = []
            for bd, bi in zip(batches_d, batches_i):
                esum = int(np.sum([d.edge_index.shape[1] for d in bd]))
                pbar.set_postfix({"structures": len(bi), "edges": esum})
                pbar.update(1)
                natoms = [len(a) for a in bi]
                counts = [[sum(a.numbers == z) for z in model_z_table] for a in bi]
                
                # Basic info
                res["natoms"].extend(natoms)
                res["counts"].extend(counts)

                missing_energy = 0
                for a, n in zip(bi, natoms):
                    e_tot = get_total_energy(a, args.energy_key) if args.energy_key else get_total_energy(a, "energy")
                    if e_tot is None:
                        res["dft_e"].append(np.nan)
                        missing_energy += 1
                    else:
                        res["dft_e"].append(float(e_tot) / float(n))
                if missing_energy and (not energy_warned):
                    avail = suggest_energy_keys(bi[0]) if bi else tuple()
                    msg = f"[Warning] Missing/invalid energies for {missing_energy}/{len(bi)} structures. "
                    if args.energy_key:
                        msg += f"Requested key='{args.energy_key}'. "
                    if avail:
                        msg += f"Available energy-like keys: {', '.join(avail)}. "
                    msg += "Use --energy-key to match your extxyz metadata."
                    print(msg)
                    energy_warned = True

                if comp_force:
                    for a in bi:
                        fval = None
                        if args.forces_key:
                            if hasattr(a, "arrays") and args.forces_key in a.arrays:
                                fval = a.arrays[args.forces_key]
                            else:
                                fval = a.info.get(args.forces_key)
                        if fval is None:
                            try:
                                fval = a.get_forces()
                            except:
                                fval = None
                        if fval is not None:
                            res["dft_f"].append(fval)
                if comp_stress:
                    st_list = []
                    for a in bi:
                        sval = None
                        if args.stress_key:
                            sval = a.info.get(args.stress_key)
                        # Virials优先（如果提供），否则按应力逻辑
                        if sval is None and args.virials_key:
                            sval = a.info.get(args.virials_key)
                        if sval is None:
                            try:
                                sval = a.get_stress(voigt=False)
                            except:
                                sval = None
                        if sval is None:
                            continue
                        sval = np.array(sval)
                        if sval.shape == (6,):
                            v = sval
                            sval = np.array([[v[0], v[5], v[4]],[v[5], v[1], v[3]],[v[4], v[3], v[2]]], dtype=float)
                        st_list.append(sval)
                    if st_list:
                        res["dft_s"].extend(st_list)
                    else:
                        comp_stress = False

                loader = torch_geometric.dataloader.DataLoader(dataset=bd, batch_size=len(bd), shuffle=False)
                batch = next(iter(loader)).to(args.device)
                batch_dict = batch.to_dict()
                
                # FIX: Read Z from ASE Atoms
                batch_z = np.concatenate([a.numbers for a in bi])
                res["all_z"].append(batch_z)
                
                grad_needed = (comp_force or comp_stress)
                with torch.set_grad_enabled(grad_needed):
                    
                    # --- CONSTRUCT SAFE CALL ARGS ---
                    call_kwargs = {}
                    if "compute_force" in forward_params or supports_kwargs:
                        call_kwargs["compute_force"] = comp_force
                    if "compute_stress" in forward_params or supports_kwargs:
                        call_kwargs["compute_stress"] = comp_stress
                    
                    # Only pass LES args if the model EXPLICITLY accepts them or has **kwargs
                    if les_enabled and (supports_les_kwarg or supports_kwargs):
                        call_kwargs["compute_les"] = True
                    if les_enabled and comp_bec and (supports_bec_kwarg or supports_kwargs):
                        call_kwargs["compute_bec"] = True
                        
                    # 1. Run Standard Model
                    out = model(batch_dict, **call_kwargs)
                    
                    # 2. Manual LES Fallback (If enabled but model didn't accept the args)
                    # Note: We check if output missing keys despite us wanting them
                    if les_enabled and "latent_charges" not in out:
                        try:
                            # Try calling model.les manually if it exists
                            if getattr(model, "les", None) is not None:
                                # Standard LES module often takes (data, output) or (data)
                                # We try passing batch_dict and the current output
                                # This handles the case where ScaleShiftMACE doesn't call self.les()
                                les_out = model.les(batch_dict, out)
                                if les_out: out.update(les_out)
                        except Exception as e_les:
                            # Silent fail to avoid crashing whole script, but print once?
                            pass

                    # --- Extract Results ---
                    ek = args.mlp_energy_key if args.mlp_energy_key else 'energy'
                    eout = out.get(ek)
                    if eout is None and ek != 'energy':
                        eout = out.get('energy')
                    energies = eout.detach().cpu().numpy() if eout is not None else torch.zeros(len(natoms)).cpu().numpy()
                    res["mlp_e"].extend([e/n for e, n in zip(energies, natoms)])

                    if comp_force:
                        fk = args.mlp_forces_key if args.mlp_forces_key else 'forces'
                        fp = out.get(fk)
                        if fp is None and fk != 'forces':
                            fp = out.get('forces')
                        f_pred = fp.detach().cpu().numpy() if fp is not None else np.zeros((batch_dict["positions"].shape[0], 3))
                        ptr = batch.ptr.cpu().numpy()
                        for j in range(len(ptr)-1): res["mlp_f"].append(f_pred[ptr[j]:ptr[j+1]])
                    
                    if comp_stress:
                        sk = args.mlp_stress_key if args.mlp_stress_key else 'stress'
                        sp = out.get(sk)
                        if sp is None and sk != 'stress':
                            sp = out.get('stress')
                        # Virials回退：如果模型提供virials键则使用
                        if sp is None:
                            vk = args.mlp_virials_key if args.mlp_virials_key else 'virials'
                            sp = out.get(vk)
                        if sp is not None:
                            res["mlp_s"].extend(sp.detach().cpu().numpy())
                    
                    # --- Export Fake Label Images ---
                    if args.export_fake_label:
                        def to_voigt(m):
                            m = np.array(m)
                            if m.shape == (3, 3):
                                return np.array([m[0,0], m[1,1], m[2,2], m[1,2], m[0,2], m[0,1]], dtype=float)
                            if m.shape == (6,):
                                return m.astype(float)
                            return None
                        ptr = batch.ptr.cpu().numpy()
                        sp_np = None
                        try:
                            if comp_stress and sp is not None:
                                sp_np = sp.detach().cpu().numpy()
                        except Exception:
                            sp_np = None
                        for j in range(len(ptr)-1):
                            a = bi[j].copy()
                            e_j = float(energies[j]) if len(energies) >= (j+1) else None
                            f_j = None
                            s_j = None
                            include_forces = comp_force and ('f_pred' in locals()) and (f_pred is not None)
                            include_stress = comp_stress and (sp_np is not None)
                            if include_forces:
                                f_slice = f_pred[ptr[j]:ptr[j+1]]
                                if f_slice is not None and f_slice.size > 0:
                                    f_j = np.array(f_slice, dtype=float)
                            if include_stress:
                                try:
                                    s_raw = sp_np[j]
                                    s_j = to_voigt(s_raw)
                                except Exception:
                                    s_j = None
                            if e_j is not None:
                                a.info["energy"] = e_j
                            if f_j is not None:
                                a.arrays["forces"] = f_j
                            if s_j is not None:
                                a.info["stress"] = s_j
                            chunk_fake_imgs.append(a)

                    # --- LES Outputs ---
                    q = out.get("latent_charges", None)
                    if q is not None:
                        q_np = q.detach().cpu().numpy().flatten().astype(np.float64)
                        res["charges"].append(q_np)
                    
                    vs = out.get("virtual_sites", None)
                    if vs is not None:
                        vs_np = vs.detach().cpu().numpy().astype(np.float64)
                        vs_norms = np.linalg.norm(vs_np, axis=1)
                        res["vs_norms"].append(vs_norms)
                    
                    bec = out.get("BEC", None)
                    if bec is not None:
                        bec_t = bec.detach().cpu()
                        N_nodes = int(batch.num_nodes)
                        M_bec = int(bec_t.shape[0]) if bec_t.dim() >= 2 else 0
                        bec_flat = bec_t.reshape(M_bec, -1).numpy().astype(np.float64) if M_bec > 0 else bec_t.view(1, -1).numpy()
                        
                        final_bec = None
                        if M_bec == N_nodes:
                            final_bec = bec_flat
                        elif satellite_scheme == "core_satellite" and M_bec == 2 * N_nodes:
                            final_bec = bec_flat[:N_nodes] + bec_flat[N_nodes:]
                        elif satellite_scheme == "core_plus_dipole_pair" and M_bec == 3 * N_nodes:
                            final_bec = bec_flat[:N_nodes] + bec_flat[N_nodes:2*N_nodes] + bec_flat[2*N_nodes:3*N_nodes]
                        elif satellite_scheme == "dipole_pair" and M_bec == 2 * N_nodes:
                            final_bec = bec_flat[:N_nodes] + bec_flat[N_nodes:]
                        else:
                            final_bec = bec_flat[:N_nodes]
                        
                        if final_bec is not None:
                            bec_m = final_bec.reshape(-1, 3, 3)
                            bec_norm = np.linalg.norm(bec_m, axis=(1, 2))
                            res["bec_norms"].append(bec_norm)

                    # Descriptors
                    node_feats = out['node_feats'].detach()
                    invs = extract_invariant(node_feats, num_layers=num_layers, num_features=num_inv_feats, l_max=l_max)
                    if feat_dim: invs = invs[:, :feat_dim]
                    
                    if HAS_TORCH_SCATTER:
                        pooled = scatter_mean(invs, batch.batch, dim=0).cpu().numpy()
                    else:
                        grp = batch.batch.cpu().numpy()
                        invs_np = invs.cpu().numpy()
                        pooled = np.array([invs_np[grp==k].mean(0) for k in range(len(bi))])
                    res["desc"].extend(pooled)

            pbar.close()
            np_save = {
                "dft_energies": np.array(res["dft_e"]),
                "mlp_energies": np.array(res["mlp_e"]),
                "mlp_descriptors": np.array(res["desc"]),
                "atom_counts": np.array(res["counts"], dtype=int),
                "num_atoms": np.array(res["natoms"], dtype=int),
                "z_numbers": np.array(model_z_table, dtype=int),
            }
            if comp_force and res["mlp_f"]:
                np_save["dft_forces"] = np.concatenate(res["dft_f"])
                np_save["mlp_forces"] = np.concatenate(res["mlp_f"])
            if comp_stress and res["mlp_s"]:
                np_save["dft_stresses"] = np.array(res["dft_s"])
                np_save["mlp_stresses"] = np.array(res["mlp_s"])
            
            if res["charges"]:
                np_save["charges"] = np.concatenate(res["charges"])
            if res["vs_norms"]:
                np_save["vs_norms"] = np.concatenate(res["vs_norms"])
            if res["bec_norms"]:
                np_save["bec_norms"] = np.concatenate(res["bec_norms"])
            if res["all_z"]:
                np_save["all_z"] = np.concatenate(res["all_z"])

            np.savez_compressed(f"{args.output_prefix}_chunk_{i}_data.npz", **np_save)
            if args.export_fake_label and len(chunk_fake_imgs) > 0:
                try:
                    write(f"{args.output_prefix}_fake_label.xyz", chunk_fake_imgs, format="extxyz", append=not first_fake_write)
                    first_fake_write = False
                except Exception as e:
                    print(f"[Warning] Saving fake label dataset for chunk {i} failed: {e}")
        print("Run complete. Data saved.")
        if args.export_fake_label:
            if first_fake_write:
                print("[Warning] No structures exported to fake label dataset.")
            else:
                print(f"Fake label dataset saved to {args.output_prefix}_fake_label.xyz")

    # =========================================================================
    # --- MODE: PLOT ---
    # =========================================================================
    elif args.mode == 'plot':
        files = []
        if args.input and os.path.exists(args.input) and args.input.lower().endswith(".npz"):
            files = [args.input]
        else:
            files = sorted(glob.glob(f"{args.output_prefix}_chunk_*_data.npz"))
        if not files: print("No data found."); return

        print(f"Loading {len(files)} chunks...")
        d_e, m_e, descs, counts, natoms = [], [], [], [], []
        d_f, m_f, d_s, m_s = [], [], [], []
        all_charges, all_vs, all_bec, all_z = [], [], [], []
        z_numbers = None
        has_f, has_s = True, True

        for f in tqdm(files):
            data = np.load(f)
            de_key = 'dft_energies'
            me_key = 'mlp_energies'
            if de_key in data and me_key in data:
                d_e.append(data[de_key]); m_e.append(data[me_key])
            else:
                alt_d = data[args.energy_key] if args.energy_key in data else None
                alt_m_key = args.mlp_energy_key if args.mlp_energy_key else args.energy_key
                alt_m = data[alt_m_key] if alt_m_key in data else None
                if alt_d is not None and alt_m is not None:
                    d_e.append(alt_d); m_e.append(alt_m)
                else:
                    continue
            descs.append(data['mlp_descriptors'])
            if 'atom_counts' in data: counts.append(data['atom_counts'])
            if 'num_atoms' in data: natoms.append(data['num_atoms'])
            if 'z_numbers' in data and z_numbers is None: z_numbers = data['z_numbers']
            
            if 'charges' in data: all_charges.append(data['charges'])
            if 'vs_norms' in data: all_vs.append(data['vs_norms'])
            if 'bec_norms' in data: all_bec.append(data['bec_norms'])
            if 'all_z' in data: all_z.append(data['all_z'])

            if 'dft_forces' in data and data['dft_forces'].size > 0: 
                d_f.append(data['dft_forces']); m_f.append(data['mlp_forces'])
            else: has_f = False
            
            if 'dft_stresses' in data and data['dft_stresses'].size > 0:
                d_s.append(data['dft_stresses']); m_s.append(data['mlp_stresses'])
            else: has_s = False
            
            if not d_f and ('forces' in data or args.forces_key in data):
                alt_df = data[args.forces_key] if args.forces_key in data else data['forces']
                alt_mfk = args.mlp_forces_key if args.mlp_forces_key else 'mlp_forces'
                if alt_mfk in data:
                    d_f.append(alt_df); m_f.append(data[alt_mfk]); has_f = True
            if not d_s:
                # 优先stress，其次virials
                alt_ds = None
                if 'stress' in data or args.stress_key in data:
                    alt_ds = data[args.stress_key] if args.stress_key in data else data['stress']
                elif 'virials' in data or args.virials_key in data:
                    alt_ds = data[args.virials_key] if args.virials_key in data else data['virials']
                if alt_ds is not None:
                    alt_msk = None
                    # 模型侧优先mlp_stresses，否则mlp_virials
                    if 'mlp_stresses' in data:
                        alt_msk = data['mlp_stresses']
                    elif args.mlp_stress_key and args.mlp_stress_key in data:
                        alt_msk = data[args.mlp_stress_key]
                    elif 'mlp_virials' in data:
                        alt_msk = data['mlp_virials']
                    elif args.mlp_virials_key and args.mlp_virials_key in data:
                        alt_msk = data[args.mlp_virials_key]
                    if alt_msk is not None:
                        d_s.append(alt_ds); m_s.append(alt_msk); has_s = True

        dft_e_pa = np.concatenate(d_e)
        mlp_e_pa = np.concatenate(m_e)
        descriptors = np.concatenate(descs)
        
        print("-" * 50)
        print("DATA LOADED. STARTING ANALYSIS...")
        
        # --- Binding Energy Logic (additive outputs) ---
        plot_dft_e_total, plot_mlp_e_total = dft_e_pa, mlp_e_pa
        have_binding = False
        dft_bind_pa = None
        mlp_bind_pa = None
        e0s = None
        if args.compute_binding_energy and counts:
            print("\n>>> Computing Binding Energies (Fitting E0)...")
            all_counts = np.concatenate(counts)
            all_natoms = np.concatenate(natoms)
            dft_e_total = dft_e_pa * all_natoms
            try:
                dft_bind_total, e0s = fit_e0_and_get_binding_energy(dft_e_total, all_counts)
                dft_bind_pa = dft_bind_total / all_natoms
                mlp_e_total = mlp_e_pa * all_natoms
                e_ref = np.dot(all_counts, e0s)
                mlp_bind_total = mlp_e_total - e_ref
                mlp_bind_pa = mlp_bind_total / all_natoms
                have_binding = True
                # Save E0s markdown table
                try:
                    if z_numbers is not None and e0s is not None:
                        lines = ["| Element | Z | E0 (eV) |", "|---|---:|---:|"]
                        for z, e0 in zip(z_numbers, e0s):
                            lines.append(f"| {chemical_symbols[int(z)]} | {int(z)} | {float(e0):.6f} |")
                        with open(f"{args.output_prefix}_e0s.md", "w", encoding="utf-8") as fp_e0:
                            fp_e0.write("\n".join(lines) + "\n")
                except Exception as e_save:
                    print(f"[Warning] Saving E0s table failed: {e_save}")
            except Exception as e:
                print(f"[Error] E0 fitting failed: {e}. Continue with Total Energy only.")

        # --- Parity Plots (Energy: additive total + binding) ---
        print("\n>>> Generating Parity Plots...")
        met_e_total = calculate_metrics(plot_dft_e_total, plot_mlp_e_total)
        print(f"  [Energy-Total] MAE={met_e_total['mae']*1e3:.2f} meV/atom, R2={met_e_total['r2']:.4f}")
        make_parity_plot(plot_dft_e_total, plot_mlp_e_total, "Total Energy (eV/atom)", 
                         "MACE Total Energy (eV/atom)", "DFT Total Energy (eV/atom)",
                         met_e_total, f"{args.output_prefix}_energy_total.png", use_quantiles=args.use_quantiles)
        if have_binding and dft_bind_pa is not None and mlp_bind_pa is not None:
            met_e_bind = calculate_metrics(dft_bind_pa, mlp_bind_pa)
            print(f"  [Energy-Binding] MAE={met_e_bind['mae']*1e3:.2f} meV/atom, R2={met_e_bind['r2']:.4f}")
            make_parity_plot(dft_bind_pa, mlp_bind_pa, "Binding Energy (eV/atom)", 
                             "MACE Binding Energy (eV/atom)", "DFT Binding Energy (eV/atom)",
                             met_e_bind, f"{args.output_prefix}_energy_binding.png", use_quantiles=args.use_quantiles)

        if has_f and d_f:
            d_f_v, m_f_v = np.concatenate(d_f), np.concatenate(m_f)
            met_f = calculate_metrics(d_f_v.flatten(), m_f_v.flatten())
            print(f"  [Force]  MAE={met_f['mae']*1e3:.2f} meV/A")
            if args.plot_components:
                idx = np.random.choice(len(d_f_v), min(50000, len(d_f_v)), replace=False)
                make_parity_plot(d_f_v[idx], m_f_v[idx], "Force Components", "MACE F (eV/Å)", "DFT F (eV/Å)", met_f, 
                                 f"{args.output_prefix}_force.png", labels=["x","y","z"], colors=['r','g','b'], use_quantiles=args.use_quantiles)
                make_parity_plot(d_f_v[idx].reshape(-1, 3), m_f_v[idx].reshape(-1, 3), "Force (Density)", "MACE F (eV/Å)", "DFT F (eV/Å)", met_f, 
                                 f"{args.output_prefix}_force_density.png", use_quantiles=args.use_quantiles)
            else:
                make_parity_plot(np.linalg.norm(d_f_v, axis=1), np.linalg.norm(m_f_v, axis=1), 
                                 "Force Norm", "MACE |F| (eV/Å)", "DFT |F| (eV/Å)", met_f, f"{args.output_prefix}_force.png", use_quantiles=args.use_quantiles)

        if has_s and d_s:
            d_s_t, m_s_t = np.concatenate(d_s), np.concatenate(m_s)
            met_s = calculate_metrics(d_s_t.flatten(), m_s_t.flatten())
            print(f"  [Stress] MAE={met_s['mae']*1e3:.2f} meV/A^3")
            
            if args.plot_components:
                comp_indices = [(0,0), (1,1), (2,2), (1,2), (0,2), (0,1)]
                labels = ["$\sigma_{xx}$", "$\sigma_{yy}$", "$\sigma_{zz}$", "$\sigma_{yz}$", "$\sigma_{xz}$", "$\sigma_{xy}$"]
                colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
                
                N = d_s_t.shape[0]
                d_s_comp = np.zeros((N, 6))
                m_s_comp = np.zeros((N, 6))
                for i, (r, c) in enumerate(comp_indices):
                    d_s_comp[:, i] = d_s_t[:, r, c]
                    m_s_comp[:, i] = m_s_t[:, r, c]
                
                if N > 100000:
                    idx = np.random.choice(N, 100000, replace=False)
                    d_plot, m_plot = d_s_comp[idx], m_s_comp[idx]
                else:
                    d_plot, m_plot = d_s_comp, m_s_comp
                
                make_parity_plot(d_plot, m_plot, "Stress Components", "MACE Stress (eV/Å^3)", "DFT Stress (eV/Å^3)", 
                                 met_s, f"{args.output_prefix}_stress.png", 
                                 labels=labels, colors=colors, use_quantiles=args.use_quantiles)
                make_parity_plot(d_plot, m_plot, "Stress (Density)", "MACE Stress (eV/Å^3)", "DFT Stress (eV/Å^3)", 
                                 met_s, f"{args.output_prefix}_stress_density.png", use_quantiles=args.use_quantiles)
            else:
                make_parity_plot(d_s_t.flatten(), m_s_t.flatten(), "Stress", "MACE (eV/Å^3)", "DFT (eV/Å^3)", 
                                 met_s, f"{args.output_prefix}_stress.png", use_quantiles=args.use_quantiles)

        if all_z:
            z_per_atom = np.concatenate(all_z)
            if all_charges:
                print("\n>>> Plotting Charge Distributions...")
                charges_flat = np.concatenate(all_charges)
                structure_sums = []
                curr = 0
                all_ns = np.concatenate(natoms)
                for n in all_ns:
                    structure_sums.append(np.sum(charges_flat[curr:curr+n]))
                    curr += n
                plot_elemental_distribution(
                    charges_flat, z_per_atom, title="Atomic Charge Distribution", xlabel="Charge (e)",
                    filename=f"{args.output_prefix}_charges.png",
                    extra_data=np.array(structure_sums), extra_label="Total Structure Charge"
                )
            if all_vs:
                print("\n>>> Plotting Virtual Site Displacements...")
                vs_flat = np.concatenate(all_vs)
                plot_elemental_distribution(
                    vs_flat, z_per_atom, title="Virtual Site Displacement", xlabel="|Displacement| (Å)",
                    filename=f"{args.output_prefix}_virtual_sites.png"
                )
            if all_bec:
                print("\n>>> Plotting BEC Norms...")
                bec_flat = np.concatenate(all_bec)
                plot_elemental_distribution(
                    bec_flat, z_per_atom, title="BEC Frobenius Norm", xlabel="Frobenius Norm",
                    filename=f"{args.output_prefix}_bec.png"
                )

        # --- PCA Analysis (UPDATED & additive) ---
        if args.plot_pca:
            print("\n>>> Running PCA Analysis...")
            pca = PCA(n_components=2)
            # 建议：如果数据量巨大，可以先 sample 一部分做 fit，再 transform 全部
            # 这里保持原样
            pca_coords = pca.fit_transform(descriptors)
            var = pca.explained_variance_ratio_
            print(f"  PCA Explained Variance: PC1={var[0]:.2%}, PC2={var[1]:.2%}")
            
            xlab = f"PC1 ({var[0]:.2%})"
            ylab = f"PC2 ({var[1]:.2%})"
            
            # 为了绘图速度，采样 10w 个点
            if len(pca_coords) > 100000:
                idx = np.random.choice(len(pca_coords), 100000, replace=False)
                xy = pca_coords[idx]
                c_en_total = plot_dft_e_total[idx]
                c_en_bind = dft_bind_pa[idx] if have_binding and dft_bind_pa is not None else None
                err_total = (plot_mlp_e_total - plot_dft_e_total)[idx]
                err_bind = (mlp_bind_pa - dft_bind_pa)[idx] if have_binding and dft_bind_pa is not None else None
            else:
                xy = pca_coords
                c_en_total = plot_dft_e_total
                c_en_bind = dft_bind_pa if have_binding and dft_bind_pa is not None else None
                err_total = (plot_mlp_e_total - plot_dft_e_total)
                err_bind = (mlp_bind_pa - dft_bind_pa) if have_binding and dft_bind_pa is not None else None

            # === 核心修改：计算鲁棒的视野范围 (IQR 方法) ===
            # 不管 args.use_quantiles 是否开启，我们都计算这个范围，
            # 只有当开启时才应用，或者你可以强制应用。
            # 这里逻辑是：只要开启 use_quantiles，就用 IQR 强力缩放。
            
            axis_xlims = None
            axis_ylims = None
            
            if args.use_quantiles:
                # 1. 计算四分位数 (25% 和 75%)
                q1 = np.quantile(xy, 0.25, axis=0)
                q3 = np.quantile(xy, 0.75, axis=0)
                iqr = q3 - q1
                
                # 2. 定义视野范围 (Tukey's Fences 标准: 1.5倍 IQR)
                # 稍微放宽到 2.0 倍 IQR，既能去离群点，又能保留边缘的 valid data
                factor = 2.0 
                x_min, x_max = q1[0] - factor * iqr[0], q3[0] + factor * iqr[0]
                y_min, y_max = q1[1] - factor * iqr[1], q3[1] + factor * iqr[1]
                
                axis_xlims = (x_min, x_max)
                axis_ylims = (y_min, y_max)
                print(f"  [PCA] Applying IQR zoom: X={axis_xlims}, Y={axis_ylims}")
            # ================================================

            # 1. Density Plot
            fig, ax = plt.subplots(figsize=(7, 6))
            try:
                # 为了 KDE 不被离群点干扰，我们只用视野内的数据计算 KDE
                if axis_xlims:
                    mask = (xy[:,0] > axis_xlims[0]) & (xy[:,0] < axis_xlims[1]) & \
                           (xy[:,1] > axis_ylims[0]) & (xy[:,1] < axis_ylims[1])
                    xy_kde = xy[mask]
                    # 如果 mask 后点太少，就回退
                    if len(xy_kde) < 100: xy_kde = xy
                else:
                    xy_kde = xy
                    
                # 再次采样以加速 KDE
                if len(xy_kde) > 5000:
                     idx_kde = np.random.choice(len(xy_kde), 5000, replace=False)
                     xy_kde = xy_kde[idx_kde]

                z = gaussian_kde(xy_kde.T)(xy_kde.T)
                # 这里画图还是画所有的 xy (但在视野外的会被自动切掉)
                # 注意：我们需要给原始 xy 匹配颜色，这比较麻烦。
                # 简单做法：还是对采样后的 xy 计算 KDE 并绘图
                
                # 重新计算用于绘图的 density (基于全部 xy 的采样)
                z_all = gaussian_kde(xy_kde.T)(xy.T) 
                sort_idx = z_all.argsort()
                sc = ax.scatter(xy[sort_idx,0], xy[sort_idx,1], c=z_all[sort_idx], s=5, cmap='viridis', rasterized=True)
                plt.colorbar(sc, label='Density')
            except Exception as e:
                print(f"[Warning] PCA Density KDE failed: {e}. Using plain scatter.")
                ax.scatter(xy[:,0], xy[:,1], s=5, alpha=0.5, c='steelblue', rasterized=True)
            
            # 强制应用视野
            if axis_xlims: ax.set_xlim(axis_xlims)
            if axis_ylims: ax.set_ylim(axis_ylims)
            
            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
            ax.set_title("Descriptor Space (Density)")
            plt.savefig(f"{args.output_prefix}_pca_density.png", dpi=300); plt.close()

            # 2. Energy Plot(s)
            fig, ax = plt.subplots(figsize=(7, 6))
            
            # 颜色映射范围：使用更严格的 5% - 95%，忽略极高能/极低能的噪点
            if args.use_quantiles:
                v_min, v_max = np.quantile(c_en_total, [0.05, 0.95])
            else:
                v_min, v_max = c_en_total.min(), c_en_total.max()
            
            sc = ax.scatter(xy[:,0], xy[:,1], c=c_en_total, 
                            s=10, # 稍微加大一点点点的大小
                            cmap='plasma', vmin=v_min, vmax=v_max, alpha=0.8, rasterized=True)
            
            cbar = plt.colorbar(sc, label="Total Energy (eV/atom)")
            
            # 强制应用视野
            if axis_xlims: ax.set_xlim(axis_xlims)
            if axis_ylims: ax.set_ylim(axis_ylims)

            ax.set_xlabel(xlab); ax.set_ylabel(ylab)
            ax.set_title(f"Descriptor Space (Total Energy)")
            plt.savefig(f"{args.output_prefix}_pca_energy_total.png", dpi=300); plt.close()
            # Backward compatible single file
            try:
                fig, ax = plt.subplots(figsize=(7, 6))
                sc = ax.scatter(xy[:,0], xy[:,1], c=c_en_total, 
                                s=10, cmap='plasma', vmin=v_min, vmax=v_max, alpha=0.8, rasterized=True)
                plt.colorbar(sc, label="Total Energy (eV/atom)")
                if axis_xlims: ax.set_xlim(axis_xlims)
                if axis_ylims: ax.set_ylim(axis_ylims)
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                ax.set_title(f"Descriptor Space (Total Energy)")
                plt.savefig(f"{args.output_prefix}_pca_energy.png", dpi=300); plt.close()
            except Exception:
                pass
            if c_en_bind is not None:
                fig, ax = plt.subplots(figsize=(7, 6))
                if args.use_quantiles:
                    v_min_b, v_max_b = np.quantile(c_en_bind, [0.05, 0.95])
                else:
                    v_min_b, v_max_b = c_en_bind.min(), c_en_bind.max()
                sc = ax.scatter(xy[:,0], xy[:,1], c=c_en_bind, 
                                s=10, cmap='plasma', vmin=v_min_b, vmax=v_max_b, alpha=0.8, rasterized=True)
                plt.colorbar(sc, label="Binding Energy (eV/atom)")
                if axis_xlims: ax.set_xlim(axis_xlims)
                if axis_ylims: ax.set_ylim(axis_ylims)
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                ax.set_title("Descriptor Space (Binding Energy)")
                plt.savefig(f"{args.output_prefix}_pca_energy_binding.png", dpi=300); plt.close()
            # 3. Energy Error Plot(s)
            def plot_error_map(err_vals, fname, label):
                fig, ax = plt.subplots(figsize=(7, 6))
                abs_err = np.abs(err_vals)
                if args.use_quantiles:
                    vmin, vmax = np.quantile(abs_err, [0.05, 0.95])
                else:
                    vmin, vmax = 0.0, float(np.max(abs_err))
                sc = ax.scatter(xy[:,0], xy[:,1], c=abs_err, s=10, cmap='plasma', vmin=vmin, vmax=vmax, alpha=0.85, rasterized=True)
                plt.colorbar(sc, label=label)
                if axis_xlims: ax.set_xlim(axis_xlims)
                if axis_ylims: ax.set_ylim(axis_ylims)
                ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                ax.set_title("Descriptor Space (Absolute Energy Error)")
                plt.savefig(fname, dpi=300); plt.close()
            plot_error_map(err_total, f"{args.output_prefix}_pca_energy_error_total.png", "|ΔE_total| (eV/atom)")
            if err_bind is not None:
                plot_error_map(err_bind, f"{args.output_prefix}_pca_energy_error_binding.png", "|ΔE Binding| (eV/atom)")
            # Force error maps (mean and max per-structure)
            try:
                if has_f and d_f and m_f and natoms:
                    d_f_v = np.concatenate(d_f)
                    m_f_v = np.concatenate(m_f)
                    all_ns = np.concatenate(natoms)
                    offs = np.cumsum(np.concatenate([[0], all_ns]))
                    err_mean = np.zeros(len(all_ns), dtype=float)
                    err_max = np.zeros(len(all_ns), dtype=float)
                    for i in range(len(all_ns)):
                        seg = slice(offs[i], offs[i+1])
                        if offs[i+1] > offs[i]:
                            ev = np.linalg.norm(m_f_v[seg] - d_f_v[seg], axis=1)
                            err_mean[i] = float(np.mean(ev))
                            err_max[i] = float(np.max(ev))
                    mean_vals = err_mean[idx] if len(pca_coords) > 100000 else err_mean
                    max_vals = err_max[idx] if len(pca_coords) > 100000 else err_max
                    # Mean error map
                    fig, ax = plt.subplots(figsize=(7, 6))
                    if args.use_quantiles:
                        vmin_m, vmax_m = np.quantile(mean_vals, [0.05, 0.95])
                    else:
                        vmin_m, vmax_m = 0.0, float(np.max(mean_vals))
                    sc = ax.scatter(xy[:,0], xy[:,1], c=mean_vals, s=10, cmap='plasma', vmin=vmin_m, vmax=vmax_m, alpha=0.85, rasterized=True)
                    plt.colorbar(sc, label="Mean Atomic Force Abs Error (eV/Å)")
                    if axis_xlims: ax.set_xlim(axis_xlims)
                    if axis_ylims: ax.set_ylim(axis_ylims)
                    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                    ax.set_title("Descriptor Space (Mean Atomic Force Abs Error)")
                    plt.savefig(f"{args.output_prefix}_pca_force_error_mean.png", dpi=300); plt.close()
                    # Max error map
                    fig, ax = plt.subplots(figsize=(7, 6))
                    if args.use_quantiles:
                        vmin_x, vmax_x = np.quantile(max_vals, [0.05, 0.95])
                    else:
                        vmin_x, vmax_x = 0.0, float(np.max(max_vals))
                    sc = ax.scatter(xy[:,0], xy[:,1], c=max_vals, s=10, cmap='plasma', vmin=vmin_x, vmax=vmax_x, alpha=0.85, rasterized=True)
                    plt.colorbar(sc, label="Max Atomic Force Abs Error (eV/Å)")
                    if axis_xlims: ax.set_xlim(axis_xlims)
                    if axis_ylims: ax.set_ylim(axis_ylims)
                    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                    ax.set_title("Descriptor Space (Max Atomic Force Abs Error)")
                    plt.savefig(f"{args.output_prefix}_pca_force_error_max.png", dpi=300); plt.close()
            except Exception as e:
                print(f"[Warning] PCA Force Error plots failed: {e}")
            print("  PCA plots saved (Density, Energy total/binding, and Error maps).")

            if args.fps_k and args.fps_k > 0:
                from kit.fps import FpsSelector
                cand_idx = np.arange(descriptors.shape[0])
                if args.fps_allow_elements and counts and z_numbers is not None:
                    tokens = [t.strip() for t in str(args.fps_allow_elements).split(",") if t.strip()]
                    allow_z = []
                    for t in tokens:
                        if t.isdigit():
                            try:
                                allow_z.append(int(t))
                            except:
                                pass
                        else:
                            try:
                                allow_z.append(int(np.where(np.array(chemical_symbols) == t)[0][0]))
                            except:
                                pass
                    if allow_z:
                        z_arr = np.array(z_numbers, dtype=int)
                        idx_map = [int(np.where(z_arr == z)[0][0]) for z in allow_z if np.any(z_arr == z)]
                        all_counts = np.concatenate(counts)
                        has_elem = np.array([np.sum(row[idx_map]) > 0 for row in all_counts], dtype=bool)
                        cand_idx = cand_idx[has_elem]
                fmin = args.fps_dft_max_force_min
                fmax = args.fps_dft_max_force_max
                if (fmin is not None or fmax is not None) and d_f and natoms:
                    all_n = np.concatenate(natoms)
                    d_all = np.concatenate(d_f)
                    norms = np.linalg.norm(d_all, axis=1)
                    offs = np.cumsum(np.concatenate([[0], all_n]))
                    max_per = np.array([norms[offs[i]:offs[i+1]].max() if offs[i+1] > offs[i] else 0.0 for i in range(len(all_n))])
                    mask = np.ones_like(max_per, dtype=bool)
                    if fmin is not None:
                        mask &= (max_per >= float(fmin))
                    if fmax is not None:
                        mask &= (max_per <= float(fmax))
                    cand_idx = cand_idx[mask[cand_idx]]
                if cand_idx.size == 0:
                    cand_idx = np.arange(descriptors.shape[0])
                selector = FpsSelector(descriptors[cand_idx])
                rel = selector.select_k(int(args.fps_k))
                sel_idx = cand_idx[rel]
                if sel_idx is not None and sel_idx.size > 0:
                    np.save(f"{args.output_prefix}_fps_indices.npy", sel_idx)
                    try:
                        sel_desc = descriptors[sel_idx]
                        np.save(f"{args.output_prefix}_fps_descriptors.npy", sel_desc)
                    except Exception as e:
                        print(f"[Warning] Saving FPS descriptors failed: {e}")
                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.scatter(pca_coords[:,0], pca_coords[:,1], s=5, alpha=0.3, c='lightgray', rasterized=True)
                    ax.scatter(pca_coords[sel_idx,0], pca_coords[sel_idx,1], s=25, alpha=0.9, c='orange', edgecolors='k', linewidths=0.3, rasterized=True)
                    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
                    ax.set_title("Descriptor PCA with FPS Selection")
                    if axis_xlims: ax.set_xlim(axis_xlims)
                    if axis_ylims: ax.set_ylim(axis_ylims)
                    plt.savefig(f"{args.output_prefix}_pca_fps.png", dpi=300)
                    plt.close()
                    print("  FPS PCA overlay saved.")
                    src_xyz = None
                    if args.source_xyz and os.path.exists(args.source_xyz):
                        src_xyz = args.source_xyz
                    elif args.input and args.input.lower().endswith((".xyz", ".extxyz")) and os.path.exists(args.input):
                        src_xyz = args.input
                    if src_xyz:
                        try:
                            imgs = list(iread(src_xyz))
                        except:
                            imgs = read(src_xyz, index=":")
                        if imgs:
                            n_total = len(imgs)
                            if n_total != descriptors.shape[0]:
                                print(f"[Warning] Source XYZ structures ({n_total}) do not match descriptors count ({descriptors.shape[0]}). Exporting by index bounds only.")
                            idx_safe = [i for i in sel_idx if 0 <= i < n_total]
                            selected_imgs = [imgs[i] for i in idx_safe]
                            write(f"{args.output_prefix}_fps.xyz", selected_imgs, format="extxyz")
                            print("  FPS xyz dataset saved (format=extxyz).")
        print("-" * 50)
        sys.stdout.flush()

    # =========================================================================
    # --- MODE: COLLATE ---
    # =========================================================================
    elif args.mode == 'collate':
        files = sorted(glob.glob(f"{args.output_prefix}_chunk_*_data.npz"))
        collated = {}
        keys = ["dft_energies", "mlp_energies", "mlp_descriptors", "atom_counts", "num_atoms", 
                "dft_forces", "mlp_forces", "dft_stresses", "mlp_stresses",
                "charges", "vs_norms", "bec_norms", "all_z"]
        for f in tqdm(files):
            d = np.load(f)
            for k in keys:
                if k in d:
                    if k not in collated: collated[k] = []
                    collated[k].append(d[k])
        for k in collated: collated[k] = np.concatenate(collated[k], axis=0)
        np.savez_compressed(f"{args.output_prefix}_collated.npz", **collated)
        print("Collation complete.")

def make_parity_plot(x, y, title, xlabel, ylabel, metrics, filename, labels=None, colors=None, use_quantiles=False):
    """Generates a professional parity plot with KDE density."""
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                           left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    x_flat, y_flat = x.flatten(), y.flatten()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_safe, y_safe = x_flat[mask], y_flat[mask]
    
    if len(x_safe) == 0: return

    if use_quantiles:
        vmin, vmax = np.quantile(np.concatenate([x_safe, y_safe]), [0.005, 0.995])
    else:
        vmin, vmax = np.min([x_safe.min(), y_safe.min()]), np.max([x_safe.max(), y_safe.max()])
    
    if np.abs(vmax - vmin) < 1e-6: vmin -= 0.1; vmax += 0.1
    pad = (vmax - vmin) * 0.05
    lim_min, lim_max = vmin - pad, vmax + pad
    
    if labels:
        for i in range(x.shape[1]):
            ax_main.scatter(y[:,i], x[:,i], s=5, alpha=0.6, label=labels[i], c=colors[i])
        ax_main.legend(loc='lower right')
        ax_histx.hist(y_safe, bins=50, density=True, alpha=0.5, color='gray')
        ax_histy.hist(x_safe, bins=50, density=True, alpha=0.5, orientation='horizontal', color='gray')
    else:
        if len(x_safe) > 50000:
            idx = np.random.choice(len(x_safe), 50000, replace=False)
            sx, sy = y_safe[idx], x_safe[idx]
        else:
            sx, sy = y_safe, x_safe
        
        try:
            xy = np.vstack([sx, sy])
            jitter = np.random.normal(0, 1e-6, xy.shape)
            z = gaussian_kde(xy + jitter)(xy)
            idx = z.argsort()
            sc = ax_main.scatter(sx[idx], sy[idx], c=z[idx], s=5, cmap='viridis', rasterized=True)
            cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.65]) 
            fig.colorbar(sc, cax=cbar_ax, label='Density')
        except:
            ax_main.scatter(sx, sy, s=5, alpha=0.5, c='steelblue')

        ax_histx.hist(y_safe, bins=50, density=True, alpha=0.6, color='steelblue')
        ax_histy.hist(x_safe, bins=50, density=True, alpha=0.6, orientation='horizontal', color='steelblue')

    ax_main.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=1)
    ax_main.set_xlim(lim_min, lim_max); ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_xlabel(xlabel); ax_main.set_ylabel(ylabel)
    ax_histx.set_title(title, fontweight='bold')

    txt = f"MAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\n$R^2$: {metrics['r2']:.4f}"
    ax_main.text(0.05, 0.95, txt, transform=ax_main.transAxes, va='top', ha='left',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_elemental_distribution(values, z_numbers, title, xlabel, filename, extra_data=None, extra_label=None):
    """
    Plots distributions of 'values' separated by element (z_numbers).
    Mimics run.py logic but uses robust plotting style.
    """
    unique_z = np.unique(z_numbers)
    # Prepare layout
    n_plots = 1 + (1 if extra_data is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(5, 4 * n_plots))
    if n_plots == 1: axes = [axes]
    
    # Plot 1: Per Element
    ax = axes[0]
    for z in unique_z:
        mask = (z_numbers == z)
        elem_vals = values[mask]
        if len(elem_vals) == 0: continue
        
        sym = chemical_symbols[z]
        mean_val = np.mean(elem_vals)
        label = f"{sym}: {mean_val:.3f}"
        
        try:
            # Try KDE first
            kde = gaussian_kde(elem_vals + np.random.normal(0, 1e-6, len(elem_vals)))
            x_grid = np.linspace(elem_vals.min(), elem_vals.max(), 200)
            ax.plot(x_grid, kde(x_grid), label=label)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.3)
        except:
            # Fallback to hist
            ax.hist(elem_vals, bins=30, density=True, alpha=0.5, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(title="Element", fontsize=8)
    ax.set_title(title)
    
    # Plot 2: Extra Data (e.g., Total Charge per Structure)
    if extra_data is not None and len(axes) > 1:
        ax2 = axes[1]
        mean_tot = np.mean(extra_data)
        label = f"{extra_label}: {mean_tot:.3f}"
        try:
            kde = gaussian_kde(extra_data + np.random.normal(0, 1e-6, len(extra_data)))
            x_grid = np.linspace(extra_data.min(), extra_data.max(), 200)
            ax2.plot(x_grid, kde(x_grid), color='k')
            ax2.fill_between(x_grid, kde(x_grid), alpha=0.3, color='k')
        except:
            ax2.hist(extra_data, bins=30, density=True, alpha=0.5, color='k')
        ax2.set_xlabel(extra_label or xlabel)
        ax2.set_title(f"Structure Aggregation")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"  Saved distribution plot: {filename}")
    plt.close()
