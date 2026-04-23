from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from .config import RDKitEmbedConfig
from .xtb import MDRunResult

KCAL_PER_MOL_TO_EV = 0.0433641153087705


@dataclass
class RDKitConformerGenerator:
    config: RDKitEmbedConfig
    last_generation_metadata: dict | None = None

    def generate(self, molecule: Atoms, run_dir: Path) -> MDRunResult:
        Chem, AllChem = self._load_dependencies()
        run_dir.mkdir(parents=True, exist_ok=True)
        rd_mol, graph_source = self._build_rdkit_mol(molecule, Chem)
        params = self._build_embed_params(AllChem)
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                rd_mol,
                numConfs=int(self.config.num_confs),
                params=params,
            )
        )
        frames: list[Atoms] = []
        n_optimized = 0
        ff_name = str(self.config.optimize_forcefield).strip().lower()
        for conf_id in conf_ids:
            energy_ev = None
            if ff_name != "none":
                energy_ev, optimized = self._optimize_and_energy(rd_mol, AllChem, conf_id=int(conf_id))
                if optimized:
                    n_optimized += 1
            atoms = self._atoms_from_conf(molecule, rd_mol, conf_id=int(conf_id))
            atoms.info["generator_backend"] = "rdkit_embed"
            atoms.info["generator_conf_id"] = int(conf_id)
            atoms.info["rdkit_graph_source"] = str(graph_source)
            if energy_ev is not None:
                atoms.info["generator_energy_ev"] = float(energy_ev)
            frames.append(atoms)
        metadata = {
            "generator_backend": "rdkit_embed",
            "n_requested_confs": int(self.config.num_confs),
            "n_embedded_confs": int(len(conf_ids)),
            "n_pruned_confs": int(max(0, int(self.config.num_confs) - len(conf_ids))),
            "n_optimized_confs": int(n_optimized),
            "forcefield": str(ff_name),
            "embed_method": str(self.config.embed_method),
            "random_seed": int(self.config.random_seed),
            "use_random_coords": bool(self.config.use_random_coords),
            "num_threads": int(self.config.num_threads),
            "graph_source": str(graph_source),
        }
        self.last_generation_metadata = metadata
        return MDRunResult(frames=frames, metadata=metadata)

    def _load_dependencies(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except Exception as exc:
            raise RuntimeError(
                "RDKit conformer generation requested but RDKit is not installed. "
                "Install rdkit and retry with --generator-backend rdkit_embed."
            ) from exc
        return Chem, AllChem

    def _build_embed_params(self, AllChem):
        method = str(self.config.embed_method).strip().lower()
        if method != "etkdg_v3":
            raise ValueError(f"Unsupported RDKit embed method: {self.config.embed_method}")
        params = AllChem.ETKDGv3()
        params.pruneRmsThresh = float(self.config.prune_rms_thresh)
        params.randomSeed = int(self.config.random_seed)
        params.useRandomCoords = bool(self.config.use_random_coords)
        params.numThreads = int(self.config.num_threads)
        return params

    def _build_rdkit_mol(self, atoms: Atoms, Chem):
        rw = Chem.RWMol()
        for sym in atoms.get_chemical_symbols():
            rw.AddAtom(Chem.Atom(str(sym)))
        bonds = self._connectivity_bonds_from_info(atoms)
        graph_source = "gaussian_connectivity"
        if not bonds:
            bonds = self._heuristic_single_bonds(atoms)
            graph_source = "heuristic_single_bonds"
        aromatic_atoms: set[int] = set()
        for i, j, order in bonds:
            bond_type = self._bond_type_from_order(order, Chem, atoms=atoms, atom_i=int(i), atom_j=int(j))
            rw.AddBond(int(i), int(j), bond_type)
            if bond_type == Chem.BondType.AROMATIC:
                aromatic_atoms.add(int(i))
                aromatic_atoms.add(int(j))
        mol = rw.GetMol()
        for idx in aromatic_atoms:
            mol.GetAtomWithIdx(int(idx)).SetIsAromatic(True)
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise RuntimeError(
                "Failed to build a sanitized RDKit molecule from the provided ASE Atoms. "
                "Provide Gaussian connectivity or use a molecule with clearer bonding."
            ) from exc
        return mol, graph_source

    @staticmethod
    def _bond_type_from_order(order: float, Chem, atoms: Atoms | None = None, atom_i: int | None = None, atom_j: int | None = None):
        val = float(order)
        if abs(val - 1.5) < 0.15 and RDKitConformerGenerator._use_aromatic_bond(
            atoms=atoms,
            atom_i=atom_i,
            atom_j=atom_j,
        ):
            return Chem.BondType.AROMATIC
        if val >= 2.5:
            return Chem.BondType.TRIPLE
        if val >= 1.5:
            return Chem.BondType.DOUBLE
        return Chem.BondType.SINGLE

    @staticmethod
    def _use_aromatic_bond(atoms: Atoms | None, atom_i: int | None, atom_j: int | None) -> bool:
        if atoms is None or atom_i is None or atom_j is None:
            return False
        aromatic_capable = {"B", "C", "N", "P", "S"}
        sym_i = str(atoms[int(atom_i)].symbol)
        sym_j = str(atoms[int(atom_j)].symbol)
        return sym_i in aromatic_capable and sym_j in aromatic_capable

    @staticmethod
    def _connectivity_bonds_from_info(atoms: Atoms) -> list[tuple[int, int, float]]:
        raw = atoms.info.get("connectivity_bonds", ())
        out: list[tuple[int, int, float]] = []
        for item in raw:
            try:
                i, j, order = item
                out.append((int(i), int(j), float(order)))
            except Exception:
                continue
        return out

    @staticmethod
    def _heuristic_single_bonds(atoms: Atoms, tau: float = 1.20) -> list[tuple[int, int, float]]:
        out: list[tuple[int, int, float]] = []
        if len(atoms) <= 1:
            return out
        z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        d = atoms.get_all_distances(mic=False)
        for i in range(len(atoms)):
            ri = float(covalent_radii[int(z[i])])
            for j in range(i + 1, len(atoms)):
                rj = float(covalent_radii[int(z[j])])
                if float(d[i, j]) <= float(tau) * (ri + rj):
                    out.append((int(i), int(j), 1.0))
        return out

    def _optimize_and_energy(self, mol, AllChem, conf_id: int) -> tuple[float | None, bool]:
        ff_name = str(self.config.optimize_forcefield).strip().lower()
        if ff_name == "uff":
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(conf_id))
            except Exception:
                ff = None
        elif ff_name == "mmff":
            try:
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = None if props is None else AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(conf_id))
            except Exception:
                ff = None
        else:
            return None, False
        if ff is None:
            return None, False
        try:
            ff.Minimize(maxIts=int(self.config.max_opt_iters))
            energy_kcal = float(ff.CalcEnergy())
        except Exception:
            return None, False
        return float(energy_kcal * KCAL_PER_MOL_TO_EV), True

    @staticmethod
    def _atoms_from_conf(reference: Atoms, mol, conf_id: int) -> Atoms:
        conf = mol.GetConformer(int(conf_id))
        positions = []
        symbols = []
        for idx in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(int(idx))
            positions.append([float(pos.x), float(pos.y), float(pos.z)])
            symbols.append(reference[idx].symbol)
        atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
        return atoms
