import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.build import bcc100, bcc110, bcc111, fcc100, fcc110, fcc111, fcc211, hcp10m10
from ase.build import surface
from ase.spacegroup import crystal

from adsorption_ensemble.site import (
    PrimitiveBuilder,
    PrimitiveEmbedder,
    PrimitiveEmbeddingConfig,
    build_site_dictionary,
    compare_graph_vs_delaunay,
    enumerate_primitives_delaunay,
)
from adsorption_ensemble.surface import ProbeScanDetector, SurfacePreprocessor, VoxelFloodDetector
from adsorption_ensemble.visualization import plot_inequivalent_sites_2d, plot_site_centers_only, plot_surface_primitives_2d
from adsorption_ensemble.site.primitives import SitePrimitive


def _replace_top_layer_atoms(slab, from_symbol: str, to_symbol: str, frac: float):
    out = slab.copy()
    z = np.asarray(out.get_positions(), dtype=float)[:, 2]
    z_top = float(np.max(z))
    top = [i for i, zi in enumerate(z) if (z_top - zi) < 1.0]
    n_swap = max(1, int(round(len(top) * float(frac))))
    for i in top[:n_swap]:
        if out[i].symbol == from_symbol:
            out[i].symbol = to_symbol
    out.info.pop("adsorbate_info", None)
    return out


class TestSitePrimitives(unittest.TestCase):
    def setUp(self):
        self.pre = SurfacePreprocessor(min_surface_atoms=6)
        self.builder = PrimitiveBuilder()

    def _build_counts(self, slab):
        ctx = self.pre.build_context(slab)
        primitives = self.builder.build(slab, ctx)
        kinds = Counter(p.kind for p in primitives)
        return ctx, primitives, kinds

    def test_pt111_sanity(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, kinds = self._build_counts(slab)
        self.assertGreater(kinds["1c"], 0)
        self.assertGreater(kinds["2c"], 0)
        self.assertGreater(len(primitives), 0)
        self.assertTrue(all(p.topo_hash for p in primitives))

    def test_builder_ignores_ase_reference_sites_even_when_requested(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        self.assertIn("adsorbate_info", slab.info)
        ctx = self.pre.build_context(slab)
        primitives = PrimitiveBuilder(use_ase_reference_sites=True).build(slab, ctx)
        self.assertGreater(len(primitives), 4)
        self.assertTrue(all(p.site_label is None for p in primitives))

    def test_pt100_sanity_has_4c(self):
        slab = fcc100("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, kinds = self._build_counts(slab)
        self.assertGreater(kinds["1c"], 0)
        self.assertGreater(kinds["2c"], 0)
        self.assertGreater(kinds["4c"], 0)
        self.assertGreater(len(primitives), 0)

    def test_pt211_sanity(self):
        slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
        _, primitives, kinds = self._build_counts(slab)
        self.assertGreater(kinds["1c"], 0)
        self.assertGreater(kinds["2c"], 0)
        self.assertGreater(len(primitives), 0)

    def test_fcc111_has_no_spurious_fourfold_sites(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, _, kinds = self._build_counts(slab)
        self.assertEqual(kinds["4c"], 0)

    def test_step_surfaces_do_not_emit_cross_level_fourfold_sites(self):
        cases = {
            "Pt_fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
            "Ru_hcp10m10": hcp10m10("Ru", size=(4, 4, 4), vacuum=12.0),
        }
        for name, slab in cases.items():
            with self.subTest(case=name):
                _, primitives, kinds = self._build_counts(slab)
                self.assertEqual(kinds["4c"], 0)
                self.assertGreaterEqual(kinds["2c"], 1)

    def test_tio2_contains_ti_in_surface_and_primitives(self):
        rutile = crystal(
            symbols=["Ti", "O"],
            basis=[(0.0, 0.0, 0.0), (0.305, 0.305, 0.0)],
            spacegroup=136,
            cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
        )
        slab = surface(rutile, (1, 1, 0), layers=6, vacuum=12.0).repeat((2, 2, 1))
        ctx, primitives, _ = self._build_counts(slab)
        surface_symbols = {slab[int(i)].symbol for i in ctx.detection.surface_atom_ids}
        self.assertEqual(surface_symbols, {"O", "Ti"})
        primitive_formulas = {"".join(sorted(slab[int(i)].symbol for i in p.atom_ids)) for p in primitives}
        self.assertTrue(any("Ti" in formula for formula in primitive_formulas))
        self.assertTrue(any("O" in formula and "Ti" in formula for formula in primitive_formulas))

    def test_local_frame_orthonormality(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
        _, primitives, _ = self._build_counts(slab)
        sample = primitives[:20]
        for p in sample:
            self.assertAlmostEqual(float(np.linalg.norm(p.normal)), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(p.t1)), 1.0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(p.t2)), 1.0, places=6)
            self.assertAlmostEqual(float(np.dot(p.normal, p.t1)), 0.0, places=5)
            self.assertAlmostEqual(float(np.dot(p.normal, p.t2)), 0.0, places=5)
            self.assertAlmostEqual(float(np.dot(p.t1, p.t2)), 0.0, places=5)

    def test_primitive_visualization_png(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx, primitives, _ = self._build_counts(slab)
        self.assertGreaterEqual(len(ctx.detection.surface_atom_ids), 8)
        with TemporaryDirectory() as td:
            out = Path(td) / "pt111_sites.png"
            out2 = Path(td) / "pt111_sites_only.png"
            out3 = Path(td) / "pt111_sites_inequivalent.png"
            path = plot_surface_primitives_2d(slab=slab, context=ctx, primitives=primitives, filename=out)
            path2 = plot_site_centers_only(slab=slab, primitives=primitives, filename=out2)
            path3 = plot_inequivalent_sites_2d(slab=slab, primitives=primitives, filename=out3)
            self.assertTrue(path.exists())
            self.assertTrue(path2.exists())
            self.assertTrue(path3.exists())
            self.assertGreater(path.stat().st_size, 1024)
            self.assertGreater(path2.stat().st_size, 1024)
            self.assertGreater(path3.stat().st_size, 1024)
            self.assertLess(abs(path.stat().st_mtime - path2.stat().st_mtime), 1.0)

    def test_close_sites_keep_higher_coordination(self):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0)
        b = PrimitiveBuilder(min_site_distance=0.1)
        p_bridge = SitePrimitive(
            kind="2c",
            atom_ids=(0, 1),
            center=np.array([1.0, 1.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            topo_hash="2c|n=2|deg=3,3",
        )
        p_hollow = SitePrimitive(
            kind="3c",
            atom_ids=(0, 1, 2),
            center=np.array([1.05, 1.0, 1.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            topo_hash="3c|n=3|deg=3,3,3",
        )
        kept = b._prune_by_center_distance([p_bridge, p_hollow], slab=slab)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].kind, "3c")

    def test_no_hollow_center_overlap_surface_atom(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx, primitives, _ = self._build_counts(slab)
        surface_pos = slab.get_positions()[ctx.detection.surface_atom_ids]
        for p in primitives:
            if p.kind not in {"3c", "4c"}:
                continue
            d = np.linalg.norm(surface_pos - p.center, axis=1).min()
            self.assertGreaterEqual(float(d), 0.20 - 1e-8)

    def test_delaunay_case_compare(self):
        slabs = [
            fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
            bcc100("Fe", size=(4, 4, 4), vacuum=12.0),
            bcc110("Fe", size=(4, 4, 4), vacuum=12.0),
            bcc111("Fe", size=(4, 4, 4), vacuum=12.0),
        ]
        for slab in slabs:
            with self.subTest(formula=slab.get_chemical_formula()):
                ctx, primitives, _ = self._build_counts(slab)
                graph_sites = {"1c": [], "2c": [], "3c": [], "4c": []}
                for p in primitives:
                    graph_sites[p.kind].append(p.atom_ids)
                try:
                    d_sites = enumerate_primitives_delaunay(
                        slab=slab,
                        surface_atom_ids=ctx.detection.surface_atom_ids,
                        normal_axis=int(ctx.classification.normal_axis),
                    )
                except ImportError:
                    self.skipTest("SciPy not available")
                cmp = compare_graph_vs_delaunay(graph_sites, d_sites)
                self.assertGreater(cmp["1c"]["graph"], 0)
                self.assertGreaterEqual(cmp["2c"]["graph"], 0)

    def test_primitive_embedding_and_clustering(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        ctx, primitives, _ = self._build_counts(slab)
        feat_dim = 8
        atom_features = np.zeros((len(slab), feat_dim), dtype=float)
        z = slab.get_atomic_numbers()
        pos = slab.get_positions()
        atom_features[:, 0] = z / np.max(z)
        atom_features[:, 1] = pos[:, 0] / (np.max(pos[:, 0]) + 1e-12)
        atom_features[:, 2] = pos[:, 1] / (np.max(pos[:, 1]) + 1e-12)
        atom_features[:, 3] = pos[:, 2] / (np.max(pos[:, 2]) + 1e-12)
        atom_features[:, 4] = np.sin(pos[:, 0])
        atom_features[:, 5] = np.cos(pos[:, 1])
        atom_features[:, 6] = np.sin(pos[:, 2])
        atom_features[:, 7] = 1.0
        embedder = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22))
        out = embedder.fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
        self.assertEqual(out.raw_count, len(primitives))
        self.assertGreater(out.basis_count, 0)
        self.assertLessEqual(out.basis_count, out.raw_count)
        self.assertGreater(out.compression_ratio, 0.0)
        self.assertLessEqual(out.compression_ratio, 1.0)
        self.assertGreater(len(out.bucket_sizes), 0)
        for p in out.primitives:
            self.assertIsNotNone(p.embedding)
            self.assertIsNotNone(p.basis_id)
        for b in out.basis_primitives:
            self.assertIsNotNone(b.embedding)
            self.assertIsNotNone(b.basis_id)
        site_dict = build_site_dictionary(out.primitives)
        self.assertEqual(site_dict["meta"]["n_basis_groups"], out.basis_count)
        expected_dim = atom_features.shape[1] + 4 + embedder.config.geom_k_nearest
        for p in out.primitives:
            self.assertEqual(len(p.embedding), expected_dim)

    def test_classic_slab_equivalent_site_compression_by_mean_pooling(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, _ = self._build_counts(slab)
        atom_features = np.ones((len(slab), 1), dtype=float)
        embedder = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=1e-12, include_geom_aux=False))
        out = embedder.fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
        unique_bucket_keys = len({embedder._bucket_key(p, slab) for p in out.primitives})
        self.assertEqual(out.basis_count, unique_bucket_keys)
        self.assertEqual(out.basis_count, 3)
        self.assertGreater(out.raw_count, out.basis_count)
        self.assertEqual(Counter(p.kind for p in out.basis_primitives), Counter({"1c": 1, "2c": 1, "3c": 1}))

    def test_classic_slab_gate_fixed_config_with_mace_features(self):
        try:
            from mace.calculators import MACECalculator
        except Exception as exc:
            self.skipTest(f"mace unavailable: {exc}")
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")
        if not bool(torch.cuda.is_available()):
            self.skipTest("CUDA is required for the MACE primitive regression test.")
        model_path = None
        env_model = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
        if env_model and Path(env_model).exists():
            model_path = env_model
        else:
            for cand in (
                "/root/.cache/mace/mace-omat-0-small.model",
            ):
                if Path(cand).exists():
                    model_path = cand
                    break
        if model_path is None:
            self.skipTest("No local MACE model available for offline test.")
        try:
            calc = MACECalculator(model_paths=[model_path], device="cuda", default_dtype="float32", enable_cueq=True)
        except ValueError as exc:
            msg = str(exc)
            if "Available heads are:" not in msg:
                raise
            head_raw = msg.split("Available heads are:", 1)[-1].strip()
            head_raw = head_raw.strip("[]")
            heads = [x.strip().strip("'").strip('"') for x in head_raw.split(",") if x.strip()]
            if not heads:
                raise
            head = "omat_pbe" if "omat_pbe" in heads else heads[0]
            calc = MACECalculator(
                model_paths=[model_path],
                device="cuda",
                default_dtype="float32",
                enable_cueq=True,
                head=head,
            )
        pre = SurfacePreprocessor(
            min_surface_atoms=6,
            primary_detector=ProbeScanDetector(grid_step=0.6),
            fallback_detector=VoxelFloodDetector(spacing=0.8),
            target_surface_fraction=0.25,
            target_count_mode="off",
        )
        slabs = {
            "fcc111": fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
            "fcc100": fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
            "fcc110": fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
            "fcc211": fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        }
        expected = {
            "fcc111": {"surface_n": 16, "raw": 96, "basis": Counter({"1c": 1, "2c": 1, "3c": 2})},
            "fcc100": {"surface_n": 16, "raw": 64, "basis": Counter({"1c": 1, "2c": 1, "4c": 1})},
            "fcc110": {"surface_n": 16, "raw": 64, "basis": Counter({"1c": 1, "2c": 2, "4c": 1})},
            "fcc211": {"surface_n": 24, "raw": 120, "basis": Counter({"1c": 2, "2c": 4, "3c": 4})},
        }
        for name, slab in slabs.items():
            with self.subTest(case=name):
                ctx = pre.build_context(slab)
                primitives = PrimitiveBuilder(min_site_distance=0.1).build(slab, ctx)
                atom_features = np.asarray(calc.get_descriptors(slab), dtype=float)
                out = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.22)).fit_transform(
                    slab=slab, primitives=primitives, atom_features=atom_features
                )
                self.assertEqual(len(ctx.detection.surface_atom_ids), expected[name]["surface_n"])
                self.assertEqual(out.raw_count, expected[name]["raw"])
                self.assertEqual(Counter(p.kind for p in out.basis_primitives), expected[name]["basis"])
                self.assertEqual(out.basis_count, int(sum(expected[name]["basis"].values())))

    def test_alloy_fcc111_species_aware_embedding_keeps_distinct_site_families(self):
        slab = _replace_top_layer_atoms(fcc111("Cu", size=(4, 4, 4), vacuum=12.0), "Cu", "Ni", frac=0.25)
        ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
        primitives = PrimitiveBuilder().build(slab, ctx)
        atom_features = (slab.get_atomic_numbers().astype(float) / (np.max(slab.get_atomic_numbers()) + 1.0e-12)).reshape(-1, 1)
        out = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=0.20)).fit_transform(
            slab=slab,
            primitives=primitives,
            atom_features=atom_features,
        )

        self.assertEqual(out.basis_count, 11)
        self.assertEqual(Counter(p.kind for p in out.basis_primitives), Counter({"1c": 2, "2c": 3, "3c": 6}))

        def _formula(atom_ids):
            symbols = Counter(str(slab[int(i)].symbol) for i in atom_ids)
            return "".join(sym if n == 1 else f"{sym}{n}" for sym, n in sorted(symbols.items()))

        self.assertEqual(
            Counter(_formula(p.atom_ids) for p in out.basis_primitives),
            Counter(
                {
                    "Ni": 1,
                    "Cu": 1,
                    "Ni2": 1,
                    "CuNi": 1,
                    "Cu2": 1,
                    "CuNi2": 2,
                    "Cu2Ni": 2,
                    "Cu3": 2,
                }
            ),
        )

    def test_build_site_dictionary_contains_atom_and_normal(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, _ = self._build_counts(slab)
        site_dict = build_site_dictionary(primitives, slab=slab)
        self.assertIn("sites", site_dict)
        self.assertIn("kinds", site_dict)
        self.assertIn("basis_groups", site_dict)
        self.assertEqual(site_dict["meta"]["n_sites"], len(primitives))
        self.assertEqual(sum(site_dict["meta"]["n_kinds"].values()), len(primitives))
        for sid, rec in site_dict["sites"].items():
            self.assertTrue(sid.startswith("site_"))
            self.assertIn(rec["kind"], {"1c", "2c", "3c", "4c"})
            self.assertGreaterEqual(len(rec["atom_ids"]), 1)
            self.assertEqual(len(rec["center"]), 3)
            self.assertEqual(len(rec["normal"]), 3)
            self.assertEqual(len(rec["t1"]), 3)
            self.assertEqual(len(rec["t2"]), 3)
            self.assertEqual(rec["coordinates"], rec["center"])
            self.assertEqual(rec["n_vector"], rec["normal"])
            self.assertEqual(rec["h_vector"], rec["t1"])
            self.assertEqual(int(rec["connectivity"]), len(rec["atom_ids"]))
            self.assertIsInstance(rec["topology"], str)
            self.assertIsInstance(rec["site_formula"], str)
        all_kind_site_ids = set()
        for ids in site_dict["kinds"].values():
            all_kind_site_ids.update(ids)
        self.assertEqual(set(site_dict["sites"].keys()), all_kind_site_ids)


if __name__ == "__main__":
    unittest.main()
