import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.build import bcc100, bcc110, bcc111, fcc100, fcc110, fcc111, fcc211

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
        expected_dim = atom_features.shape[1]
        for p in out.primitives:
            self.assertEqual(len(p.embedding), expected_dim)

    def test_classic_slab_equivalent_site_compression_by_mean_pooling(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, _ = self._build_counts(slab)
        atom_features = np.ones((len(slab), 1), dtype=float)
        embedder = PrimitiveEmbedder(PrimitiveEmbeddingConfig(l2_distance_threshold=1e-12))
        out = embedder.fit_transform(slab=slab, primitives=primitives, atom_features=atom_features)
        unique_topo = len({p.topo_hash for p in out.primitives})
        self.assertEqual(out.basis_count, unique_topo)
        self.assertLess(out.basis_count, out.raw_count)

    def test_classic_slab_gate_fixed_config_with_mace_features(self):
        try:
            from mace.calculators import MACECalculator
        except Exception as exc:
            self.skipTest(f"mace unavailable: {exc}")
        model_path = None
        env_model = str(os.environ.get("AE_MACE_MODEL_PATH", "")).strip()
        if env_model and Path(env_model).exists():
            model_path = env_model
        else:
            for cand in (
                "/root/.cache/mace/mace-omat-0-small.model",
                "/root/.cache/mace/mace-mh-1.model",
                "/root/.cache/mace/MACE-OFF23_small.model",
            ):
                if Path(cand).exists():
                    model_path = cand
                    break
        if model_path is None:
            self.skipTest("No local MACE model available for offline test.")
        try:
            calc = MACECalculator(model_paths=[model_path], device="cpu", default_dtype="float64")
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
            calc = MACECalculator(model_paths=[model_path], device="cpu", default_dtype="float64", head=head)
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
            "fcc111": {"surface_n": 16, "raw": 96, "basis": 4},
            "fcc100": {"surface_n": 16, "raw": 64, "basis": 3},
            "fcc110": {"surface_n": 32, "raw": 192, "basis": 7},
            "fcc211": {"surface_n": 24, "raw": 128, "basis": 14},
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
                self.assertEqual(out.basis_count, expected[name]["basis"])

    def test_build_site_dictionary_contains_atom_and_normal(self):
        slab = fcc111("Pt", size=(4, 4, 4), vacuum=12.0)
        _, primitives, _ = self._build_counts(slab)
        site_dict = build_site_dictionary(primitives)
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
        all_kind_site_ids = set()
        for ids in site_dict["kinds"].values():
            all_kind_site_ids.update(ids)
        self.assertEqual(set(site_dict["sites"].keys()), all_kind_site_ids)


if __name__ == "__main__":
    unittest.main()
