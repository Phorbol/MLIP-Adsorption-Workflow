import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase import Atom
from ase.build import bulk, fcc100, fcc110, fcc111, fcc211, molecule, surface

from adsorption_ensemble.selection import DualThresholdSelector, EnergyWindowFilter, FarthestPointSamplingSelector, RMSDSelector
from adsorption_ensemble.surface import ExposedSurfaceGraphBuilder, ProbeScanDetector, SlabClassifier, SurfacePreprocessor, VoxelFloodDetector, export_surface_detection_report


class TestSurfacePipeline(unittest.TestCase):
    def test_slab_classifier_on_multiple_systems(self):
        classifier = SlabClassifier(vacuum_threshold=5.0, vacuum_ratio_threshold=1.4)
        slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0)
        bulk_cell = bulk("Cu", "fcc", a=3.6, cubic=True)
        mol = molecule("H2O")
        self.assertTrue(classifier.classify(slab).is_slab)
        self.assertFalse(classifier.classify(bulk_cell).is_slab)
        self.assertFalse(classifier.classify(mol).is_slab)

    def test_surface_detector_probe_and_voxel(self):
        slab = fcc211("Pd", size=(6, 4, 4), vacuum=12.0)
        classifier = SlabClassifier(vacuum_threshold=5.0, vacuum_ratio_threshold=1.4)
        result = classifier.classify(slab)
        self.assertTrue(result.is_slab)
        self.assertIsNotNone(result.normal_axis)
        normal_axis = int(result.normal_axis)
        probe = ProbeScanDetector(grid_step=0.8, probe_radius=0.6)
        voxel = VoxelFloodDetector(spacing=1.0)
        r_probe = probe.detect(slab, normal_axis=normal_axis)
        r_voxel = voxel.detect(slab, normal_axis=normal_axis)
        self.assertGreater(len(r_probe.surface_atom_ids), 0)
        z = slab.get_positions()[:, normal_axis]
        self.assertGreater(float(np.mean(z[r_probe.surface_atom_ids])), float(np.mean(z)) - 0.5)
        if len(r_voxel.surface_atom_ids) > 0:
            self.assertGreater(float(np.mean(z[r_voxel.surface_atom_ids])), float(np.mean(z)) - 0.5)

    def test_graph_builder(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        cls = SlabClassifier(vacuum_threshold=5.0, vacuum_ratio_threshold=1.4).classify(slab)
        det = ProbeScanDetector(grid_step=0.8).detect(slab, normal_axis=int(cls.normal_axis))
        graph = ExposedSurfaceGraphBuilder(neighbor_scale=1.3).build(slab, det.surface_atom_ids)
        self.assertGreater(len(graph.surface_atom_ids), 0)
        self.assertGreater(len(graph.edges), 0)
        s = set(graph.surface_atom_ids)
        for i, j in graph.edges:
            self.assertIn(i, s)
            self.assertIn(j, s)

    def test_surface_preprocessor_end_to_end(self):
        cu_bulk = bulk("Cu", "fcc", a=3.6, cubic=True)
        slab = surface(cu_bulk, (2, 1, 1), layers=4, vacuum=12.0)
        pre = SurfacePreprocessor(min_surface_atoms=6)
        ctx = pre.build_context(slab)
        self.assertTrue(ctx.classification.is_slab)
        self.assertGreater(len(ctx.detection.surface_atom_ids), 0)
        self.assertGreater(len(ctx.graph.edges), 0)

    def test_fcc211_keeps_step_levels(self):
        slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
        pre = SurfacePreprocessor(min_surface_atoms=6)
        ctx = pre.build_context(slab)
        self.assertGreaterEqual(len(ctx.detection.surface_atom_ids), 24)
        z = np.round(slab.get_positions()[ctx.detection.surface_atom_ids, int(ctx.classification.normal_axis)], 6)
        self.assertGreaterEqual(len(np.unique(z)), 3)

    def test_four_layer_cases_have_quarter_surface_atoms(self):
        slabs = [
            fcc111("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc100("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc110("Pt", size=(4, 4, 4), vacuum=12.0),
            fcc211("Pt", size=(6, 4, 4), vacuum=12.0),
        ]
        pre = SurfacePreprocessor(min_surface_atoms=6)
        for slab in slabs:
            with self.subTest(formula=slab.get_chemical_formula()):
                ctx = pre.build_context(slab)
                self.assertEqual(len(ctx.detection.surface_atom_ids), len(slab) // 4)

    def test_fcc110_defect_surface_counts(self):
        slab = fcc110("Pt", size=(4, 4, 4), vacuum=12.0)
        vac = slab.copy()
        del vac[int(np.argmax(vac.get_positions()[:, 2]))]
        ada = slab.copy()
        z = ada.get_positions()[:, 2]
        top_ids = np.where(z > np.max(z) - 0.2)[0]
        center = np.mean(ada.get_positions()[top_ids], axis=0)
        ada.append(Atom("Pt", position=center + np.array([0.0, 0.0, 1.8])))
        pre = SurfacePreprocessor(min_surface_atoms=6)
        ctx_v = pre.build_context(vac)
        ctx_a = pre.build_context(ada)
        self.assertEqual(len(ctx_v.detection.surface_atom_ids), 15)
        self.assertEqual(len(ctx_a.detection.surface_atom_ids), 17)

    def test_fcc211_defect_surface_counts(self):
        slab = fcc211("Pt", size=(6, 4, 4), vacuum=12.0)
        vac = slab.copy()
        del vac[int(np.argmax(vac.get_positions()[:, 2]))]
        ada = slab.copy()
        z = ada.get_positions()[:, 2]
        top_ids = np.where(z > np.max(z) - 0.2)[0]
        center = np.mean(ada.get_positions()[top_ids], axis=0)
        ada.append(Atom("Pt", position=center + np.array([0.0, 0.0, 1.8])))
        pre = SurfacePreprocessor(min_surface_atoms=6)
        ctx_v = pre.build_context(vac)
        ctx_a = pre.build_context(ada)
        self.assertEqual(len(ctx_v.detection.surface_atom_ids), 23)
        self.assertEqual(len(ctx_a.detection.surface_atom_ids), 25)

    def test_auto_layers_target_on_non_four_layer(self):
        slab = fcc111("Pt", size=(4, 4, 6), vacuum=12.0)
        pre = SurfacePreprocessor(min_surface_atoms=6, target_count_mode="auto_layers")
        ctx = pre.build_context(slab)
        self.assertEqual(int(ctx.detection.diagnostics.get("estimated_layers", 0)), 6)
        self.assertEqual(ctx.detection.diagnostics.get("target_mode"), "auto_layers")
        self.assertEqual(len(ctx.detection.surface_atom_ids), int(ctx.detection.diagnostics.get("target_surface_count", -1)))

    def test_auto_layers_on_oxide_and_alloy(self):
        mgo = surface(bulk("MgO", "rocksalt", a=4.21, cubic=True), (1, 0, 0), layers=4, vacuum=12.0).repeat((2, 2, 1))
        cu = surface(bulk("Cu", "fcc", a=3.6, cubic=True), (2, 1, 1), layers=4, vacuum=12.0).repeat((2, 2, 1))
        symbols = cu.get_chemical_symbols()
        z = cu.get_positions()[:, 2]
        top_ids = np.where(z > np.max(z) - 2.5)[0]
        for ii, idx in enumerate(sorted(top_ids, key=lambda i: z[i], reverse=True)):
            if ii % 3 == 0:
                symbols[idx] = "Ni"
        cu.set_chemical_symbols(symbols)
        pre = SurfacePreprocessor(min_surface_atoms=6, target_count_mode="auto_layers")
        for slab in (mgo, cu):
            with self.subTest(formula=slab.get_chemical_formula()):
                ctx = pre.build_context(slab)
                self.assertGreater(len(ctx.detection.surface_atom_ids), 0)
                self.assertGreater(len(ctx.graph.edges), 0)

    def test_selection_strategies(self):
        energies = np.array([0.0, 0.03, 0.10, 0.35, 0.36], dtype=float)
        feats = np.array(
            [
                [0.0, 0.0],
                [0.01, 0.0],
                [0.3, 0.2],
                [1.0, 1.0],
                [1.05, 1.0],
            ],
            dtype=float,
        )
        dual = DualThresholdSelector(EnergyWindowFilter(delta_e=0.2), RMSDSelector(threshold=0.05))
        selected = dual.select(energies=energies, features=feats)
        self.assertIn(0, selected)
        self.assertIn(2, selected)
        self.assertNotIn(3, selected)
        fps = FarthestPointSamplingSelector(random_seed=0)
        picked = fps.select(features=feats, k=3)
        self.assertEqual(len(set(picked)), len(picked))
        self.assertEqual(len(picked), 3)

    def test_export_surface_detection_report(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
        ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
        with TemporaryDirectory() as td:
            files = export_surface_detection_report(slab, ctx, output_dir=td)
            self.assertTrue(Path(files["surface_csv"]).exists())
            self.assertTrue(Path(files["surface_xyz"]).exists())
            self.assertTrue(Path(files["tagged_slab"]).exists())


if __name__ == "__main__":
    unittest.main()
