import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from ase.build import fcc111, molecule

from adsorption_ensemble.basin import BasinConfig, build_basin_dictionary, run_basin_ablation, run_named_basin_ablation
from adsorption_ensemble.basin.reporting import build_node_inflation_audit
from adsorption_ensemble.basin.types import Basin
from adsorption_ensemble.node import NodeConfig, basin_to_node
from adsorption_ensemble.pose import PoseSamplerConfig
from adsorption_ensemble.workflows import AdsorptionWorkflowConfig, run_adsorption_workflow


class TestBasinReporting(unittest.TestCase):
    def test_workflow_writes_basin_dictionary_and_ablation(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.4,
                height_step=0.2,
                max_poses_per_site=2,
                random_seed=0,
            ),
            basin_config=BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=1.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=2,
            save_basin_dictionary=True,
            save_basin_ablation=True,
            basin_ablation_metrics=("signature_only", "rmsd"),
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            self.assertTrue((Path(td) / "basin_dictionary.json").exists())
            self.assertTrue((Path(td) / "basin_ablation.json").exists())
            self.assertTrue((Path(td) / "raw_site_dictionary.json").exists())
            self.assertTrue((Path(td) / "selected_site_dictionary.json").exists())
            self.assertTrue((Path(td) / "sites.png").exists())
            self.assertTrue((Path(td) / "sites_only.png").exists())
            self.assertTrue((Path(td) / "sites_inequivalent.png").exists())
            self.assertIn("basin_dictionary_json", result.artifacts)
            self.assertIn("basin_ablation_json", result.artifacts)

    def test_basin_dictionary_and_ablation_functions(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=3,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.4,
                height_step=0.2,
                max_poses_per_site=2,
                random_seed=0,
            ),
            basin_config=BasinConfig(relax_maxf=0.1, relax_steps=1, energy_window_ev=1.0, desorption_min_bonds=0, work_dir=None),
            max_primitives=2,
            save_basin_dictionary=False,
            save_basin_ablation=False,
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            result = run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            energy_min = result.basin_result.summary.get("energy_min_ev")
            nodes = [basin_to_node(b, slab_n=len(slab), cfg=NodeConfig(), energy_min_ev=energy_min) for b in result.basin_result.basins]
            basin_dict = build_basin_dictionary(
                result.basin_result,
                pose_frames=result.pose_frames,
                nodes=nodes,
                slab_n=len(slab),
                surface_reference=slab,
            )
            self.assertIn("basins", basin_dict)
            self.assertGreaterEqual(len(basin_dict["basins"]), 1)
            self.assertIn("binding_adsorbate_symbols", basin_dict["basins"][0])
            self.assertIn("member_site_labels", basin_dict["basins"][0])
            self.assertIn("node_id_legacy", basin_dict["basins"][0])
            self.assertIn("surface_env_key", basin_dict["basins"][0])
            self.assertIn("surface_geometry_key", basin_dict["basins"][0])
            self.assertIn("node_inflation_audit", basin_dict)
            self.assertIsInstance(basin_dict["node_inflation_audit"], dict)
            ablation = run_basin_ablation(
                frames=result.pose_frames,
                slab_ref=slab,
                adsorbate_ref=ads,
                slab_n=len(slab),
                normal_axis=int(result.surface_context.classification.normal_axis),
                base_config=cfg.basin_config,
                metrics=("signature_only", "rmsd", "binding_surface_distance", "mace_node_l2"),
            )
            self.assertIn("metrics", ablation)
            self.assertIn("signature_only", ablation["metrics"])
            self.assertIn("rmsd", ablation["metrics"])
            self.assertIn("binding_surface_distance", ablation["metrics"])
            self.assertIn("mace_node_l2", ablation["metrics"])
            self.assertIn(ablation["metrics"]["mace_node_l2"]["status"], {"ok", "error"})

    def test_run_named_basin_ablation_supports_custom_configs(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        frame = slab + ads
        out = run_named_basin_ablation(
            frames=[frame.copy(), frame.copy()],
            slab_ref=slab,
            adsorbate_ref=ads,
            slab_n=len(slab),
            normal_axis=2,
            configs={
                "prov_rmsd": BasinConfig(relax_maxf=0.1, relax_steps=1, energy_window_ev=1.0, dedup_metric="rmsd", signature_mode="provenance", work_dir=None),
                "binding_surface": BasinConfig(relax_maxf=0.1, relax_steps=1, energy_window_ev=1.0, dedup_metric="binding_surface_distance", signature_mode="provenance", work_dir=None),
                "pure_rmsd": BasinConfig(relax_maxf=0.1, relax_steps=1, energy_window_ev=1.0, dedup_metric="pure_rmsd", signature_mode="provenance", work_dir=None),
            },
        )
        self.assertIn("prov_rmsd", out["configs"])
        self.assertIn("binding_surface", out["configs"])
        self.assertIn("pure_rmsd", out["configs"])
        self.assertEqual(out["configs"]["prov_rmsd"]["status"], "ok")
        self.assertEqual(out["configs"]["binding_surface"]["status"], "ok")
        self.assertEqual(out["configs"]["pure_rmsd"]["status"], "ok")

    def test_workflow_nodes_json_includes_surface_identity_fields(self):
        slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0)
        ads = molecule("CO")
        cfg = AdsorptionWorkflowConfig(
            pose_sampler_config=PoseSamplerConfig(
                n_rotations=1,
                n_azimuth=2,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.6,
                max_height=2.0,
                height_step=0.2,
                max_poses_per_site=1,
                random_seed=0,
            ),
            basin_config=BasinConfig(
                relax_maxf=0.1,
                relax_steps=1,
                energy_window_ev=1.0,
                desorption_min_bonds=0,
                work_dir=None,
            ),
            max_primitives=1,
        )
        with TemporaryDirectory() as td:
            cfg.work_dir = Path(td)
            run_adsorption_workflow(slab=slab, adsorbate=ads, config=cfg)
            nodes = json.loads((Path(td) / "nodes.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(nodes), 1)
            self.assertIn("node_id_legacy", nodes[0])
            self.assertIn("surface_env_key", nodes[0])
            self.assertIn("surface_geometry_key", nodes[0])

    def test_node_inflation_audit_detects_equivalent_top_site_split(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        top_ids = [i for i, z in enumerate(slab.positions[:, 2]) if abs(z - max(slab.positions[:, 2])) < 1e-8]
        self.assertGreaterEqual(len(top_ids), 2)
        ads = molecule("CO")

        frame_a = slab.copy() + ads.copy()
        frame_a.positions[len(slab) :] += slab.positions[int(top_ids[0])] + np.array([0.0, 0.0, 1.8])
        frame_b = slab.copy() + ads.copy()
        frame_b.positions[len(slab) :] += slab.positions[int(top_ids[1])] + np.array([0.0, 0.0, 1.8])
        basins = [
            Basin(
                basin_id=0,
                atoms=frame_a,
                energy_ev=0.0,
                member_candidate_ids=[0],
                binding_pairs=[(0, int(top_ids[0]))],
                denticity=1,
                signature="sig_top",
            ),
            Basin(
                basin_id=1,
                atoms=frame_b,
                energy_ev=0.01,
                member_candidate_ids=[1],
                binding_pairs=[(0, int(top_ids[1]))],
                denticity=1,
                signature="sig_top",
            ),
        ]
        observed_nodes = [basin_to_node(b, slab_n=len(slab), cfg=NodeConfig(), surface_reference=slab) for b in basins]
        audit = build_node_inflation_audit(
            basins=basins,
            slab_n=len(slab),
            surface_reference=slab,
            observed_nodes=observed_nodes,
            node_cfg=NodeConfig(),
        )
        self.assertEqual(audit["counts"]["observed_node_ids"], 2)
        self.assertEqual(audit["counts"]["legacy_absolute"], 2)
        self.assertEqual(audit["counts"]["surface_geometry"], 1)
        self.assertEqual(audit["counts"]["inflation_observed_vs_surface_geometry"], 1)
        self.assertEqual(len(audit["redundant_surface_geometry_splits"]), 1)
        self.assertEqual(sorted(audit["redundant_surface_geometry_splits"][0]["basin_ids"]), [0, 1])

    def test_node_inflation_audit_keeps_distinct_geometries_separate(self):
        slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
        ads = molecule("CO")
        frame_top = slab.copy() + ads.copy()
        frame_bridge = slab.copy() + ads.copy()
        frame_top.positions[len(slab) :] += slab.positions[0] + np.array([0.0, 0.0, 1.8])
        frame_bridge.positions[len(slab) :] += 0.5 * (slab.positions[0] + slab.positions[1]) + np.array([0.0, 0.0, 1.8])
        basins = [
            Basin(
                basin_id=0,
                atoms=frame_top,
                energy_ev=0.0,
                member_candidate_ids=[0],
                binding_pairs=[(0, 0)],
                denticity=1,
                signature="sig_top",
            ),
            Basin(
                basin_id=1,
                atoms=frame_bridge,
                energy_ev=0.02,
                member_candidate_ids=[1],
                binding_pairs=[(0, 0), (0, 1)],
                denticity=1,
                signature="sig_bridge",
            ),
        ]
        observed_nodes = [basin_to_node(b, slab_n=len(slab), cfg=NodeConfig(), surface_reference=slab) for b in basins]
        audit = build_node_inflation_audit(
            basins=basins,
            slab_n=len(slab),
            surface_reference=slab,
            observed_nodes=observed_nodes,
            node_cfg=NodeConfig(),
        )
        self.assertEqual(audit["counts"]["surface_geometry"], 2)
        self.assertEqual(len(audit["redundant_surface_geometry_splits"]), 0)
