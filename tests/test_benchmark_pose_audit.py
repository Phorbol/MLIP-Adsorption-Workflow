import unittest

from ase.build import fcc111, molecule

from adsorption_ensemble.benchmark.pose_audit import classify_tilt_bin, summarize_pose_frames
from adsorption_ensemble.pose import PoseSampler, PoseSamplerConfig
from adsorption_ensemble.site import PrimitiveBuilder
from adsorption_ensemble.surface import SurfacePreprocessor


class TestBenchmarkPoseAudit(unittest.TestCase):
    def test_classify_tilt_bin_thresholds(self):
        self.assertEqual(classify_tilt_bin(0.0), "upright")
        self.assertEqual(classify_tilt_bin(9.9), "upright")
        self.assertEqual(classify_tilt_bin(10.0), "tilted")
        self.assertEqual(classify_tilt_bin(59.9), "tilted")
        self.assertEqual(classify_tilt_bin(60.0), "flat")

    def test_summarize_pose_frames_reports_orientation_and_site_counts(self):
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=12.0)
        ads = molecule("H2O")
        ctx = SurfacePreprocessor(min_surface_atoms=6).build_context(slab)
        primitives = PrimitiveBuilder().build(slab, ctx)
        sampler = PoseSampler(
            PoseSamplerConfig(
                n_rotations=4,
                n_azimuth=2,
                n_shifts=1,
                shift_radius=0.0,
                min_height=1.2,
                max_height=3.2,
                height_step=0.1,
                max_poses_per_site=8,
                prune_com_distance=0.0,
                prune_rot_distance=0.0,
                random_seed=0,
            )
        )
        poses = sampler.sample(
            slab=slab,
            adsorbate=ads,
            primitives=[primitives[0]],
            surface_atom_ids=ctx.detection.surface_atom_ids,
        )
        frames = []
        for pose in poses:
            frame = slab + pose.atoms
            primitive = primitives[int(pose.primitive_index)]
            frame.info["primitive_index"] = int(pose.primitive_index)
            frame.info["basis_id"] = -1 if pose.basis_id is None else int(pose.basis_id)
            frame.info["site_kind"] = str(primitive.kind)
            frame.info["site_label"] = str(getattr(primitive, "site_label", None) or primitive.kind)
            frame.info["rotation_index"] = int(pose.rotation_index)
            frame.info["azimuth_index"] = int(pose.azimuth_index)
            frame.info["height"] = float(pose.height)
            frames.append(frame)

        summary = summarize_pose_frames(frames, slab_n=len(slab), primitives=[primitives[0]])
        self.assertEqual(summary["n_pose_frames"], len(frames))
        self.assertEqual(summary["n_unique_primitives"], 1)
        self.assertEqual(len(summary["sites"]), 1)
        self.assertEqual(sum(summary["orientation_bin_counts"].values()), len(frames))
        self.assertIn("tilt_deg_stats", summary["sites"][0])


if __name__ == "__main__":
    unittest.main()
