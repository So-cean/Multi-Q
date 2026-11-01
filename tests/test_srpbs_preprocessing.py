"""Regression tests for SRPBS preprocessing geometry handling."""

from pathlib import Path
import stat
import tempfile
import unittest

import nibabel as nib
import numpy as np


from multiq.preprocessing import (
    atomic_save_nifti,
    copy_spatial_forms,
    subject_minmax_to_output,
)


class SRPBSPreprocessingTests(unittest.TestCase):
    def test_mask_uses_source_qform_and_sform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.nii.gz"
            mask_path = root / "mask.nii.gz"
            source_qform = np.array(
                [[1, 0, 0, -10.25], [0, 1, 0, -20.5], [0, 0, 1, 4.75], [0, 0, 0, 1]],
                dtype=float,
            )
            source_sform = source_qform.copy()
            source_sform[0, 3] += 0.001
            source = nib.Nifti1Image(np.ones((5, 6, 7)), source_sform)
            source.set_qform(source_qform, 2)
            source.set_sform(source_sform, 2)
            nib.save(source, source_path)

            shifted = source_sform.copy()
            shifted[1, 3] += 0.002
            nib.save(
                nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.uint8), shifted), mask_path
            )

            copy_spatial_forms(source_path, mask_path)
            fixed = nib.load(mask_path)
            expected = nib.load(source_path)
            np.testing.assert_allclose(fixed.get_qform(), expected.get_qform())
            np.testing.assert_allclose(fixed.get_sform(), expected.get_sform())
            self.assertEqual(int(fixed.header["qform_code"]), 2)
            self.assertEqual(int(fixed.header["sform_code"]), 2)
            self.assertEqual(stat.S_IMODE(mask_path.stat().st_mode), 0o664)

    def test_atomic_nifti_is_group_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.nii.gz"
            image = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
            atomic_save_nifti(path, np.ones((3, 4, 5)), image.affine, image.header)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o664)

    def test_subject_minmax_is_independent_per_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.nii.gz"
            output_path = root / "normalized.nii.gz"
            values = np.zeros((3, 3, 3), dtype=np.float32)
            values[1, 1, 1] = 10.0
            values[1, 1, 2] = 30.0
            nib.save(nib.Nifti1Image(values, np.eye(4)), source_path)

            statistics = subject_minmax_to_output(source_path, output_path)
            normalized = np.asarray(nib.load(output_path).dataobj)

            self.assertEqual(
                statistics, {"foreground_min": 0.0, "foreground_max": 30.0}
            )
            self.assertAlmostEqual(float(normalized[1, 1, 1]), 1.0 / 3.0, places=6)
            self.assertEqual(float(normalized[1, 1, 2]), 1.0)
            self.assertEqual(float(normalized[0, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
