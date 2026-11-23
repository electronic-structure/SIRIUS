"""Dataset identity and independent ultrasoft kinetic-augmentation import."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from upf2_to_json import add_kinetic_augmentation


class KineticAugmentationTest(unittest.TestCase):
    def test_import_and_rejections(self):
        upf = '<UPF>synthetic matching setup</UPF>\n'
        pp = {'pseudo_potential': {
            'header': {'pseudo_type': 'US', 'is_ultrasoft': True},
            'radial_grid': [0.01, 0.1, 1.0],
            'beta_projectors': [{'angular_momentum': 0}, {'angular_momentum': 1}],
            'core_charge_density': [0.3, 0.2, 0.0]}}
        data = {'upf_sha256': hashlib.sha256(upf.encode()).hexdigest(),
                'core_kinetic_density': [0.7, 0.4, 0.0],
                'kinetic_augmentation': [
                    {'i': i, 'j': j, 'angular_momentum': l, 'radial_function': [v, -v, 0.0]}
                    for i, j, l, v in [(0, 0, 0, 0.1), (0, 1, 1, 0.2),
                                       (1, 1, 0, 0.3), (1, 1, 2, 0.0)]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'kinetic.json'
            path.write_text(json.dumps(data))
            result = copy.deepcopy(pp)
            add_kinetic_augmentation(result, upf, path)
            self.assertEqual(result['pseudo_potential']['kinetic_augmentation'], data['kinetic_augmentation'])
            self.assertEqual(result['pseudo_potential']['core_kinetic_density'], data['core_kinetic_density'])
            for bad in [dict(data, upf_sha256='wrong'),
                        dict(data, kinetic_augmentation=data['kinetic_augmentation'][:-1]),
                        dict(data, kinetic_augmentation=data['kinetic_augmentation'] * 2),
                        dict(data, core_kinetic_density=[0, 0, 0]),
                        dict(data, core_kinetic_density=[float('inf'), 0, 0])]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    add_kinetic_augmentation(copy.deepcopy(pp), upf, path)


if __name__ == '__main__':
    unittest.main()
