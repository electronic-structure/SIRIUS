"""Small synthetic tests for the optional ATOMPAW core-tau import."""

import copy
from ctypes.util import find_library
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from upf2_to_json import add_paw_core_kinetic_density
from upf_to_json import main


class PawCoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'matching core.xml'
        self.grid = [0.0, 0.5, 1.0]
        self.fields = {
            'ae_core_density': [4.0, 3.0, 2.0],
            'pseudo_core_density': [2.0, 1.5, 1.0],
            'ae_core_kinetic_energy_density': [3.0, 2.5, 2.0],
            'pseudo_core_kinetic_energy_density': [1.0, 0.75, 0.5],
        }
        self.root = ET.fromstring('''<paw_dataset version="0.7">
          <atom symbol="C" Z="6" core="2" valence="4"/>
          <generator name="atompaw-4.2.0.4"/>
          <radial_grid id="r"><values>0 .5 1</values></radial_grid>
        </paw_dataset>''')
        for tag, values in self.fields.items():
            ET.SubElement(self.root, tag, grid='r').text = ' '.join(
                str(v * math.sqrt(4 * math.pi)) for v in values)
        self.data = {'pseudo_potential': {
            'header': {'pseudo_type': 'PAW', 'element': 'C', 'z_valence': 4,
                       'core_correction': True},
            'radial_grid': self.grid[:],
            'core_charge_density': self.fields['pseudo_core_density'][:],
            'paw_data': {'ae_core_charge_density': self.fields['ae_core_density'][:]},
        }}

    def run_import(self):
        ET.ElementTree(self.root).write(self.path)
        add_paw_core_kinetic_density(self.data, self.path)

    def test_exact_grid_units_and_idempotence(self):
        original = copy.deepcopy(self.data)
        self.run_import()
        self.run_import()
        pp = self.data['pseudo_potential']
        for owner, key, tag in [
                (pp['paw_data'], 'ae_core_kinetic_density', 'ae_core_kinetic_energy_density'),
                (pp, 'core_kinetic_density', 'pseudo_core_kinetic_energy_density')]:
            for actual, expected in zip(owner.pop(key), self.fields[tag]):
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(self.data, original)

    @unittest.skipUnless(find_library('gsl'), 'Shared GSL unavailable')
    def test_resampling(self):
        pp = self.data['pseudo_potential']
        pp['radial_grid'] = [0.0, 0.1, 0.5, 0.8, 1.0]
        pp['paw_data']['ae_core_charge_density'] = [4 - 2*r for r in pp['radial_grid']]
        pp['core_charge_density'] = [2 - r for r in pp['radial_grid']]
        self.run_import()
        for radius, actual in zip(pp['radial_grid'], pp['paw_data']['ae_core_kinetic_density']):
            self.assertAlmostEqual(actual, 3 - radius)
        for radius, actual in zip(pp['radial_grid'], pp['core_kinetic_density']):
            self.assertAlmostEqual(actual, 1 - radius/2)

    def test_separate_field_grid(self):
        radial = copy.deepcopy(self.root.find('radial_grid'))
        radial.set('id', 'tau-grid')
        self.root.append(radial)
        self.root.find('ae_core_kinetic_energy_density').set('grid', 'tau-grid')
        self.run_import()

    def test_core_free_requires_explicit_zero_tau(self):
        self.root.find('atom').set('valence', '6')
        self.root.find('atom').set('core', '0')
        for tag in self.fields:
            self.root.find(tag).text = '0 0 0'
        pp = self.data['pseudo_potential']
        pp['header']['z_valence'] = 6
        pp['core_charge_density'] = [0.0] * 3
        pp['paw_data']['ae_core_charge_density'] = [0.0] * 3
        self.run_import()
        self.assertEqual(pp['paw_data']['ae_core_kinetic_density'], [0.0] * 3)

    def test_no_pseudo_core_correction(self):
        pp = self.data['pseudo_potential']
        pp['header']['core_correction'] = False
        del pp['core_charge_density']
        for tag in ('pseudo_core_density', 'pseudo_core_kinetic_energy_density'):
            self.root.find(tag).text = '0 0 0'
        self.run_import()
        self.assertEqual(pp['core_kinetic_density'], [0.0] * 3)
        self.assertNotIn('core_charge_density', pp)

    def test_invalid_xml_leaves_dataset_unchanged(self):
        root = copy.deepcopy(self.root)
        for tag, attribute, value in [
                ('atom', 'symbol', 'N'), ('atom', 'valence', '5'),
                ('atom', 'core', '0'), ('atom', 'Z', 'nan'),
                ('generator', 'name', 'unknown'),
                ('radial_grid/values', None, '0 0 1'),
                ('radial_grid/values', None, '0 nan 1'),
                ('radial_grid/values', None, '0 1'),
                ('radial_grid/values', None, '.1 .5 1'),
                ('ae_core_density', None, '0 0 0'),
                ('pseudo_core_density', None, '0 0 0'),
                ('ae_core_kinetic_energy_density', None, '1 2'),
                ('ae_core_kinetic_energy_density', None, '0 0 0'),
                ('pseudo_core_kinetic_energy_density', None, '0 0 0'),
                ('ae_core_kinetic_energy_density', None, '-1 1 1'),
                ('pseudo_core_kinetic_energy_density', None, '1 inf 1'),
                ('ae_core_kinetic_energy_density', 'grid', 'absent')]:
            with self.subTest(tag=tag, attribute=attribute, value=value):
                self.root = copy.deepcopy(root)
                node = self.root.find(tag)
                if attribute:
                    node.set(attribute, value)
                else:
                    node.text = value
                original = copy.deepcopy(self.data)
                with self.assertRaises(ValueError):
                    self.run_import()
                self.assertEqual(self.data, original)

    def test_missing_or_duplicate_fields(self):
        for tag in ['atom', 'generator', 'radial_grid', *self.fields]:
            for duplicate in (False, True):
                with self.subTest(tag=tag, duplicate=duplicate):
                    saved = copy.deepcopy(self.root)
                    if duplicate:
                        self.root.append(copy.deepcopy(self.root.find(tag)))
                    else:
                        self.root.remove(self.root.find(tag))
                    with self.assertRaises(ValueError):
                        self.run_import()
                    self.root = saved

    def test_malformed_xml(self):
        self.path.write_text('<paw_dataset>')
        with self.assertRaisesRegex(ValueError, 'Invalid PAW-XML'):
            add_paw_core_kinetic_density(self.data, self.path)

    def test_reject_non_paw_and_conflicting_existing_tau(self):
        pp = self.data['pseudo_potential']
        pp['header']['pseudo_type'] = 'NC'
        with self.assertRaisesRegex(ValueError, 'PAW UPF'):
            self.run_import()
        pp['header']['pseudo_type'] = 'PAW'
        pp['core_kinetic_density'] = [99.0] * 3
        original = copy.deepcopy(self.data)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.run_import()
        self.assertEqual(original, self.data)

    def test_missing_gsl(self):
        self.data['pseudo_potential']['radial_grid'] = [0, .4, 1]
        with patch('ctypes.util.find_library', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'GSL shared library'):
                self.run_import()

    def test_cli_optional_import_and_output_preserved_on_error(self):
        upf = Path(self.tmp.name) / 'source file.UPF'
        upf.write_text('test')
        output = Path(self.tmp.name) / 'output file.json'
        ET.ElementTree(self.root).write(self.path)
        argv = ['upf_to_json.py', str(upf), '-o', str(output)]
        with patch('upf_to_json.parse_upf_from_string', side_effect=lambda _: copy.deepcopy(self.data)):
            with patch('sys.argv', argv):
                main()
            plain = json.loads(output.read_text())['pseudo_potential']
            self.assertNotIn('core_kinetic_density', plain)
            argv += ['--paw-core-xml', str(self.path)]
            with patch('sys.argv', argv):
                main()
            self.assertIn('core_kinetic_density', json.loads(output.read_text())['pseudo_potential'])
            before = output.read_bytes()
            self.root.remove(self.root.find('ae_core_kinetic_energy_density'))
            ET.ElementTree(self.root).write(self.path)
            with patch('sys.argv', argv), patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(before, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
