import itertools
import os
import json

import reframe as rfm
import reframe.utility.sanity as sn

test_folders = ['test01', 'test02', 'test03', 'test04', 'test05', 'test06', 'test07', 'test08',
    'test09', 'test10', 'test11', 'test12', 'test13', 'test14', 'test15', 'test16', 'test17', 'test18',
    'test19', 'test20', 'test21', 'test22', 'test23', 'test24', 'test25', 'test26', 'test27', 'test28',
    'test29', 'test30']

@sn.deferrable
def load_json(filename):
    '''This will load a json data from a file.'''
    raw_data = sn.extractsingle(r'(?s).+', filename).evaluate()
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError as e:
        raise SanityError('failed to parse JSON file') from e

@sn.deferrable
def energy_diff(filename, data_ref):
    ''' Return the difference between obtained and reference total energies'''
    parsed_output = load_json(filename)
    return sn.abs(parsed_output['ground_state']['energy']['total'] -
                       data_ref['ground_state']['energy']['total'])

@sn.deferrable
def stress_diff(filename, data_ref):
    ''' Return the difference between obtained and reference stress tensor components'''
    parsed_output = load_json(filename)
    if 'stress' in parsed_output['ground_state'] and 'stress' in data_ref['ground_state']:
        return sn.sum(sn.abs(parsed_output['ground_state']['stress'][i][j] -
                             data_ref['ground_state']['stress'][i][j]) for i in [0, 1, 2] for j in [0, 1, 2])
    else:
        return sn.abs(0)

@sn.deferrable
def forces_diff(filename, data_ref):
    ''' Return the difference between obtained and reference atomic forces'''
    parsed_output = load_json(filename)
    if 'forces' in parsed_output['ground_state'] and 'forces' in data_ref['ground_state']:
        na = parsed_output['ground_state']['num_atoms'].evaluate()
        return sn.sum(sn.abs(parsed_output['ground_state']['forces'][i][j] -
                             data_ref['ground_state']['forces'][i][j]) for i in range(na) for j in [0, 1, 2])
    else:
        return sn.abs(0)

class sirius_scf_base_test(rfm.RunOnlyRegressionTest):
    def __init__(self, num_ranks, test_folder):
        super().__init__()
        self.descr = 'SCF check'
        self.valid_systems = ['osx', 'daint', 'linux', 'localhost']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-intel','builtin']

        self.num_tasks = num_ranks
        if self.current_system.name == 'daint':
        #    self.modules = ['PrgEnv-intel', 'cray-hdf5', 'cudatoolkit', 'gcc', 'daint-gpu', 'EasyBuild-custom/cscs',
        #                    'GSL/2.5-CrayIntel-18.08', 'libxc/4.2.3-CrayIntel-18.08', 'magma/2.4.0-CrayIntel-18.08-cuda-9.1',
        #                    'spglib/1.12.0-CrayIntel-18.08']
            self.num_tasks_per_node = 1
            self.num_cpus_per_task = 12
            self.variables = {
                'OMP_NUM_THREADS': str(self.num_cpus_per_task),
                'MKL_NUM_THREADS': str(self.num_cpus_per_task)
            }

        self.executable = 'sirius.scf'
        self.sourcesdir = '../../verification/' + test_folder

        data_ref = load_json('output_ref.json')

        fout = 'output.json'

        self.sanity_patterns = sn.all([
            sn.assert_found(r'converged after', self.stdout, msg="Calculation didn't converge"),
            sn.assert_lt(energy_diff(fout, data_ref), 1e-5, msg="Total energy is different"),
            sn.assert_lt(stress_diff(fout, data_ref), 1e-5, msg="Stress tensor is different"),
            sn.assert_lt(forces_diff(fout, data_ref), 1e-5, msg="Atomic forces are different")
        ])

        self.executable_opts = ['--output=output.json']


#@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
@rfm.simple_test
class sirius_scf_serial(sirius_scf_base_test):
    test_folder = parameter(test_folders)
    def __init__(self):
        super().__init__(1, self.test_folder)
        self.tags = {'serial'}


#@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
#class sirius_scf_serial_parallel_k(sirius_scf_base_test):
#    def __init__(self, test_folder):
#        super().__init__(2, test_folder)
#        self.tags = {'parallel_k'}
#
#
#@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
#class sirius_scf_serial_parallel_band_22(sirius_scf_base_test):
#    def __init__(self, test_folder):
#        super().__init__(4, test_folder)
#        self.tags = {'parallel_band'}
#        self.executable_opts.append('--mpi_grid=2:2')
#
#
#@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
#class sirius_scf_serial_parallel_band_12(sirius_scf_base_test):
#    def __init__(self, test_folder):
#        super().__init__(2, test_folder)
#        self.tags = {'parallel_band'}
#        self.executable_opts.append('--mpi_grid=1:2')
#
#
#@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
#class sirius_scf_serial_parallel_band_21(sirius_scf_base_test):
#    def __init__(self, test_folder):
#        super().__init__(2, test_folder)
#        self.tags = {'parallel_band'}
#        self.executable_opts.append('--mpi_grid=2:1')
#
