import itertools
import os
import json

import reframe as rfm
import reframe.utility.sanity as sn

test_folders = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8',
    'test9', 'test10', 'test11', 'test12', 'test13', 'test14', 'test15', 'test16', 'test17']


@sn.sanity_function
def load_json(filename):
    '''This will load a json data from a file.'''
    raw_data = sn.extractsingle(r'(?s).+', filename).evaluate()
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError as e:
        raise SanityError('failed to parse JSON file') from e

@sn.sanity_function
def energy_diff(filename, data_ref):
    ''' Return the difference between obtained and reference total energies'''
    parsed_output = load_json(filename)
    return sn.abs(parsed_output['ground_state']['energy']['total'] -
                       data_ref['ground_state']['energy']['total'])

@sn.sanity_function
def stress_diff(filename, data_ref):
    ''' Return the difference between obtained and reference stress tensor components'''
    parsed_output = load_json(filename)
    if 'stress' in parsed_output['ground_state'] and 'stress' in data_ref['ground_state']:
        return sn.sum(sn.abs(parsed_output['ground_state']['stress'][i][j] -
                             data_ref['ground_state']['stress'][i][j]) for i in [0, 1, 2] for j in [0, 1, 2])
    else:
        return sn.abs(0)

class sirius_scf_base_test(rfm.RunOnlyRegressionTest):
    def __init__(self, num_ranks, test_folder):
        super().__init__()
        self.descr = 'SCF check'
        self.valid_systems = ['osx']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.num_tasks = num_ranks

        self.executable = 'sirius.scf'
        self.sourcesdir = '../../verification/' + test_folder

        data_ref = load_json('output_ref.json')

        fout = 'output.json'

        self.sanity_patterns = sn.all([
            sn.assert_found(r'converged after', self.stdout, msg="Calculation didn't converge"),
            sn.assert_lt(energy_diff(fout, data_ref), 1e-5, msg="Total energy is different"),
            sn.assert_lt(stress_diff(fout, data_ref), 1e-5, msg="Stress tensor is different")
        ])

        self.executable_opts = ['--output=output.json']


@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
class sirius_scf_serial(sirius_scf_base_test):
    def __init__(self, test_folder):
        super().__init__(1, test_folder)
        self.tags = {'serial'}


@rfm.parameterized_test(*([test_folder] for test_folder in test_folders))
class sirius_scf_serial_parallel_k(sirius_scf_base_test):
    def __init__(self, test_folder):
        super().__init__(2, test_folder)
        self.tags = {'parallel_k'}


