
class ReframeSettings:
    reframe_module = 'reframe'
    job_poll_intervals = [1, 2, 3]
    job_submit_timeout = 60
    checks_path = ['checks/']
    checks_path_recurse = False
    site_configuration = {
        'systems': {
            'osx': {
                'descr': 'OSX notebook with MacPort',
                'hostnames': ['localhost'],
                'modules_system': None,
                'resourcesdir': '',
                'partitions': {
                    'cpu': {
                        'scheduler': 'local+mpirun',
                        'environs': ['PrgEnv-gnu'],
                        'descr': 'CPU execution',
                        'max_jobs': 1
                    }
                }
            }
        },

        'environments': {
            'osx': {
                'PrgEnv-gnu': {
                    'type': 'ProgEnvironment',
                    'modules': [],
                    'cc':  'mpicc',
                    'cxx': 'mpic++',
                    'ftn': 'mpif90',
                }
            }
        }
    }

    logging_config = {
        'level': 'DEBUG',
        'handlers': [
            {
                'type': 'file',
                'name': 'reframe.log',
                'level': 'DEBUG',
                'format': '[%(asctime)s] %(levelname)s: '
                          '%(check_info)s: %(message)s',
                'append': False,
            },

            # Output handling
            {
                'type': 'stream',
                'name': 'stdout',
                'level': 'INFO',
                'format': '%(message)s'
            },
            {
                'type': 'file',
                'name': 'reframe.out',
                'level': 'INFO',
                'format': '%(message)s',
                'append': False,
            }
        ]
    }

    #perf_logging_config = {
    #    'level': 'DEBUG',
    #    'handlers': [
    #        #@ {
    #        #@     'type': 'graylog',
    #        #@     'host': 'your-server-here',
    #        #@     'port': 12345,
    #        #@     'level': 'INFO',
    #        #@     'format': '%(message)s',
    #        #@     'extras': {
    #        #@         'facility': 'reframe',
    #        #@         'data-version': '1.0',
    #        #@     }
    #        #@ },
    #        {
    #            'type': 'filelog',
    #            'prefix': '%(check_system)s/%(check_partition)s',
    #            'level': 'INFO',
    #            'format': (
    #                '%(asctime)s|reframe %(version)s|'
    #                '%(check_info)s|jobid=%(check_jobid)s|'
    #                '%(check_perf_var)s=%(check_perf_value)s|'
    #                'ref=%(check_perf_ref)s '
    #                '(l=%(check_perf_lower_thres)s, '
    #                'u=%(check_perf_upper_thres)s)|'
    #                '%(check_perf_unit)s'
    #            ),
    #            'append': True
    #        }
    #    ]
    #}

settings = ReframeSettings()

