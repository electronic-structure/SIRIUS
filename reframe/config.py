site_configuration = {
    'systems': [
        {
            'name' : 'localhost',
            'descr': 'local workstation',
            'hostnames': ['localhost'],
            'resourcesdir': '',
            'partitions': [
                {
                    'name' : 'cpu',
                    'scheduler': 'local',
                    'launcher' : 'local',
                    'environs': ['builtin'],
                    'descr': 'CPU execution',
                    'max_jobs': 1
                }
            ]
        }
    ]
}
#site_configuration = {
#    'systems': [
#        {
#            'name' : 'osx',
#            'descr': 'OSX notebook with MacPort',
#            'hostnames': ['localhost'],
#            'resourcesdir': '',
#            'partitions': [
#                {
#                    'name' : 'cpu',
#                    'scheduler': 'local',
#                    'launcher' : 'mpirun',
#                    'environs': ['builtin'],
#                    'descr': 'CPU execution',
#                    'max_jobs': 1
#                }
#            ]
#        },
#        {
#            'name' : 'linux',
#            'descr': 'Ubuntu linux box',
#            'hostnames': ['localhost'],
#            'resourcesdir': '',
#            'partitions': [
#                {
#                    'name' : 'cpu',
#                    'scheduler': 'local',
#                    'launcher' : 'local',
#                    'environs': ['builtin'],
#                    'descr': 'CPU execution',
#                    'max_jobs': 1
#                }
#            ]
#        }
#    ],
#    'environments': [
#        {
#            'name': 'builtin',
#            'target_systems': ['osx:cpu', 'linux:cpu'],
#            'cc': 'mpicc',
#            'cxx': 'mpic++',
#            'ftn': 'mpif90'
#        }
#    ],
#    'logging': [
#        {
#            'level': 'debug',
#            'handlers': [
#                {
#                    'type': 'file',
#                    'name': 'reframe.log',
#                    'level': 'debug',
#                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
#                    'append': False
#                },
#                {
#                    'type': 'stream',
#                    'name': 'stdout',
#                    'level': 'info',
#                    'format': '%(message)s'
#                },
#                {
#                    'type': 'file',
#                    'name': 'reframe.out',
#                    'level': 'info',
#                    'format': '%(message)s',
#                    'append': False
#                }
#            ],
#            'handlers_perflog': [
#                {
#                    'type': 'filelog',
#                    'prefix': '%(check_system)s/%(check_partition)s',
#                    'level': 'info',
#                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|num_tasks=%(check_num_tasks)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',   # noqa: E501
#                    'datefmt': '%FT%T%:z',
#                    'append': True
#                },
#                {
#                    'type': 'graylog',
#                    'address': 'graylog-server:12345',
#                    'level': 'info',
#                    'format': '%(message)s',
#                    'extras': {
#                        'facility': 'reframe',
#                        'data-version': '1.0',
#                    }
#                }
#            ]
#        }
#    ],
#    'general': [
#        {
#            'check_search_path': [
#                'checks/'
#            ],
#            'check_search_recursive': True
#        }
#    ]
#}
#
