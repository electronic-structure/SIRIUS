#!/usr/bin/env -S python -u -m mpi4py
from sirius.nlcg import run, validate_config
import argparse
# from sirius import save_state
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--sirius-config', '-s', default='sirius.json')
    parser.add_argument('--input', '-i', default='nlcg.yaml')
    parser.add_argument('--dump-on-error', '-e',
                        action='store_true', default=False)

    args = parser.parse_args()
    yaml_config = args.input
    ycfg = yaml.safe_load(open(yaml_config, 'r'))
    # validate CG-config
    ycfg['CG'] = validate_config(ycfg['CG'])

    def callback(kset,
                 interval=ycfg['CG']['callback_interval'],
                 **kwargs):
        # E = kwargs['E']

        def _callback(fn, it, **kwargs):
            if it % interval == 0:
                pass
                # kset.ctx().create_storage_file()
                # store_density_potential(E.density, E.potential)
                # mag_mom = E.density.compute_atomic_mag_mom()
                # save_state({'f': fn, 'ek': kset.e, 'mag_mom': mag_mom},
                #            kset=kset, prefix='fn_%05d_' % it)
                # save_state({'f': fn, 'ek': kset.e},
                #            kset=kset, prefix='fn_%05d_' % it)
        return _callback

    def error_callback(kset, **kwargs):
        # E = kwargs['E']

        if args.dump_on_error:
            def _callback(prefix='nlcg_dump', **kwargs):
                pass
                # kset.ctx().create_storage_file()
                # store_density_potential(E.density, E.potential)
                # mag_mom = E.density.compute_atomic_mag_mom()
                # dump_dict = {**kwargs, 'mag_mom': mag_mom}
                # save_state(dump_dict,
                #            kset=kset, prefix=prefix)
        else:
            def _callback(*args, **kwargs):
                return None
        return _callback

    run(ycfg, args.sirius_config,
        callback=callback,
        error_callback=error_callback)
