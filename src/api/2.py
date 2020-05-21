import re
import yaml

di = yaml.safe_load(
'''

sirius_create_context:
  return: void*
  doc: Create context of the simulation.
  full_doc: ['Simulation context is the \f[ aaaa \f] complex data structure that holds all the parameters of the individual simulation.', 'The context must be created, populated with the correct parameters and initialized before using all subsequent', 'SIRIUS functions.']
  arguments:
    fcomm:
      type: int
      atrr: in, required
      doc: Entire communicator of the simulation.
''')

print(di)
