#import sirius
import json
import simulation_context as sc

param = {
    "parameters" : {
        "pw_cutoff" : 33.0
    }
}

#sc.simulation_context()

ctx = sc.Simulation_context(json.dumps(param))
#ctx = sc.Simulation_context(param) #old version
#print(json.dumps(param))
print(ctx.pw_cutoff())
ctx.set_pw_cutoff(12) #works
print(ctx.pw_cutoff())
