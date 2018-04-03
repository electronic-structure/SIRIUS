import sirius
import json

param = {
    "parameters" : {
        "pw_cutoff" : 33.0
    }
}

ctx = sirius.Simulation_context(json.dumps(param))
print(ctx.pw_cutoff())
ctx.set_pw_cutoff(12)
print(ctx.pw_cutoff)
