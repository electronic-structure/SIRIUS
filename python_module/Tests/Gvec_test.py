import sirius
sirius.initialize()

#test 1
M = [[1,0,0],[0,1,0],[0,0,1]]

Mcpp = sirius.matrix3d(M)
Gmax = 10
gvec = sirius.Gvec(Mcpp, Gmax, False)


v = gvec.gvec(1111)
print(v)
print(type(v))

idx = gvec.index_by_gvec([1,3,7])
#print(idx)

v_alt = gvec.gvec_alt(1111)
print(v_alt)
print(type(v_alt))

#gvec.zcol(gvec, 123)

for j in range(gvec.count()):
    ig = gvec.offset() + j
    G = sirius.vector3d_int(gvec.gvec(ig))
    jg = gvec.index_by_gvec(G)
    if(jg != ig):
        print("wrong index!")

sirius.finalize()
