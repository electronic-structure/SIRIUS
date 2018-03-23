import sirius
sirius.initialize()

#test 1
M = [[1,0,0],[0,1,0],[0,0,1]]

Mcpp = sirius.matrix3d(M)
Gmax = 10
gvec = sirius.Gvec(Mcpp, Gmax, False)


v = gvec.gvec(1111)
print(type(v))



#gvec.index_by_gvec([1,2,3])

gvec.zcol(gvec, 123)

for j in range(gvec.count()):
    ig = gvec.offset() + j
    G = sirius.vector3d_int(gvec.gvec(ig))
    jg = gvec.index_by_gvec(G)
    if(jg != ig):
        print("wrong index!")

sirius.finalize()
