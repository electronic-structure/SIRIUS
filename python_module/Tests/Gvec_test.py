import sirius
sirius.initialize()

#test 1
M = [[1,0,0],[0,1,0],[0,0,1]]

Mcpp = sirius.matrix3d(M)
Gmax = 10
gvec = sirius.Gvec(Mcpp, Gmax, False)


v = gvec.gvec(1111) #find via index. return
print(v)
print(type(v))

idx = gvec.index_by_gvec([1,3,7]) #find via list
print(idx)

v_alt = gvec.gvec_alt(1111) #find via index. return
print(v_alt)
print(type(v_alt))

print("num_zcol = ", gvec.num_zcol()) #just for testing

print(gvec.num_gvec()) #just for testing

zcolu = gvec.zcol(0) #return a dictionary
print("zcol =", zcolu)

for j in range(gvec.count()): #test just as in the examples.
    ig = gvec.offset() + j
    G = sirius.vector3d_int(gvec.gvec(ig))
    jg = gvec.index_by_gvec(G)
    if(jg != ig):
        print("wrong index!")

sirius.finalize()
