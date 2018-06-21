import sirius

# test 1
M = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

Mcpp = sirius.matrix3d(M)
Gmax = 10
gvec = sirius.Gvec(Mcpp, Gmax, False)

# find via index. return
v = gvec.gvec(1111)
print(v)
print(type(v))

# find via list
idx = gvec.index_by_gvec([1, 3, 7])
print(idx)

# find via index. return
v_alt = gvec.gvec_alt(1111)
print(v_alt)
print(type(v_alt))

# just for testing
print("num_zcol = ", gvec.num_zcol())

# just for testing
print(gvec.num_gvec())


# return a dictionary
zcolu = gvec.zcol(0)
print("zcol =", zcolu)

# test just as in the examples.
for j in range(gvec.count()):
    ig = gvec.offset() + j
    G = sirius.vector3d_int(gvec.gvec(ig))
    jg = gvec.index_by_gvec(G)
    if (jg != ig):
        print("wrong index!")
