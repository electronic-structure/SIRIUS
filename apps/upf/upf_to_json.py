import json
import sys
import re
import pprint

def read_until(fin, tag):
    while True:
        line = fin.readline()
        if not line:
            print "Unexpected end of file"
            sys.exit(-1)
        if tag in line: return
        
def read_mesh_data(in_file, npoints, out_data):
    while True:
        s = in_file.readline().split()
        for k in range(len(s)): out_data.append(float(s[k]))
        if len(out_data) == npoints: return

def read_mesh_data_v2(in_file, npoints):
    out_data = []
    while True:
        s = in_file.readline().split()
        for k in range(len(s)): out_data.append(float(s[k]))
        if len(out_data) == npoints: break
    return out_data

def parse_header(upf_dict):
    
    upf_dict["header"] = {}
    
    upf = open(sys.argv[1], "r")
    
    read_until(upf, "<PP_HEADER>")

    s = upf.readline().split()
    upf_dict["header"]["version"] = int(s[0]);
    
    s = upf.readline().split()
    upf_dict["header"]["element"] = s[0]
    
    s = upf.readline().split()
    upf_dict["header"]["pp_type"] = s[0]
    
    s = upf.readline().split()
    upf_dict["header"]["nlcc"] = (s[0][0] == 'T' or s[0][0] == 't') or 0

    upf_dict["header"]["dft"] = upf.readline()[:20]

    s = upf.readline().split()
    upf_dict["header"]["zp"] = float(s[0])
    
    s = upf.readline().split()
    upf_dict["header"]["etotps"] = float(s[0])
    
    s = upf.readline().split()
    upf_dict["header"]["ecutwfc"] = float(s[0])
    upf_dict["header"]["ecutrho"] = float(s[1])

    s = upf.readline().split()
    upf_dict["header"]["lmax"] = int(s[0])
    
    s = upf.readline().split()
    upf_dict["header"]["nmesh"] = int(s[0])
    
    s = upf.readline().split()
    upf_dict["header"]["nwfc"] = int(s[0])
    upf_dict["header"]["nbeta"] = int(s[1])

    upf.readline()

    upf_dict["header"]["wavefunctions"] = []
    for i in range(upf_dict["header"]["nwfc"]):
        upf_dict["header"]["wavefunctions"].append({})
        s = upf.readline().split()
        upf_dict["header"]["wavefunctions"][i]["els"] = s[0]
        upf_dict["header"]["wavefunctions"][i]["lchi"] = int(s[1])
        upf_dict["header"]["wavefunctions"][i]["oc"] = float(s[2])
             
    upf.close();

# 
# QE subroutine read_pseudo_mesh
#
def parse_mesh(upf_dict):

    upf_dict["mesh"] = {}

    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_R>")

    # =================================================================  
    # call scan_begin (iunps, "R", .false.)  
    # read (iunps, *, err = 100, end = 100) (upf%r(ir), ir=1,upf%mesh )
    # call scan_end (iunps, "R")  
    # =================================================================
    upf_dict["mesh"]["r"] = read_mesh_data_v2(upf, upf_dict["header"]["nmesh"])

    read_until(upf, "<PP_RAB>")

    # ===================================================================
    # call scan_begin (iunps, "RAB", .false.)  
    # read (iunps, *, err = 101, end = 101) (upf%rab(ir), ir=1,upf%mesh )
    # call scan_end (iunps, "RAB")  
    # ===================================================================
    upf_dict["mesh"]["rab"] = read_mesh_data_v2(upf, upf_dict["header"]["nmesh"])

    upf.close()

#
# QE subroutine read_pseudo_nlcc
#
def parse_nlcc(upf_dict):
    
    upf_dict["rho_atc"] = []

    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_NLCC>")

    # =======================================================================
    # ALLOCATE( upf%rho_atc( upf%mesh ) )
    # read (iunps, *, err = 100, end = 100) (upf%rho_atc(ir), ir=1,upf%mesh )
    # =======================================================================
    read_mesh_data(upf, upf_dict["header"]["nmesh"], upf_dict["rho_atc"])

    upf.close()

#
# QE subroutine read_pseudo_local
#
def parse_local(upf_dict):

    upf_dict["vloc"] = []

    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_LOCAL>")

    # =================================================================
    # ALLOCATE( upf%vloc( upf%mesh ) )
    # read (iunps, *, err=100, end=100) (upf%vloc(ir) , ir=1,upf%mesh )
    # =================================================================
    read_mesh_data(upf, upf_dict["header"]["nmesh"], upf_dict["vloc"])

    upf.close()

#
# QE subroutine read_pseudo_nl 
#
def parse_non_local(upf_dict):

    upf_dict["non_local"] = {}
    upf_dict["non_local"]["beta"] = []
    upf_dict["non_local"]["dij"] = []
    upf_dict["non_local"]["qij"] = {}
    # for i in range(upf_dict["header"]["nbeta"]):
    #     upf_dict["non_local"]["qqq"].append([])
    #     for j in range(upf_dict["header"]["nbeta"]):
    #         upf_dict["non_local"]["qqq"][i].append(0.0)
        
    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_NONLOCAL>")

    # ======================================================================  
    #   do nb = 1, upf%nbeta 
    #      call scan_begin (iunps, "BETA", .false.)  
    #      read (iunps, *, err = 100, end = 100) idum, upf%lll(nb), dummy
    #      read (iunps, *, err = 100, end = 100) ikk  
    #      upf%kbeta(nb) = ikk
    #      upf%kkbeta = MAX ( upf%kkbeta, upf%kbeta(nb) )  
    #      read (iunps, *, err = 100, end = 100) (upf%beta(ir,nb), ir=1,ikk)
    # 
    #      read (iunps, *, err=200,iostat=ios) upf%rcut(nb), upf%rcutus(nb)
    #      read (iunps, *, err=200,iostat=ios) upf%els_beta(nb)
    #      call scan_end (iunps, "BETA")  
    # 200  continue
    #   enddo
    # ======================================================================
    for i in range(upf_dict["header"]["nbeta"]):
        read_until(upf, "<PP_BETA>")
        upf_dict["non_local"]["beta"].append({})
        s = upf.readline().split()
        upf_dict["non_local"]["beta"][i]["lll"] = int(s[1])
        s = upf.readline()
        upf_dict["non_local"]["beta"][i]["kbeta"] = int(s)

        upf_dict["non_local"]["beta"][i]["beta"] = []
        read_mesh_data(upf, upf_dict["non_local"]["beta"][i]["kbeta"], upf_dict["non_local"]["beta"][i]["beta"])
        
        line = upf.readline()
        if not "</PP_BETA>" in line:
            upf_dict["non_local"]["beta"][i]["rcut"] = float(s[0])
            upf_dict["non_local"]["beta"][i]["rcutus"] = float(s[0])
            line = upf.readline()
            if not "</PP_BETA>" in line: upf_dict["non_local"]["beta"][i]["els_beta"] = float(s[0])

    # ================================================================
    # call scan_begin (iunps, "DIJ", .false.)  
    # read (iunps, *, err = 101, end = 101) nd, dummy  
    # do icon = 1, nd
    #    read (iunps, *, err = 101, end = 101) nb, mb, upf%dion(nb,mb)
    #    upf%dion (mb,nb) = upf%dion (nb,mb)  
    # enddo
    # call scan_end (iunps, "DIJ")  
    # ================================================================
    read_until(upf, "<PP_DIJ>")
    s = upf.readline().split()
    nd = int(s[0])
    for k in range(nd):
        s = upf.readline().split()
        upf_dict["non_local"]["dij"].append({})
        upf_dict["non_local"]["dij"][k]["ij"] = [int(s[0]) - 1, int(s[1]) - 1]
        upf_dict["non_local"]["dij"][k]["dion"] = float(s[2])

    # =============================================    
    # call scan_begin (iunps, "QIJ", .false.)  
    # read (iunps, *, err = 102, end = 102) upf%nqf
    # upf%nqlc = 2 * upf%lmax  + 1
    # =============================================
    read_until(upf, "<PP_QIJ>")
    s = upf.readline().split()
    upf_dict["non_local"]["qij"]["nqf"] = int(s[0])
    nqlc = upf_dict["header"]["lmax"] * 2 + 1

    # =======================================================================
    # if ( upf%nqf /= 0) then
    #    call scan_begin (iunps, "RINNER", .false.)  
    #    read (iunps,*,err=103,end=103) ( idum, upf%rinner(i), i=1,upf%nqlc )
    #    call scan_end (iunps, "RINNER")  
    # end if
    # =======================================================================
    if upf_dict["non_local"]["qij"]["nqf"] != 0:
        read_until(upf, "<PP_RINNER>")
        upf_dict["non_local"]["qij"]["rinner"] = []
        for i in range(nqlc):
            s = upf.readline().split()
            upf_dict["non_local"]["qij"]["rinner"].append(float(s[1]))
        read_until(upf, "</PP_RINNER>")
    
    # ================================================================================== 
    # do nb = 1, upf%nbeta
    #    do mb = nb, upf%nbeta
    #       read (iunps,*,err=102,end=102) idum, idum, ldum, dummy
    #       !"  i    j   (l)"
    #       if (ldum /= upf%lll(mb) ) then
    #         call errore ('read_pseudo_nl','inconsistent angular momentum for Q_ij', 1)
    #       end if
    #       read (iunps,*,err=104,end=104) upf%qqq(nb,mb), dummy
    #       ! "Q_int"
    #       upf%qqq(mb,nb) = upf%qqq(nb,mb)  
    #       ! ijv is the combined (nb,mb) index
    #       ijv = mb * (mb-1) / 2 + nb
    #       IF (upf%q_with_l .or. upf%tpawp) THEN
    #          l1=upf%lll(nb)
    #          l2=upf%lll(mb)
    #          DO l=abs(l1-l2),l1+l2
    #             read (iunps, *, err=105, end=105) (upf%qfuncl(n,ijv,l), &
    #                                                n=1,upf%mesh)
    #          END DO
    #       ELSE
    #          read (iunps, *, err=105, end=105) (upf%qfunc(n,ijv), n=1,upf%mesh)
    #       ENDIF
    # ==================================================================================
    upf_dict["non_local"]["qij"]["q"] = []
    for i in range(upf_dict["header"]["nbeta"]):
        for j in range(i, upf_dict["header"]["nbeta"]):
            
            s = upf.readline().split()
            if int(s[2]) != upf_dict["non_local"]["beta"][j]["lll"]:
                print "inconsistent angular momentum"
                sys.exit(-1)

            if int(s[0]) != i + 1 or int(s[1]) != j + 1: 
                print "inconsistent ij indices"
                sys.exit(-1)

            upf_dict["non_local"]["qij"]["q"].append({})
            upf_dict["non_local"]["qij"]["q"][-1]["ij"] = [i, j]
            upf_dict["non_local"]["qij"]["q"][-1]["qfunc"] = []
            
            s = upf.readline().split()
            upf_dict["non_local"]["qij"]["q"][-1]["qqq"] = float(s[0])

            read_mesh_data(upf, upf_dict["header"]["nmesh"], upf_dict["non_local"]["qij"]["q"][-1]["qfunc"])
            
            # ======================================================================
            # if ( upf%nqf > 0 ) then
            #   call scan_begin (iunps, "QFCOEF", .false.)  
            #   read (iunps,*,err=106,end=106) &
            #             ( ( upf%qfcoef(i,lp,nb,mb), i=1,upf%nqf ), lp=1,upf%nqlc )
            #   do i = 1, upf%nqf
            #      do lp = 1, upf%nqlc
            #         upf%qfcoef(i,lp,mb,nb) = upf%qfcoef(i,lp,nb,mb)
            #      end do
            #   end do
            #   call scan_end (iunps, "QFCOEF")  
            # end if
            #=======================================================================
            if upf_dict["non_local"]["qij"]["nqf"] > 0:
                read_until(upf, "<PP_QFCOEF>")
                upf_dict["non_local"]["qij"]["q"][-1]["qfcoef"] = []

                read_mesh_data(upf, upf_dict["non_local"]["qij"]["nqf"] * nqlc, upf_dict["non_local"]["qij"]["q"][-1]["qfcoef"])

                read_until(upf, "</PP_QFCOEF>")

    upf.close()


#
# QE subroutine read_pseudo_pswfc
#
def parse_pswfc(upf_dict):
    
    upf_dict["pswfc"] = []

    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_PSWFC>")

    # =======================================================================
    # ALLOCATE( upf%chi( upf%mesh, MAX( upf%nwfc, 1 ) ) )
    # upf%chi = 0.0_DP
    # do nb = 1, upf%nwfc  
    #    read (iunps, *, err=100, end=100) dummy  !Wavefunction labels
    #    read (iunps, *, err=100, end=100) ( upf%chi(ir,nb), ir=1,upf%mesh )
    # enddo
    # =======================================================================
    for i in range(upf_dict["header"]["nwfc"]):
        upf.readline()
        upf_dict["pswfc"].append([])
        read_mesh_data(upf, upf_dict["header"]["nmesh"], upf_dict["pswfc"][-1])

    upf.close()

#
# QE subroutine read_pseudo_rhoatom
#
def parse_rhoatom(upf_dict):

    upf_dict["rho_at"] = []

    upf = open(sys.argv[1], "r")

    read_until(upf, "<PP_RHOATOM>")

    # ================================================================
    # ALLOCATE( upf%rho_at( upf%mesh ) )
    # read (iunps,*,err=100,end=100) ( upf%rho_at(ir), ir=1,upf%mesh )
    # ================================================================
    read_mesh_data(upf, upf_dict["header"]["nmesh"], upf_dict["rho_at"])

    upf.close()

    
def main():

    upf_dict = {}

    parse_header(upf_dict)
    parse_mesh(upf_dict)
    if upf_dict["header"]["nlcc"] == 1: parse_nlcc(upf_dict)
    parse_local(upf_dict)
    parse_non_local(upf_dict)
    parse_pswfc(upf_dict)
    parse_rhoatom(upf_dict)

    fout = open("pp.json", "w")
    # regular expression magic: match comma,space,newline ,\s\n with the follwing conditions: 
    # digit before (?<=[0-9]) and arbitrary number of spaces and minus or digit after (?=\s*[-|0-9])
    # fout.write(re.sub(r"(?<=[0-9]),\s\n(?=\s*[-|0-9])", r",", json.dumps(upf_dict, indent=2)))

    # this is even better
    fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(upf_dict, indent=2)))
    fout.close()

if __name__ == "__main__":
    main()
