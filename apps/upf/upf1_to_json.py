# Copyright (c) 2013 Anton Kozhevnikov, Thomas Schulthess
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import sys
import re
import io

def read_until(fin, tag):
    while True:
        line = fin.readline()
        if not line:
            print ("Unexpected end of file")
            sys.exit(-1)
        if tag in line: return

def read_mesh_data(in_file, npoints):
    out_data = []
    while True:
        s = in_file.readline().split()
        for k in range(len(s)): out_data.append(float(s[k]))
        if len(out_data) == npoints: break
    return out_data

def parse_header(upf_dict, upf_str):

    #print "parsing header"

    upf_dict["header"] = {}
    upf = io.StringIO(upf_str)
    read_until(upf, "<PP_HEADER>")

    s = upf.readline().split()
    #upf_dict["header"]["version"] = int(s[0]);

    s = upf.readline().split()
    upf_dict['header']['element'] = s[0].strip()

    s = upf.readline().split()
    upf_dict['header']['pseudo_type'] = s[0]

    s = upf.readline().split()
    upf_dict['header']['core_correction'] = (s[0][0] == 'T' or s[0][0] == 't') or 0

    s = upf.readline();
    #upf_dict["header"]["dft"] = s[:20]

    s = upf.readline().split()
    upf_dict['header']['z_valence'] = float(s[0])

    s = upf.readline().split()
    #upf_dict["header"]["etotps"] = float(s[0])

    s = upf.readline().split()
    #upf_dict["header"]["ecutwfc"] = float(s[0])
    #upf_dict["header"]["ecutrho"] = float(s[1])

    s = upf.readline().split()
    upf_dict['header']['l_max'] = int(s[0])

    s = upf.readline().split()
    upf_dict['header']['mesh_size'] = int(s[0])

    s = upf.readline().split()
    upf_dict["header"]["number_of_wfc"] = int(s[0])
    upf_dict['header']['number_of_proj'] = int(s[1])

    upf.readline()

    #upf_dict["header"]["wavefunctions"] = []
    #for i in range(upf_dict["header"]["nwfc"]):
    #    upf_dict["header"]["wavefunctions"].append({})
    #    s = upf.readline().split()
    #    upf_dict["header"]["wavefunctions"][i]["els"] = s[0]
    #    upf_dict["header"]["wavefunctions"][i]["lchi"] = int(s[1])
    #    upf_dict["header"]["wavefunctions"][i]["oc"] = float(s[2])

    upf.close();

#
# QE subroutine read_pseudo_mesh
#
def parse_mesh(upf_dict, upf_str):

    #print "parsing mesh"

    upf = io.StringIO(upf_str)
    read_until(upf, "<PP_R>")

    # =================================================================
    # call scan_begin (iunps, "R", .false.)
    # read (iunps, *, err = 100, end = 100) (upf%r(ir), ir=1,upf%mesh )
    # call scan_end (iunps, "R")
    # =================================================================
    upf_dict['radial_grid'] = read_mesh_data(upf, upf_dict['header']['mesh_size'])

    read_until(upf, "<PP_RAB>")

    # ===================================================================
    # call scan_begin (iunps, "RAB", .false.)
    # read (iunps, *, err = 101, end = 101) (upf%rab(ir), ir=1,upf%mesh )
    # call scan_end (iunps, "RAB")
    # ===================================================================
    #upf_dict["mesh"]["rab"] = read_mesh_data(upf, upf_dict["header"]["nmesh"])

    upf.close()

#
# QE subroutine read_pseudo_nlcc
#
def parse_nlcc(upf_dict, upf_str):

    #print "parsing nlcc"

    upf = io.StringIO(upf_str)
    read_until(upf, "<PP_NLCC>")

    # =======================================================================
    # ALLOCATE( upf%rho_atc( upf%mesh ) )
    # read (iunps, *, err = 100, end = 100) (upf%rho_atc(ir), ir=1,upf%mesh )
    # =======================================================================
    upf_dict['core_charge_density'] = read_mesh_data(upf, upf_dict['header']['mesh_size'])

    upf.close()

#
# QE subroutine read_pseudo_local
#
def parse_local(upf_dict, upf_str):

    #print "parsing local"

    upf_dict['local_potential'] = []
    upf = io.StringIO(upf_str)

    read_until(upf, "<PP_LOCAL>")

    # =================================================================
    # ALLOCATE( upf%vloc( upf%mesh ) )
    # read (iunps, *, err=100, end=100) (upf%vloc(ir) , ir=1,upf%mesh )
    # =================================================================
    vloc = read_mesh_data(upf, upf_dict['header']['mesh_size'])
    upf_dict['local_potential'] = [v / 2 for v in vloc]

    upf.close()

#
# QE subroutine read_pseudo_nl
#
def parse_non_local(upf_dict, upf_str):

    #print "parsing non-local"

    upf_dict['beta_projectors'] = []
    upf_dict['D_ion'] = []

    upf = io.StringIO(upf_str)
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
    for i in range(upf_dict['header']['number_of_proj']):
        read_until(upf, "<PP_BETA>")
        upf_dict['beta_projectors'].append({})
        upf_dict['beta_projectors'][i]['label'] = ''
        s = upf.readline().split()
        upf_dict['beta_projectors'][i]['angular_momentum'] = int(s[1])
        s = upf.readline()
        #upf_dict['beta_projectors'][i]['cutoff_radius_index'] = int(s)
        nr = int(s)
        beta = read_mesh_data(upf, nr)

        upf_dict['beta_projectors'][i]['radial_function'] = beta #read_mesh_data(upf, upf_dict['beta_projectors'][i]['cutoff_radius_index'])

        upf_dict['beta_projectors'][i]['cutoff_radius'] = 0.0
        upf_dict['beta_projectors'][i]['ultrasoft_cutoff_radius'] = 0.0

        #line = upf.readline()
        #if not "</PP_BETA>" in line:
            #s = line.split()
            #upf_dict['beta_projectors'][i]['cutoff_radius'] = float(s[0])
            #upf_dict['beta_projectors'][i]['ultrasoft_cutoff_radius'] = float(s[1])
            #s = upf.readline().split()
            #if not "</PP_BETA>" in line: upf_dict['beta_projectors'][i]['els_beta'] = float(s[0])

    # ================================================================
    # call scan_begin (iunps, "DIJ", .false.)
    # read (iunps, *, err = 101, end = 101) nd, dummy
    # do icon = 1, nd
    #    read (iunps, *, err = 101, end = 101) nb, mb, upf%dion(nb,mb)
    #    upf%dion (mb,nb) = upf%dion (nb,mb)
    # enddo
    # call scan_end (iunps, "DIJ")
    # ================================================================
    nb = upf_dict['header']['number_of_proj']
    dij = [0 for i in range(nb * nb)]
    read_until(upf, "<PP_DIJ>")
    s = upf.readline().split()
    nd = int(s[0])
    for k in range(nd):
        s = upf.readline().split()
        i = int(s[0]) - 1
        j = int(s[1]) - 1
        dij[i * nb + j] = float(s[2]) / 2 # convert to Hartree
        dij[j * nb + i] = float(s[2]) / 2 # convert to Hartree

    upf_dict['D_ion'] = dij

    if upf_dict['header']['pseudo_type'] != "US": return

    upf_dict['augmentation'] = []

    # =============================================
    # call scan_begin (iunps, "QIJ", .false.)
    # read (iunps, *, err = 102, end = 102) upf%nqf
    # upf%nqlc = 2 * upf%lmax  + 1
    # =============================================
    read_until(upf, "<PP_QIJ>")
    s = upf.readline().split()
    num_q_coef = int(s[0])
    #nqlc = upf_dict['header']['l_max'] * 2 + 1

    # =======================================================================
    # if ( upf%nqf /= 0) then
    #    call scan_begin (iunps, "RINNER", .false.)
    #    read (iunps,*,err=103,end=103) ( idum, upf%rinner(i), i=1,upf%nqlc )
    #    call scan_end (iunps, "RINNER")
    # end if
    # =======================================================================
    if num_q_coef != 0:
        R_inner = []
        read_until(upf, "<PP_RINNER>")
        for i in range(2 * upf_dict['header']['l_max'] + 1):
            s = upf.readline().split()
            R_inner.append(float(s[1]))
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

    nb = upf_dict['header']['number_of_proj']
    l_max = upf_dict['header']['l_max']

    for i in range(nb):
        li = upf_dict['beta_projectors'][i]['angular_momentum']
        for j in range(i, nb):
            lj = upf_dict['beta_projectors'][j]['angular_momentum']

            s = upf.readline().split()
            if int(s[2]) != lj:
                print("inconsistent angular momentum")
                sys.exit(-1)

            if int(s[0]) != i + 1 or int(s[1]) != j + 1:
                print("inconsistent ij indices")
                sys.exit(-1)

            s = upf.readline().split()

            qij = read_mesh_data(upf, upf_dict['header']['mesh_size'])


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
            if num_q_coef > 0:
                read_until(upf, "<PP_QFCOEF>")

                q_coefs = read_mesh_data(upf, num_q_coef * (2 * l_max + 1))

                read_until(upf, "</PP_QFCOEF>")

            if num_q_coef > 0:
                # constuct Qij(r) for each l
                for l in range(abs(li-lj), li+lj+1):
                    if (li + lj + l) % 2 == 0:
                        qij_fixed = [qij[k] for k in range(upf_dict['header']['mesh_size'])]

                        for ir in range(upf_dict['header']['mesh_size']):
                            x = upf_dict['radial_grid'][ir]
                            x2 = x * x
                            if x < R_inner[l]:
                                qij_fixed[ir] = q_coefs[0 + l * num_q_coef]
                                for n in range(1, num_q_coef):
                                    qij_fixed[ir] += q_coefs[n + l * num_q_coef] * x2**n;
                                qij_fixed[ir] *= x**(l + 2);

                        qij_dict = {}
                        qij_dict['radial_function'] = qij_fixed
                        qij_dict['i'] = i
                        qij_dict['j'] = j
                        qij_dict['angular_momentum'] = l
                        upf_dict['augmentation'].append(qij_dict)


    upf.close()


#
# QE subroutine read_pseudo_pswfc
#
def parse_pswfc(upf_dict, upf_str):

    #print "parsing wfc"

    upf_dict['atomic_wave_functions'] = []

    upf = io.StringIO(upf_str)
    read_until(upf, "<PP_PSWFC>")

    # =======================================================================
    # ALLOCATE( upf%chi( upf%mesh, MAX( upf%nwfc, 1 ) ) )
    # upf%chi = 0.0_DP
    # do nb = 1, upf%nwfc
    #    read (iunps, *, err=100, end=100) dummy  !Wavefunction labels
    #    read (iunps, *, err=100, end=100) ( upf%chi(ir,nb), ir=1,upf%mesh )
    # enddo
    # =======================================================================
    for i in range(upf_dict['header']['number_of_wfc']):
        wf = {}
        s = upf.readline().split()
        wf['label'] = s[0]
        wf['angular_momentum'] = int(s[1])
        wf['occupation'] = float(s[2])

        wf['radial_function'] = read_mesh_data(upf, upf_dict['header']['mesh_size'])
        upf_dict['atomic_wave_functions'].append(wf)
    upf.close()

#
# QE subroutine read_pseudo_rhoatom
#
def parse_rhoatom(upf_dict, upf_str):

    #print "parsing rhoatm"
    upf = io.StringIO(upf_str)
    read_until(upf, "<PP_RHOATOM>")

    # ================================================================
    # ALLOCATE( upf%rho_at( upf%mesh ) )
    # read (iunps,*,err=100,end=100) ( upf%rho_at(ir), ir=1,upf%mesh )
    # ================================================================
    upf_dict['total_charge_density'] = read_mesh_data(upf, upf_dict['header']['mesh_size'])

    upf.close()


def parse_upf1_from_string(upf1_str):
    upf_dict = {}

    parse_header(upf_dict, upf1_str)
    parse_mesh(upf_dict, upf1_str)
    if upf_dict['header']['core_correction'] == 1: parse_nlcc(upf_dict, upf1_str)
    parse_local(upf_dict, upf1_str)
    parse_non_local(upf_dict, upf1_str)
    parse_pswfc(upf_dict, upf1_str)
    parse_rhoatom(upf_dict, upf1_str)

    pp_dict = {}
    pp_dict['pseudo_potential'] = upf_dict

    return pp_dict

def main():

    with open(sys.argv[1], 'r') as fh:
        upf = fh.read()
    pp_dict = parse_upf1_from_string(upf)

    fout = open(sys.argv[1] + ".json", "w")

    # Match comma, space, newline and an arbitrary number of spaces ',\s\n\s*' with the
    # following conditions: a digit before (?<=[0-9]) and a minus or a digit after (?=[-|0-9]).
    # Replace found sequence with comma and space.
    fout.write(re.sub(r"(?<=[0-9]),\s\n\s*(?=[-|0-9])", r", ", json.dumps(pp_dict, indent=2)))
    fout.close()

if __name__ == "__main__":
    main()
