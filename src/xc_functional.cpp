// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that 
// the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the 
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions 
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file xc_functional.cpp
 *   
 *  \brief Contains initialization of Libxc functional names.
 */

#include <string>
#include <map>
#include "xc_functional.h"

namespace sirius {

std::map<std::string, int> init_libxc_functionals()
{
    std::map<std::string, int> f;

    f["XC_LDA_X"] = XC_LDA_X; /* Exchange */
    f["XC_LDA_C_WIGNER"] = XC_LDA_C_WIGNER; /* Wigner parametrization */
    f["XC_LDA_C_RPA"] = XC_LDA_C_RPA; /* Random Phase Approximation */
    f["XC_LDA_C_HL"] = XC_LDA_C_HL; /* Hedin & Lundqvist */
    f["XC_LDA_C_GL"] = XC_LDA_C_GL; /* Gunnarson & Lundqvist */
    f["XC_LDA_C_XALPHA"] = XC_LDA_C_XALPHA; /* Slater Xalpha */
    f["XC_LDA_C_VWN"] = XC_LDA_C_VWN; /* Vosko, Wilk, & Nusair (5) */
    f["XC_LDA_C_VWN_RPA"] = XC_LDA_C_VWN_RPA; /* Vosko, Wilk, & Nusair (RPA) */
    f["XC_LDA_C_PZ"] = XC_LDA_C_PZ; /* Perdew & Zunger */
    f["XC_LDA_C_PZ_MOD"] = XC_LDA_C_PZ_MOD; /* Perdew & Zunger (Modified) */
    f["XC_LDA_C_OB_PZ"] = XC_LDA_C_OB_PZ; /* Ortiz & Ballone (PZ) */
    f["XC_LDA_C_PW"] = XC_LDA_C_PW; /* Perdew & Wang */
    f["XC_LDA_C_PW_MOD"] = XC_LDA_C_PW_MOD; /* Perdew & Wang (Modified) */
    f["XC_LDA_C_OB_PW"] = XC_LDA_C_OB_PW; /* Ortiz & Ballone (PW) */
    f["XC_LDA_C_2D_AMGB"] = XC_LDA_C_2D_AMGB; /* Attaccalite et al */
    f["XC_LDA_C_2D_PRM"] = XC_LDA_C_2D_PRM; /* Pittalis, Rasanen & Marques correlation in 2D */
    f["XC_LDA_C_vBH"] = XC_LDA_C_vBH; /* von Barth & Hedin */
    f["XC_LDA_C_1D_CSC"] = XC_LDA_C_1D_CSC; /* Casula, Sorella, and Senatore 1D correlation */
    f["XC_LDA_X_2D"] = XC_LDA_X_2D; /* Exchange in 2D */
    f["XC_LDA_XC_TETER93"] = XC_LDA_XC_TETER93; /* Teter 93 parametrization */
    f["XC_LDA_X_1D"] = XC_LDA_X_1D; /* Exchange in 1D */
    f["XC_LDA_C_ML1"] = XC_LDA_C_ML1; /* Modified LSD (version 1) of Proynov and Salahub */
    f["XC_LDA_C_ML2"] = XC_LDA_C_ML2; /* Modified LSD (version 2) of Proynov and Salahub */
    f["XC_LDA_C_GOMBAS"] = XC_LDA_C_GOMBAS; /* Gombas parametrization */
    f["XC_LDA_C_PW_RPA"] = XC_LDA_C_PW_RPA; /* Perdew & Wang fit of the RPA */
    f["XC_LDA_C_1D_LOOS"] = XC_LDA_C_1D_LOOS; /* P-F Loos correlation LDA */
    f["XC_LDA_C_RC04"] = XC_LDA_C_RC04; /* Ragot-Cortona */
    f["XC_LDA_C_VWN_1"] = XC_LDA_C_VWN_1; /* Vosko, Wilk, & Nusair (1) */
    f["XC_LDA_C_VWN_2"] = XC_LDA_C_VWN_2; /* Vosko, Wilk, & Nusair (2) */
    f["XC_LDA_C_VWN_3"] = XC_LDA_C_VWN_3; /* Vosko, Wilk, & Nusair (3) */
    f["XC_LDA_C_VWN_4"] = XC_LDA_C_VWN_4; /* Vosko, Wilk, & Nusair (4) */
    f["XC_LDA_XC_ZLP"] = XC_LDA_XC_ZLP; /* Zhao, Levy & Parr, Eq. (20) */
    f["XC_LDA_K_TF"] = XC_LDA_K_TF; /* Thomas-Fermi kinetic energy functional */
    f["XC_LDA_K_LP"] = XC_LDA_K_LP; /* Lee and Parr Gaussian ansatz */
    f["XC_LDA_XC_KSDT"] = XC_LDA_XC_KSDT; /* Karasiev et al. parametrization */
    f["XC_GGA_X_GAM"] = XC_GGA_X_GAM; /* GAM functional from Minnesota */
    f["XC_GGA_C_GAM"] = XC_GGA_C_GAM; /* GAM functional from Minnesota */
    f["XC_GGA_X_HCTH_A"] = XC_GGA_X_HCTH_A; /* HCTH-A */
    f["XC_GGA_X_EV93"] = XC_GGA_X_EV93; /* Engel and Vosko */
    f["XC_GGA_X_BGCP"] = XC_GGA_X_BGCP; /* Burke, Cancio, Gould, and Pittalis */
    f["XC_GGA_C_BGCP"] = XC_GGA_C_BGCP; /* Burke, Cancio, Gould, and Pittalis */
    f["XC_GGA_X_LAMBDA_OC2_N"] = XC_GGA_X_LAMBDA_OC2_N; /* lambda_OC2(N) version of PBE */
    f["XC_GGA_X_B86_R"] = XC_GGA_X_B86_R; /* Revised Becke 86 Xalpha,beta,gamma (with mod. grad. correction) */
    f["XC_GGA_X_LAMBDA_CH_N"] = XC_GGA_X_LAMBDA_CH_N; /* lambda_CH(N) version of PBE */
    f["XC_GGA_X_LAMBDA_LO_N"] = XC_GGA_X_LAMBDA_LO_N; /* lambda_LO(N) version of PBE */
    f["XC_GGA_X_HJS_B88_V2"] = XC_GGA_X_HJS_B88_V2; /* HJS screened exchange corrected B88 version */
    f["XC_GGA_C_Q2D"] = XC_GGA_C_Q2D; /* Chiodo et al */
    f["XC_GGA_X_Q2D"] = XC_GGA_X_Q2D; /* Chiodo et al */
    f["XC_GGA_X_PBE_MOL"] = XC_GGA_X_PBE_MOL; /* Del Campo, Gazquez, Trickey and Vela (PBE-like) */
    f["XC_GGA_K_TFVW"] = XC_GGA_K_TFVW; /* Thomas-Fermi plus von Weiszaecker correction */
    f["XC_GGA_K_REVAPBEINT"] = XC_GGA_K_REVAPBEINT; /* interpolated version of REVAPBE */
    f["XC_GGA_K_APBEINT"] = XC_GGA_K_APBEINT; /* interpolated version of APBE */
    f["XC_GGA_K_REVAPBE"] = XC_GGA_K_REVAPBE; /* revised APBE */
    f["XC_GGA_X_AK13"] = XC_GGA_X_AK13; /* Armiento & Kuemmel 2013 */
    f["XC_GGA_K_MEYER"] = XC_GGA_K_MEYER; /* Meyer,  Wang, and Young */
    f["XC_GGA_X_LV_RPW86"] = XC_GGA_X_LV_RPW86; /* Berland and Hyldgaard */
    f["XC_GGA_X_PBE_TCA"] = XC_GGA_X_PBE_TCA; /* PBE revised by Tognetti et al */
    f["XC_GGA_X_PBEINT"] = XC_GGA_X_PBEINT; /* PBE for hybrid interfaces */
    f["XC_GGA_C_ZPBEINT"] = XC_GGA_C_ZPBEINT; /* spin-dependent gradient correction to PBEint */
    f["XC_GGA_C_PBEINT"] = XC_GGA_C_PBEINT; /* PBE for hybrid interfaces */
    f["XC_GGA_C_ZPBESOL"] = XC_GGA_C_ZPBESOL; /* spin-dependent gradient correction to PBEsol */
    f["XC_GGA_XC_OPBE_D"] = XC_GGA_XC_OPBE_D; /* oPBE_D functional of Goerigk and Grimme */
    f["XC_GGA_XC_OPWLYP_D"] = XC_GGA_XC_OPWLYP_D; /* oPWLYP-D functional of Goerigk and Grimme */
    f["XC_GGA_XC_OBLYP_D"] = XC_GGA_XC_OBLYP_D; /* oBLYP-D functional of Goerigk and Grimme */
    f["XC_GGA_X_VMT84_GE"] = XC_GGA_X_VMT84_GE; /* VMT{8,4} with constraint satisfaction with mu = mu_GE */
    f["XC_GGA_X_VMT84_PBE"] = XC_GGA_X_VMT84_PBE; /* VMT{8,4} with constraint satisfaction with mu = mu_PBE */
    f["XC_GGA_X_VMT_GE"] = XC_GGA_X_VMT_GE; /* Vela, Medel, and Trickey with mu = mu_GE */
    f["XC_GGA_X_VMT_PBE"] = XC_GGA_X_VMT_PBE; /* Vela, Medel, and Trickey with mu = mu_PBE */
    f["XC_GGA_C_N12_SX"] = XC_GGA_C_N12_SX; /* N12-SX functional from Minnesota */
    f["XC_GGA_C_N12"] = XC_GGA_C_N12; /* N12 functional from Minnesota */
    f["XC_GGA_X_N12"] = XC_GGA_X_N12; /* N12 functional from Minnesota */
    f["XC_GGA_C_REGTPSS"] = XC_GGA_C_REGTPSS; /* Regularized TPSS correlation (ex-VPBE) */
    f["XC_GGA_C_OP_XALPHA"] = XC_GGA_C_OP_XALPHA; /* one-parameter progressive functional (XALPHA version) */
    f["XC_GGA_C_OP_G96"] = XC_GGA_C_OP_G96; /* one-parameter progressive functional (G96 version) */
    f["XC_GGA_C_OP_PBE"] = XC_GGA_C_OP_PBE; /* one-parameter progressive functional (PBE version) */
    f["XC_GGA_C_OP_B88"] = XC_GGA_C_OP_B88; /* one-parameter progressive functional (B88 version) */
    f["XC_GGA_C_FT97"] = XC_GGA_C_FT97; /* Filatov & Thiel correlation */
    f["XC_GGA_C_SPBE"] = XC_GGA_C_SPBE; /* PBE correlation to be used with the SSB exchange */
    f["XC_GGA_X_SSB_SW"] = XC_GGA_X_SSB_SW; /* Swarta, Sola and Bickelhaupt correction to PBE */
    f["XC_GGA_X_SSB"] = XC_GGA_X_SSB; /* Swarta, Sola and Bickelhaupt */
    f["XC_GGA_X_SSB_D"] = XC_GGA_X_SSB_D; /* Swarta, Sola and Bickelhaupt dispersion */
    f["XC_GGA_XC_HCTH_407P"] = XC_GGA_XC_HCTH_407P; /* HCTH/407+ */
    f["XC_GGA_XC_HCTH_P76"] = XC_GGA_XC_HCTH_P76; /* HCTH p=7/6 */
    f["XC_GGA_XC_HCTH_P14"] = XC_GGA_XC_HCTH_P14; /* HCTH p=1/4 */
    f["XC_GGA_XC_B97_GGA1"] = XC_GGA_XC_B97_GGA1; /* Becke 97 GGA-1 */
    f["XC_GGA_C_HCTH_A"] = XC_GGA_C_HCTH_A; /* HCTH-A */
    f["XC_GGA_X_BPCCAC"] = XC_GGA_X_BPCCAC; /* BPCCAC (GRAC for the energy) */
    f["XC_GGA_C_REVTCA"] = XC_GGA_C_REVTCA; /* Tognetti, Cortona, Adamo (revised) */
    f["XC_GGA_C_TCA"] = XC_GGA_C_TCA; /* Tognetti, Cortona, Adamo */
    f["XC_GGA_X_PBE"] = XC_GGA_X_PBE; /* Perdew, Burke & Ernzerhof exchange */
    f["XC_GGA_X_PBE_R"] = XC_GGA_X_PBE_R; /* Perdew, Burke & Ernzerhof exchange (revised) */
    f["XC_GGA_X_B86"] = XC_GGA_X_B86; /* Becke 86 Xalpha,beta,gamma */
    f["XC_GGA_X_HERMAN"] = XC_GGA_X_HERMAN; /* Herman et al original GGA */
    f["XC_GGA_X_B86_MGC"] = XC_GGA_X_B86_MGC; /* Becke 86 Xalpha,beta,gamma (with mod. grad. correction) */
    f["XC_GGA_X_B88"] = XC_GGA_X_B88; /* Becke 88 */
    f["XC_GGA_X_G96"] = XC_GGA_X_G96; /* Gill 96 */
    f["XC_GGA_X_PW86"] = XC_GGA_X_PW86; /* Perdew & Wang 86 */
    f["XC_GGA_X_PW91"] = XC_GGA_X_PW91; /* Perdew & Wang 91 */
    f["XC_GGA_X_OPTX"] = XC_GGA_X_OPTX; /* Handy & Cohen OPTX 01 */
    f["XC_GGA_X_DK87_R1"] = XC_GGA_X_DK87_R1; /* dePristo & Kress 87 (version R1) */
    f["XC_GGA_X_DK87_R2"] = XC_GGA_X_DK87_R2; /* dePristo & Kress 87 (version R2) */
    f["XC_GGA_X_LG93"] = XC_GGA_X_LG93; /* Lacks & Gordon 93 */
    f["XC_GGA_X_FT97_A"] = XC_GGA_X_FT97_A; /* Filatov & Thiel 97 (version A) */
    f["XC_GGA_X_FT97_B"] = XC_GGA_X_FT97_B; /* Filatov & Thiel 97 (version B) */
    f["XC_GGA_X_PBE_SOL"] = XC_GGA_X_PBE_SOL; /* Perdew, Burke & Ernzerhof exchange (solids) */
    f["XC_GGA_X_RPBE"] = XC_GGA_X_RPBE; /* Hammer, Hansen & Norskov (PBE-like) */
    f["XC_GGA_X_WC"] = XC_GGA_X_WC; /* Wu & Cohen */
    f["XC_GGA_X_MPW91"] = XC_GGA_X_MPW91; /* Modified form of PW91 by Adamo & Barone */
    f["XC_GGA_X_AM05"] = XC_GGA_X_AM05; /* Armiento & Mattsson 05 exchange */
    f["XC_GGA_X_PBEA"] = XC_GGA_X_PBEA; /* Madsen (PBE-like) */
    f["XC_GGA_X_MPBE"] = XC_GGA_X_MPBE; /* Adamo & Barone modification to PBE */
    f["XC_GGA_X_XPBE"] = XC_GGA_X_XPBE; /* xPBE reparametrization by Xu & Goddard */
    f["XC_GGA_X_2D_B86_MGC"] = XC_GGA_X_2D_B86_MGC; /* Becke 86 MGC for 2D systems */
    f["XC_GGA_X_BAYESIAN"] = XC_GGA_X_BAYESIAN; /* Bayesian best fit for the enhancement factor */
    f["XC_GGA_X_PBE_JSJR"] = XC_GGA_X_PBE_JSJR; /* JSJR reparametrization by Pedroza, Silva & Capelle */
    f["XC_GGA_X_2D_B88"] = XC_GGA_X_2D_B88; /* Becke 88 in 2D */
    f["XC_GGA_X_2D_B86"] = XC_GGA_X_2D_B86; /* Becke 86 Xalpha,beta,gamma */
    f["XC_GGA_X_2D_PBE"] = XC_GGA_X_2D_PBE; /* Perdew, Burke & Ernzerhof exchange in 2D */
    f["XC_GGA_C_PBE"] = XC_GGA_C_PBE; /* Perdew, Burke & Ernzerhof correlation */
    f["XC_GGA_C_LYP"] = XC_GGA_C_LYP; /* Lee, Yang & Parr */
    f["XC_GGA_C_P86"] = XC_GGA_C_P86; /* Perdew 86 */
    f["XC_GGA_C_PBE_SOL"] = XC_GGA_C_PBE_SOL; /* Perdew, Burke & Ernzerhof correlation SOL */
    f["XC_GGA_C_PW91"] = XC_GGA_C_PW91; /* Perdew & Wang 91 */
    f["XC_GGA_C_AM05"] = XC_GGA_C_AM05; /* Armiento & Mattsson 05 correlation */
    f["XC_GGA_C_XPBE"] = XC_GGA_C_XPBE; /* xPBE reparametrization by Xu & Goddard */
    f["XC_GGA_C_LM"] = XC_GGA_C_LM; /* Langreth and Mehl correlation */
    f["XC_GGA_C_PBE_JRGX"] = XC_GGA_C_PBE_JRGX; /* JRGX reparametrization by Pedroza, Silva & Capelle */
    f["XC_GGA_X_OPTB88_VDW"] = XC_GGA_X_OPTB88_VDW; /* Becke 88 reoptimized to be used with vdW functional of Dion et al */
    f["XC_GGA_X_PBEK1_VDW"] = XC_GGA_X_PBEK1_VDW; /* PBE reparametrization for vdW */
    f["XC_GGA_X_OPTPBE_VDW"] = XC_GGA_X_OPTPBE_VDW; /* PBE reparametrization for vdW */
    f["XC_GGA_X_RGE2"] = XC_GGA_X_RGE2; /* Regularized PBE */
    f["XC_GGA_C_RGE2"] = XC_GGA_C_RGE2; /* Regularized PBE */
    f["XC_GGA_X_RPW86"] = XC_GGA_X_RPW86; /* refitted Perdew & Wang 86 */
    f["XC_GGA_X_KT1"] = XC_GGA_X_KT1; /* Keal and Tozer version 1 */
    f["XC_GGA_XC_KT2"] = XC_GGA_XC_KT2; /* Keal and Tozer version 2 */
    f["XC_GGA_C_WL"] = XC_GGA_C_WL; /* Wilson & Levy */
    f["XC_GGA_C_WI"] = XC_GGA_C_WI; /* Wilson & Ivanov */
    f["XC_GGA_X_MB88"] = XC_GGA_X_MB88; /* Modified Becke 88 for proton transfer */
    f["XC_GGA_X_SOGGA"] = XC_GGA_X_SOGGA; /* Second-order generalized gradient approximation */
    f["XC_GGA_X_SOGGA11"] = XC_GGA_X_SOGGA11; /* Second-order generalized gradient approximation 2011 */
    f["XC_GGA_C_SOGGA11"] = XC_GGA_C_SOGGA11; /* Second-order generalized gradient approximation 2011 */
    f["XC_GGA_C_WI0"] = XC_GGA_C_WI0; /* Wilson & Ivanov initial version */
    f["XC_GGA_XC_TH1"] = XC_GGA_XC_TH1; /* Tozer and Handy v. 1 */
    f["XC_GGA_XC_TH2"] = XC_GGA_XC_TH2; /* Tozer and Handy v. 2 */
    f["XC_GGA_XC_TH3"] = XC_GGA_XC_TH3; /* Tozer and Handy v. 3 */
    f["XC_GGA_XC_TH4"] = XC_GGA_XC_TH4; /* Tozer and Handy v. 4 */
    f["XC_GGA_X_C09X"] = XC_GGA_X_C09X; /* C09x to be used with the VdW of Rutgers-Chalmers */
    f["XC_GGA_C_SOGGA11_X"] = XC_GGA_C_SOGGA11_X; /* To be used with HYB_GGA_X_SOGGA11_X */
    f["XC_GGA_X_LB"] = XC_GGA_X_LB; /* van Leeuwen & Baerends */
    f["XC_GGA_XC_HCTH_93"] = XC_GGA_XC_HCTH_93; /* HCTH functional fitted to  93 molecules */
    f["XC_GGA_XC_HCTH_120"] = XC_GGA_XC_HCTH_120; /* HCTH functional fitted to 120 molecules */
    f["XC_GGA_XC_HCTH_147"] = XC_GGA_XC_HCTH_147; /* HCTH functional fitted to 147 molecules */
    f["XC_GGA_XC_HCTH_407"] = XC_GGA_XC_HCTH_407; /* HCTH functional fitted to 407 molecules */
    f["XC_GGA_XC_EDF1"] = XC_GGA_XC_EDF1; /* Empirical functionals from Adamson, Gill, and Pople */
    f["XC_GGA_XC_XLYP"] = XC_GGA_XC_XLYP; /* XLYP functional */
    f["XC_GGA_XC_B97_D"] = XC_GGA_XC_B97_D; /* Grimme functional to be used with C6 vdW term */
    f["XC_GGA_XC_PBE1W"] = XC_GGA_XC_PBE1W; /* Functionals fitted for water */
    f["XC_GGA_XC_MPWLYP1W"] = XC_GGA_XC_MPWLYP1W; /* Functionals fitted for water */
    f["XC_GGA_XC_PBELYP1W"] = XC_GGA_XC_PBELYP1W; /* Functionals fitted for water */
    f["XC_GGA_X_LBM"] = XC_GGA_X_LBM; /* van Leeuwen & Baerends modified */
    f["XC_GGA_X_OL2"] = XC_GGA_X_OL2; /* Exchange form based on Ou-Yang and Levy v.2 */
    f["XC_GGA_X_APBE"] = XC_GGA_X_APBE; /* mu fixed from the semiclassical neutral atom */
    f["XC_GGA_K_APBE"] = XC_GGA_K_APBE; /* mu fixed from the semiclassical neutral atom */
    f["XC_GGA_C_APBE"] = XC_GGA_C_APBE; /* mu fixed from the semiclassical neutral atom */
    f["XC_GGA_K_TW1"] = XC_GGA_K_TW1; /* Tran and Wesolowski set 1 (Table II) */
    f["XC_GGA_K_TW2"] = XC_GGA_K_TW2; /* Tran and Wesolowski set 2 (Table II) */
    f["XC_GGA_K_TW3"] = XC_GGA_K_TW3; /* Tran and Wesolowski set 3 (Table II) */
    f["XC_GGA_K_TW4"] = XC_GGA_K_TW4; /* Tran and Wesolowski set 4 (Table II) */
    f["XC_GGA_X_HTBS"] = XC_GGA_X_HTBS; /* Haas, Tran, Blaha, and Schwarz */
    f["XC_GGA_X_AIRY"] = XC_GGA_X_AIRY; /* Constantin et al based on the Airy gas */
    f["XC_GGA_X_LAG"] = XC_GGA_X_LAG; /* Local Airy Gas */
    f["XC_GGA_XC_MOHLYP"] = XC_GGA_XC_MOHLYP; /* Functional for organometallic chemistry */
    f["XC_GGA_XC_MOHLYP2"] = XC_GGA_XC_MOHLYP2; /* Functional for barrier heights */
    f["XC_GGA_XC_TH_FL"] = XC_GGA_XC_TH_FL; /* Tozer and Handy v. FL */
    f["XC_GGA_XC_TH_FC"] = XC_GGA_XC_TH_FC; /* Tozer and Handy v. FC */
    f["XC_GGA_XC_TH_FCFO"] = XC_GGA_XC_TH_FCFO; /* Tozer and Handy v. FCFO */
    f["XC_GGA_XC_TH_FCO"] = XC_GGA_XC_TH_FCO; /* Tozer and Handy v. FCO */
    f["XC_GGA_C_OPTC"] = XC_GGA_C_OPTC; /* Optimized correlation functional of Cohen and Handy */
    f["XC_GGA_C_PBELOC"] = XC_GGA_C_PBELOC; /* Semilocal dynamical correlation */
    f["XC_GGA_XC_VV10"] = XC_GGA_XC_VV10; /* Vydrov and Van Voorhis */
    f["XC_GGA_C_PBEFE"] = XC_GGA_C_PBEFE; /* PBE for formation energies */
    f["XC_GGA_C_OP_PW91"] = XC_GGA_C_OP_PW91; /* one-parameter progressive functional (PW91 version) */
    f["XC_GGA_X_PBEFE"] = XC_GGA_X_PBEFE; /* PBE for formation energies */
    f["XC_GGA_X_CAP"] = XC_GGA_X_CAP; /* Correct Asymptotic Potential */
    f["XC_GGA_K_VW"] = XC_GGA_K_VW; /* von Weiszaecker functional */
    f["XC_GGA_K_GE2"] = XC_GGA_K_GE2; /* Second-order gradient expansion (l = 1/9) */
    f["XC_GGA_K_GOLDEN"] = XC_GGA_K_GOLDEN; /* TF-lambda-vW form by Golden (l = 13/45) */
    f["XC_GGA_K_YT65"] = XC_GGA_K_YT65; /* TF-lambda-vW form by Yonei and Tomishima (l = 1/5) */
    f["XC_GGA_K_BALTIN"] = XC_GGA_K_BALTIN; /* TF-lambda-vW form by Baltin (l = 5/9) */
    f["XC_GGA_K_LIEB"] = XC_GGA_K_LIEB; /* TF-lambda-vW form by Lieb (l = 0.185909191) */
    f["XC_GGA_K_ABSP1"] = XC_GGA_K_ABSP1; /* gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)] */
    f["XC_GGA_K_ABSP2"] = XC_GGA_K_ABSP2; /* gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)] */
    f["XC_GGA_K_GR"] = XC_GGA_K_GR; /* gamma-TFvW form by Gazquez and Robles */
    f["XC_GGA_K_LUDENA"] = XC_GGA_K_LUDENA; /* gamma-TFvW form by Ludena */
    f["XC_GGA_K_GP85"] = XC_GGA_K_GP85; /* gamma-TFvW form by Ghosh and Parr */
    f["XC_GGA_K_PEARSON"] = XC_GGA_K_PEARSON; /* Pearson */
    f["XC_GGA_K_OL1"] = XC_GGA_K_OL1; /* Ou-Yang and Levy v.1 */
    f["XC_GGA_K_OL2"] = XC_GGA_K_OL2; /* Ou-Yang and Levy v.2 */
    f["XC_GGA_K_FR_B88"] = XC_GGA_K_FR_B88; /* Fuentealba & Reyes (B88 version) */
    f["XC_GGA_K_FR_PW86"] = XC_GGA_K_FR_PW86; /* Fuentealba & Reyes (PW86 version) */
    f["XC_GGA_K_DK"] = XC_GGA_K_DK; /* DePristo and Kress */
    f["XC_GGA_K_PERDEW"] = XC_GGA_K_PERDEW; /* Perdew */
    f["XC_GGA_K_VSK"] = XC_GGA_K_VSK; /* Vitos, Skriver, and Kollar */
    f["XC_GGA_K_VJKS"] = XC_GGA_K_VJKS; /* Vitos, Johansson, Kollar, and Skriver */
    f["XC_GGA_K_ERNZERHOF"] = XC_GGA_K_ERNZERHOF; /* Ernzerhof */
    f["XC_GGA_K_LC94"] = XC_GGA_K_LC94; /* Lembarki & Chermette */
    f["XC_GGA_K_LLP"] = XC_GGA_K_LLP; /* Lee, Lee & Parr */
    f["XC_GGA_K_THAKKAR"] = XC_GGA_K_THAKKAR; /* Thakkar 1992 */
    f["XC_GGA_X_WPBEH"] = XC_GGA_X_WPBEH; /* short-range version of the PBE */
    f["XC_GGA_X_HJS_PBE"] = XC_GGA_X_HJS_PBE; /* HJS screened exchange PBE version */
    f["XC_GGA_X_HJS_PBE_SOL"] = XC_GGA_X_HJS_PBE_SOL; /* HJS screened exchange PBE_SOL version */
    f["XC_GGA_X_HJS_B88"] = XC_GGA_X_HJS_B88; /* HJS screened exchange B88 version */
    f["XC_GGA_X_HJS_B97X"] = XC_GGA_X_HJS_B97X; /* HJS screened exchange B97x version */
    f["XC_GGA_X_ITYH"] = XC_GGA_X_ITYH; /* short-range recipe for exchange GGA functionals */
    f["XC_GGA_X_SFAT"] = XC_GGA_X_SFAT; /* short-range recipe for exchange GGA functionals */
    f["XC_HYB_GGA_X_N12_SX"] = XC_HYB_GGA_X_N12_SX; /* N12-SX functional from Minnesota */
    f["XC_HYB_GGA_XC_B97_1p"] = XC_HYB_GGA_XC_B97_1p; /* version of B97 by Cohen and Handy */
    f["XC_HYB_GGA_XC_B3PW91"] = XC_HYB_GGA_XC_B3PW91; /* The original (ACM) hybrid of Becke */
    f["XC_HYB_GGA_XC_B3LYP"] = XC_HYB_GGA_XC_B3LYP; /* The (in)famous B3LYP */
    f["XC_HYB_GGA_XC_B3P86"] = XC_HYB_GGA_XC_B3P86; /* Perdew 86 hybrid similar to B3PW91 */
    f["XC_HYB_GGA_XC_O3LYP"] = XC_HYB_GGA_XC_O3LYP; /* hybrid using the optx functional */
    f["XC_HYB_GGA_XC_mPW1K"] = XC_HYB_GGA_XC_mPW1K; /* mixture of mPW91 and PW91 optimized for kinetics */
    f["XC_HYB_GGA_XC_PBEH"] = XC_HYB_GGA_XC_PBEH; /* aka PBE0 or PBE1PBE */
    f["XC_HYB_GGA_XC_B97"] = XC_HYB_GGA_XC_B97; /* Becke 97 */
    f["XC_HYB_GGA_XC_B97_1"] = XC_HYB_GGA_XC_B97_1; /* Becke 97-1 */
    f["XC_HYB_GGA_XC_B97_2"] = XC_HYB_GGA_XC_B97_2; /* Becke 97-2 */
    f["XC_HYB_GGA_XC_X3LYP"] = XC_HYB_GGA_XC_X3LYP; /* hybrid by Xu and Goddard */
    f["XC_HYB_GGA_XC_B1WC"] = XC_HYB_GGA_XC_B1WC; /* Becke 1-parameter mixture of WC and PBE */
    f["XC_HYB_GGA_XC_B97_K"] = XC_HYB_GGA_XC_B97_K; /* Boese-Martin for Kinetics */
    f["XC_HYB_GGA_XC_B97_3"] = XC_HYB_GGA_XC_B97_3; /* Becke 97-3 */
    f["XC_HYB_GGA_XC_MPW3PW"] = XC_HYB_GGA_XC_MPW3PW; /* mixture with the mPW functional */
    f["XC_HYB_GGA_XC_B1LYP"] = XC_HYB_GGA_XC_B1LYP; /* Becke 1-parameter mixture of B88 and LYP */
    f["XC_HYB_GGA_XC_B1PW91"] = XC_HYB_GGA_XC_B1PW91; /* Becke 1-parameter mixture of B88 and PW91 */
    f["XC_HYB_GGA_XC_mPW1PW"] = XC_HYB_GGA_XC_mPW1PW; /* Becke 1-parameter mixture of mPW91 and PW91 */
    f["XC_HYB_GGA_XC_MPW3LYP"] = XC_HYB_GGA_XC_MPW3LYP; /* mixture of mPW and LYP */
    f["XC_HYB_GGA_XC_SB98_1a"] = XC_HYB_GGA_XC_SB98_1a; /* Schmider-Becke 98 parameterization 1a */
    f["XC_HYB_GGA_XC_SB98_1b"] = XC_HYB_GGA_XC_SB98_1b; /* Schmider-Becke 98 parameterization 1b */
    f["XC_HYB_GGA_XC_SB98_1c"] = XC_HYB_GGA_XC_SB98_1c; /* Schmider-Becke 98 parameterization 1c */
    f["XC_HYB_GGA_XC_SB98_2a"] = XC_HYB_GGA_XC_SB98_2a; /* Schmider-Becke 98 parameterization 2a */
    f["XC_HYB_GGA_XC_SB98_2b"] = XC_HYB_GGA_XC_SB98_2b; /* Schmider-Becke 98 parameterization 2b */
    f["XC_HYB_GGA_XC_SB98_2c"] = XC_HYB_GGA_XC_SB98_2c; /* Schmider-Becke 98 parameterization 2c */
    f["XC_HYB_GGA_X_SOGGA11_X"] = XC_HYB_GGA_X_SOGGA11_X; /* Hybrid based on SOGGA11 form */
    f["XC_HYB_GGA_XC_HSE03"] = XC_HYB_GGA_XC_HSE03; /* the 2003 version of the screened hybrid HSE */
    f["XC_HYB_GGA_XC_HSE06"] = XC_HYB_GGA_XC_HSE06; /* the 2006 version of the screened hybrid HSE */
    f["XC_HYB_GGA_XC_HJS_PBE"] = XC_HYB_GGA_XC_HJS_PBE; /* HJS hybrid screened exchange PBE version */
    f["XC_HYB_GGA_XC_HJS_PBE_SOL"] = XC_HYB_GGA_XC_HJS_PBE_SOL; /* HJS hybrid screened exchange PBE_SOL version */
    f["XC_HYB_GGA_XC_HJS_B88"] = XC_HYB_GGA_XC_HJS_B88; /* HJS hybrid screened exchange B88 version */
    f["XC_HYB_GGA_XC_HJS_B97X"] = XC_HYB_GGA_XC_HJS_B97X; /* HJS hybrid screened exchange B97x version */
    f["XC_HYB_GGA_XC_CAM_B3LYP"] = XC_HYB_GGA_XC_CAM_B3LYP; /* CAM version of B3LYP */
    f["XC_HYB_GGA_XC_TUNED_CAM_B3LYP"] = XC_HYB_GGA_XC_TUNED_CAM_B3LYP; /* CAM version of B3LYP tuned for excitations */
    f["XC_HYB_GGA_XC_BHANDH"] = XC_HYB_GGA_XC_BHANDH; /* Becke half-and-half */
    f["XC_HYB_GGA_XC_BHANDHLYP"] = XC_HYB_GGA_XC_BHANDHLYP; /* Becke half-and-half with B88 exchange */
    f["XC_HYB_GGA_XC_MB3LYP_RC04"] = XC_HYB_GGA_XC_MB3LYP_RC04; /* B3LYP with RC04 LDA */
    f["XC_HYB_GGA_XC_MPWLYP1M"] = XC_HYB_GGA_XC_MPWLYP1M; /* MPW with 1 par. for metals/LYP */
    f["XC_HYB_GGA_XC_REVB3LYP"] = XC_HYB_GGA_XC_REVB3LYP; /* Revised B3LYP */
    f["XC_HYB_GGA_XC_CAMY_BLYP"] = XC_HYB_GGA_XC_CAMY_BLYP; /* BLYP with yukawa screening */
    f["XC_HYB_GGA_XC_PBE0_13"] = XC_HYB_GGA_XC_PBE0_13; /* PBE0-1/3 */
    f["XC_HYB_GGA_XC_B3LYPs"] = XC_HYB_GGA_XC_B3LYPs; /* B3LYP * functional */
    f["XC_HYB_GGA_XC_WB97"] = XC_HYB_GGA_XC_WB97; /* Chai and Head-Gordon */
    f["XC_HYB_GGA_XC_WB97X"] = XC_HYB_GGA_XC_WB97X; /* Chai and Head-Gordon */
    f["XC_HYB_GGA_XC_LRC_WPBEH"] = XC_HYB_GGA_XC_LRC_WPBEH; /* Long-range corrected functional by Rorhdanz et al */
    f["XC_HYB_GGA_XC_WB97X_V"] = XC_HYB_GGA_XC_WB97X_V; /* Mardirossian and Head-Gordon */
    f["XC_HYB_GGA_XC_LCY_PBE"] = XC_HYB_GGA_XC_LCY_PBE; /* PBE with yukawa screening */
    f["XC_HYB_GGA_XC_LCY_BLYP"] = XC_HYB_GGA_XC_LCY_BLYP; /* BLYP with yukawa screening */
    f["XC_HYB_GGA_XC_LC_VV10"] = XC_HYB_GGA_XC_LC_VV10; /* Vydrov and Van Voorhis */
    f["XC_HYB_GGA_XC_CAMY_B3LYP"] = XC_HYB_GGA_XC_CAMY_B3LYP; /* B3LYP with Yukawa screening */
    f["XC_HYB_GGA_XC_WB97X_D"] = XC_HYB_GGA_XC_WB97X_D; /* Chai and Head-Gordon */
    f["XC_HYB_GGA_XC_HPBEINT"] = XC_HYB_GGA_XC_HPBEINT; /* hPBEint */
    f["XC_HYB_GGA_XC_LRC_WPBE"] = XC_HYB_GGA_XC_LRC_WPBE; /* Long-range corrected functional by Rorhdanz et al */
    f["XC_HYB_GGA_XC_B3LYP5"] = XC_HYB_GGA_XC_B3LYP5; /* B3LYP with VWN functional 5 instead of RPA */
    f["XC_HYB_GGA_XC_EDF2"] = XC_HYB_GGA_XC_EDF2; /* Empirical functional from Lin, George and Gill */
    f["XC_HYB_GGA_XC_CAP0"] = XC_HYB_GGA_XC_CAP0; /* Correct Asymptotic Potential hybrid */
    f["XC_MGGA_C_DLDF"] = XC_MGGA_C_DLDF; /* Dispersionless Density Functional */
    f["XC_MGGA_XC_ZLP"] = XC_MGGA_XC_ZLP; /* Zhao, Levy & Parr, Eq. (21) */
    f["XC_MGGA_XC_OTPSS_D"] = XC_MGGA_XC_OTPSS_D; /* oTPSS_D functional of Goerigk and Grimme */
    f["XC_MGGA_C_CS"] = XC_MGGA_C_CS; /* Colle and Salvetti */
    f["XC_MGGA_C_MN12_SX"] = XC_MGGA_C_MN12_SX; /* Worker for MN12-SX functional */
    f["XC_MGGA_C_MN12_L"] = XC_MGGA_C_MN12_L; /* MN12-L functional from Minnesota */
    f["XC_MGGA_C_M11_L"] = XC_MGGA_C_M11_L; /* M11-L functional from Minnesota */
    f["XC_MGGA_C_M11"] = XC_MGGA_C_M11; /* Worker for M11 functional */
    f["XC_MGGA_C_M08_SO"] = XC_MGGA_C_M08_SO; /* Worker for M08-SO functional */
    f["XC_MGGA_C_M08_HX"] = XC_MGGA_C_M08_HX; /* Worker for M08-HX functional */
    f["XC_MGGA_X_LTA"] = XC_MGGA_X_LTA; /* Local tau approximation of Ernzerhof & Scuseria */
    f["XC_MGGA_X_TPSS"] = XC_MGGA_X_TPSS; /* Perdew, Tao, Staroverov & Scuseria exchange */
    f["XC_MGGA_X_M06_L"] = XC_MGGA_X_M06_L; /* M06-Local functional of Minnesota */
    f["XC_MGGA_X_GVT4"] = XC_MGGA_X_GVT4; /* GVT4 from Van Voorhis and Scuseria */
    f["XC_MGGA_X_TAU_HCTH"] = XC_MGGA_X_TAU_HCTH; /* tau-HCTH from Boese and Handy */
    f["XC_MGGA_X_BR89"] = XC_MGGA_X_BR89; /* Becke-Roussel 89 */
    f["XC_MGGA_X_BJ06"] = XC_MGGA_X_BJ06; /* Becke & Johnson correction to Becke-Roussel 89 */
    f["XC_MGGA_X_TB09"] = XC_MGGA_X_TB09; /* Tran & Blaha correction to Becke & Johnson */
    f["XC_MGGA_X_RPP09"] = XC_MGGA_X_RPP09; /* Rasanen, Pittalis, and Proetto correction to Becke & Johnson */
    f["XC_MGGA_X_2D_PRHG07"] = XC_MGGA_X_2D_PRHG07; /* Pittalis, Rasanen, Helbig, Gross Exchange Functional */
    f["XC_MGGA_X_2D_PRHG07_PRP10"] = XC_MGGA_X_2D_PRHG07_PRP10; /* PRGH07 with PRP10 correction */
    f["XC_MGGA_X_REVTPSS"] = XC_MGGA_X_REVTPSS; /* revised Perdew, Tao, Staroverov & Scuseria exchange */
    f["XC_MGGA_X_PKZB"] = XC_MGGA_X_PKZB; /* Perdew, Kurth, Zupan, and Blaha */
    f["XC_MGGA_X_M05"] = XC_MGGA_X_M05; /* Worker for M05 functional */
    f["XC_MGGA_X_M05_2X"] = XC_MGGA_X_M05_2X; /* Worker for M05-2X functional */
    f["XC_MGGA_X_M06_HF"] = XC_MGGA_X_M06_HF; /* Worker for M06-HF functional */
    f["XC_MGGA_X_M06"] = XC_MGGA_X_M06; /* Worker for M06 functional */
    f["XC_MGGA_X_M06_2X"] = XC_MGGA_X_M06_2X; /* Worker for M06-2X functional */
    f["XC_MGGA_X_M08_HX"] = XC_MGGA_X_M08_HX; /* Worker for M08-HX functional */
    f["XC_MGGA_X_M08_SO"] = XC_MGGA_X_M08_SO; /* Worker for M08-SO functional */
    f["XC_MGGA_X_MS0"] = XC_MGGA_X_MS0; /* MS exchange of Sun, Xiao, and Ruzsinszky */
    f["XC_MGGA_X_MS1"] = XC_MGGA_X_MS1; /* MS1 exchange of Sun, et al */
    f["XC_MGGA_X_MS2"] = XC_MGGA_X_MS2; /* MS2 exchange of Sun, et al */
    f["XC_MGGA_X_M11"] = XC_MGGA_X_M11; /* Worker for M11 functional */
    f["XC_MGGA_X_M11_L"] = XC_MGGA_X_M11_L; /* M11-L functional from Minnesota */
    f["XC_MGGA_X_MN12_L"] = XC_MGGA_X_MN12_L; /* MN12-L functional from Minnesota */
    f["XC_MGGA_C_CC06"] = XC_MGGA_C_CC06; /* Cancio and Chou 2006 */
    f["XC_MGGA_X_MK00"] = XC_MGGA_X_MK00; /* Exchange for accurate virtual orbital energies */
    f["XC_MGGA_C_TPSS"] = XC_MGGA_C_TPSS; /* Perdew, Tao, Staroverov & Scuseria correlation */
    f["XC_MGGA_C_VSXC"] = XC_MGGA_C_VSXC; /* VSxc from Van Voorhis and Scuseria (correlation part) */
    f["XC_MGGA_C_M06_L"] = XC_MGGA_C_M06_L; /* M06-Local functional from Minnesota */
    f["XC_MGGA_C_M06_HF"] = XC_MGGA_C_M06_HF; /* Worker for M06-HF functional */
    f["XC_MGGA_C_M06"] = XC_MGGA_C_M06; /* Worker for M06 functional */
    f["XC_MGGA_C_M06_2X"] = XC_MGGA_C_M06_2X; /* Worker for M06-2X functional */
    f["XC_MGGA_C_M05"] = XC_MGGA_C_M05; /* Worker for M05 functional */
    f["XC_MGGA_C_M05_2X"] = XC_MGGA_C_M05_2X; /* Worker for M05-2X functional */
    f["XC_MGGA_C_PKZB"] = XC_MGGA_C_PKZB; /* Perdew, Kurth, Zupan, and Blaha */
    f["XC_MGGA_C_BC95"] = XC_MGGA_C_BC95; /* Becke correlation 95 */
    f["XC_MGGA_C_REVTPSS"] = XC_MGGA_C_REVTPSS; /* revised TPSS correlation */
    f["XC_MGGA_XC_TPSSLYP1W"] = XC_MGGA_XC_TPSSLYP1W; /* Functionals fitted for water */
    f["XC_MGGA_X_MK00B"] = XC_MGGA_X_MK00B; /* Exchange for accurate virtual orbital energies (v. B) */
    f["XC_MGGA_X_BLOC"] = XC_MGGA_X_BLOC; /* functional with balanced localization */
    f["XC_MGGA_X_MODTPSS"] = XC_MGGA_X_MODTPSS; /* Modified Perdew, Tao, Staroverov & Scuseria exchange */
    f["XC_MGGA_C_TPSSLOC"] = XC_MGGA_C_TPSSLOC; /* Semilocal dynamical correlation */
    f["XC_MGGA_X_MBEEF"] = XC_MGGA_X_MBEEF; /* mBEEF exchange */
    f["XC_MGGA_X_MBEEFVDW"] = XC_MGGA_X_MBEEFVDW; /* mBEEF-vdW exchange */
    f["XC_MGGA_XC_B97M_V"] = XC_MGGA_XC_B97M_V; /* Mardirossian and Head-Gordon */
    f["XC_MGGA_X_MVS"] = XC_MGGA_X_MVS; /* MVS exchange of Sun, Perdew, and Ruzsinszky */
    f["XC_MGGA_X_MN15_L"] = XC_MGGA_X_MN15_L; /* MN15-L functional from Minnesota */
    f["XC_MGGA_C_MN15_L"] = XC_MGGA_C_MN15_L; /* MN15-L functional from Minnesota */
    f["XC_MGGA_X_SCAN"] = XC_MGGA_X_SCAN; /* SCAN exchange of Sun, Ruzsinszky, and Perdew */
    f["XC_MGGA_C_SCAN"] = XC_MGGA_C_SCAN; /* SCAN correlation */
    f["XC_MGGA_C_MN15"] = XC_MGGA_C_MN15; /* MN15 functional from Minnesota */
    f["XC_HYB_MGGA_X_DLDF"] = XC_HYB_MGGA_X_DLDF; /* Dispersionless Density Functional */
    f["XC_HYB_MGGA_X_MS2H"] = XC_HYB_MGGA_X_MS2H; /* MS2 hybrid exchange of Sun, et al */
    f["XC_HYB_MGGA_X_MN12_SX"] = XC_HYB_MGGA_X_MN12_SX; /* MN12-SX hybrid functional from Minnesota */
    f["XC_HYB_MGGA_X_SCAN0"] = XC_HYB_MGGA_X_SCAN0; /* SCAN hybrid */
    f["XC_HYB_MGGA_X_MN15"] = XC_HYB_MGGA_X_MN15; /* MN15 functional from Minnesota */
    f["XC_HYB_MGGA_XC_M05"] = XC_HYB_MGGA_XC_M05; /* M05 functional from Minnesota */
    f["XC_HYB_MGGA_XC_M05_2X"] = XC_HYB_MGGA_XC_M05_2X; /* M05-2X functional from Minnesota */
    f["XC_HYB_MGGA_XC_B88B95"] = XC_HYB_MGGA_XC_B88B95; /* Mixture of B88 with BC95 (B1B95) */
    f["XC_HYB_MGGA_XC_B86B95"] = XC_HYB_MGGA_XC_B86B95; /* Mixture of B86 with BC95 */
    f["XC_HYB_MGGA_XC_PW86B95"] = XC_HYB_MGGA_XC_PW86B95; /* Mixture of PW86 with BC95 */
    f["XC_HYB_MGGA_XC_BB1K"] = XC_HYB_MGGA_XC_BB1K; /* Mixture of B88 with BC95 from Zhao and Truhlar */
    f["XC_HYB_MGGA_XC_M06_HF"] = XC_HYB_MGGA_XC_M06_HF; /* M06-HF functional from Minnesota */
    f["XC_HYB_MGGA_XC_MPW1B95"] = XC_HYB_MGGA_XC_MPW1B95; /* Mixture of mPW91 with BC95 from Zhao and Truhlar */
    f["XC_HYB_MGGA_XC_MPWB1K"] = XC_HYB_MGGA_XC_MPWB1K; /* Mixture of mPW91 with BC95 for kinetics */
    f["XC_HYB_MGGA_XC_X1B95"] = XC_HYB_MGGA_XC_X1B95; /* Mixture of X with BC95 */
    f["XC_HYB_MGGA_XC_XB1K"] = XC_HYB_MGGA_XC_XB1K; /* Mixture of X with BC95 for kinetics */
    f["XC_HYB_MGGA_XC_M06"] = XC_HYB_MGGA_XC_M06; /* M06 functional from Minnesota */
    f["XC_HYB_MGGA_XC_M06_2X"] = XC_HYB_MGGA_XC_M06_2X; /* M06-2X functional from Minnesota */
    f["XC_HYB_MGGA_XC_PW6B95"] = XC_HYB_MGGA_XC_PW6B95; /* Mixture of PW91 with BC95 from Zhao and Truhlar */
    f["XC_HYB_MGGA_XC_PWB6K"] = XC_HYB_MGGA_XC_PWB6K; /* Mixture of PW91 with BC95 from Zhao and Truhlar for kinetics */
    f["XC_HYB_MGGA_XC_TPSSH"] = XC_HYB_MGGA_XC_TPSSH; /* TPSS hybrid */
    f["XC_HYB_MGGA_XC_REVTPSSH"] = XC_HYB_MGGA_XC_REVTPSSH; /* revTPSS hybrid */
    f["XC_HYB_MGGA_XC_M08_HX"] = XC_HYB_MGGA_XC_M08_HX; /* M08-HX functional from Minnesota */
    f["XC_HYB_MGGA_XC_M08_SO"] = XC_HYB_MGGA_XC_M08_SO; /* M08-SO functional from Minnesota */
    f["XC_HYB_MGGA_XC_M11"] = XC_HYB_MGGA_XC_M11; /* M11    functional from Minnesota */
    f["XC_HYB_MGGA_X_MVSH"] = XC_HYB_MGGA_X_MVSH; /* MVS hybrid */
    f["XC_HYB_MGGA_XC_WB97M_V"] = XC_HYB_MGGA_XC_WB97M_V; /* Mardirossian and Head-Gordon */

    return f;
}

/* list of XC functionals from xc_funcs.h */
std::map<std::string, int> libxc_functionals = init_libxc_functionals();

}


