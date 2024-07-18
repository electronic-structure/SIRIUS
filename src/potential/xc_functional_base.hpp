/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file xc_functional.hpp
 *
 *  \brief Contains implementation of sirius::XC_functional_base class.
 */

#ifndef __XC_FUNCTIONAL_BASE_HPP__
#define __XC_FUNCTIONAL_BASE_HPP__

#include <xc.h>
#include <string.h>
#include <memory>
#include <map>
#include <stdexcept>
#include <iostream>
#include "core/rte/rte.hpp"
#include "core/ostream_tools.hpp"

namespace sirius {

const std::map<std::string, int> libxc_functionals = {
#if defined(XC_LDA_X)
        {"XC_LDA_X", XC_LDA_X}, /*Exchange */
#endif
#if defined(XC_LDA_C_WIGNER)
        {"XC_LDA_C_WIGNER", XC_LDA_C_WIGNER}, /*Wigner parametrization */
#endif
#if defined(XC_LDA_C_RPA)
        {"XC_LDA_C_RPA", XC_LDA_C_RPA}, /*Random Phase Approximation */
#endif
#if defined(XC_LDA_C_HL)
        {"XC_LDA_C_HL", XC_LDA_C_HL}, /*Hedin & Lundqvist */
#endif
#if defined(XC_LDA_C_GL)
        {"XC_LDA_C_GL", XC_LDA_C_GL}, /* Gunnarson & Lundqvist */
#endif
#if defined(XC_LDA_C_XALPHA)
        {"XC_LDA_C_XALPHA", XC_LDA_C_XALPHA}, /* Slater Xalpha */
#endif
#if defined(XC_LDA_C_VWN)
        {"XC_LDA_C_VWN", XC_LDA_C_VWN}, /*Vosko, Wilk, & Nusair (5) */
#endif
#if defined(XC_LDA_C_VWN_RPA)
        {"XC_LDA_C_VWN_RPA", XC_LDA_C_VWN_RPA}, /*Vosko, Wilk, & Nusair (RPA) */
#endif
#if defined(XC_LDA_C_PZ)
        {"XC_LDA_C_PZ", XC_LDA_C_PZ}, /*Perdew & Zunger */
#endif
#if defined(XC_LDA_C_PZ_MOD)
        {"XC_LDA_C_PZ_MOD", XC_LDA_C_PZ_MOD}, /* Perdew & Zunger (Modified) */
#endif
#if defined(XC_LDA_C_OB_PZ)
        {"XC_LDA_C_OB_PZ", XC_LDA_C_OB_PZ}, /* Ortiz & Ballone (PZ) */
#endif
#if defined(XC_LDA_C_PW)
        {"XC_LDA_C_PW", XC_LDA_C_PW}, /*Perdew & Wang */
#endif
#if defined(XC_LDA_C_PW_MOD)
        {"XC_LDA_C_PW_MOD", XC_LDA_C_PW_MOD}, /* Perdew & Wang (Modified) */
#endif
#if defined(XC_LDA_C_OB_PW)
        {"XC_LDA_C_OB_PW", XC_LDA_C_OB_PW}, /* Ortiz & Ballone (PW) */
#endif
#if defined(XC_LDA_C_2D_AMGB)
        {"XC_LDA_C_2D_AMGB", XC_LDA_C_2D_AMGB}, /*Attaccalite et al */
#endif
#if defined(XC_LDA_C_2D_PRM)
        {"XC_LDA_C_2D_PRM", XC_LDA_C_2D_PRM}, /*Pittalis, Rasanen & Marques correlation in 2D */
#endif
#if defined(XC_LDA_C_vBH)
        {"XC_LDA_C_vBH", XC_LDA_C_vBH}, /* von Barth & Hedin */
#endif
#if defined(XC_LDA_C_1D_CSC)
        {"XC_LDA_C_1D_CSC", XC_LDA_C_1D_CSC}, /*Casula, Sorella, and Senatore 1D correlation */
#endif
#if defined(XC_LDA_X_2D)
        {"XC_LDA_X_2D", XC_LDA_X_2D}, /*Exchange in 2D */
#endif
#if defined(XC_LDA_XC_TETER93)
        {"XC_LDA_XC_TETER93", XC_LDA_XC_TETER93}, /*Teter 93 parametrization */
#endif
#if defined(XC_LDA_X_1D)
        {"XC_LDA_X_1D", XC_LDA_X_1D}, /*Exchange in 1D */
#endif
#if defined(XC_LDA_C_ML1)
        {"XC_LDA_C_ML1", XC_LDA_C_ML1}, /*Modified LSD (version 1) of Proynov and Salahub */
#endif
#if defined(XC_LDA_C_ML2)
        {"XC_LDA_C_ML2", XC_LDA_C_ML2}, /* Modified LSD (version 2) of Proynov and Salahub */
#endif
#if defined(XC_LDA_C_GOMBAS)
        {"XC_LDA_C_GOMBAS", XC_LDA_C_GOMBAS}, /*Gombas parametrization */
#endif
#if defined(XC_LDA_C_PW_RPA)
        {"XC_LDA_C_PW_RPA", XC_LDA_C_PW_RPA}, /* Perdew & Wang fit of the RPA */
#endif
#if defined(XC_LDA_C_1D_LOOS)
        {"XC_LDA_C_1D_LOOS", XC_LDA_C_1D_LOOS}, /*P-F Loos correlation LDA */
#endif
#if defined(XC_LDA_C_RC04)
        {"XC_LDA_C_RC04", XC_LDA_C_RC04}, /*Ragot-Cortona */
#endif
#if defined(XC_LDA_C_VWN_1)
        {"XC_LDA_C_VWN_1", XC_LDA_C_VWN_1}, /*Vosko, Wilk, & Nusair (1) */
#endif
#if defined(XC_LDA_C_VWN_2)
        {"XC_LDA_C_VWN_2", XC_LDA_C_VWN_2}, /*Vosko, Wilk, & Nusair (2) */
#endif
#if defined(XC_LDA_C_VWN_3)
        {"XC_LDA_C_VWN_3", XC_LDA_C_VWN_3}, /*Vosko, Wilk, & Nusair (3) */
#endif
#if defined(XC_LDA_C_VWN_4)
        {"XC_LDA_C_VWN_4", XC_LDA_C_VWN_4}, /*Vosko, Wilk, & Nusair (4) */
#endif
#if defined(XC_LDA_XC_ZLP)
        {"XC_LDA_XC_ZLP", XC_LDA_XC_ZLP}, /*Zhao, Levy & Parr, Eq. (20) */
#endif
#if defined(XC_LDA_K_TF)
        {"XC_LDA_K_TF", XC_LDA_K_TF}, /*Thomas-Fermi kinetic energy functional */
#endif
#if defined(XC_LDA_K_LP)
        {"XC_LDA_K_LP", XC_LDA_K_LP}, /* Lee and Parr Gaussian ansatz */
#endif
#if defined(XC_LDA_XC_KSDT)
        {"XC_LDA_XC_KSDT", XC_LDA_XC_KSDT}, /*Karasiev et al. parametrization */
#endif
#if defined(XC_LDA_C_CHACHIYO)
        {"XC_LDA_C_CHACHIYO", XC_LDA_C_CHACHIYO}, /*Chachiyo simple 2 parameter correlation */
#endif
#if defined(XC_LDA_C_LP96)
        {"XC_LDA_C_LP96", XC_LDA_C_LP96}, /*Liu-Parr correlation */
#endif
#if defined(XC_LDA_X_REL)
        {"XC_LDA_X_REL", XC_LDA_X_REL}, /*Relativistic exchange */
#endif
#if defined(XC_LDA_XC_1D_EHWLRG_1)
        {"XC_LDA_XC_1D_EHWLRG_1", XC_LDA_XC_1D_EHWLRG_1}, /*LDA constructed from slab-like systems of 1 electron */
#endif
#if defined(XC_LDA_XC_1D_EHWLRG_2)
        {"XC_LDA_XC_1D_EHWLRG_2", XC_LDA_XC_1D_EHWLRG_2}, /* LDA constructed from slab-like systems of 2 electrons */
#endif
#if defined(XC_LDA_XC_1D_EHWLRG_3)
        {"XC_LDA_XC_1D_EHWLRG_3", XC_LDA_XC_1D_EHWLRG_3}, /* LDA constructed from slab-like systems of 3 electrons */
#endif
#if defined(XC_LDA_X_ERF)
        {"XC_LDA_X_ERF", XC_LDA_X_ERF}, /*Attenuated exchange LDA (erf) */
#endif
#if defined(XC_LDA_XC_LP_A)
        {"XC_LDA_XC_LP_A", XC_LDA_XC_LP_A}, /* Lee-Parr reparametrization B */
#endif
#if defined(XC_LDA_XC_LP_B)
        {"XC_LDA_XC_LP_B", XC_LDA_XC_LP_B}, /* Lee-Parr reparametrization B */
#endif
#if defined(XC_LDA_X_RAE)
        {"XC_LDA_X_RAE", XC_LDA_X_RAE}, /* Rae self-energy corrected exchange */
#endif
#if defined(XC_LDA_K_ZLP)
        {"XC_LDA_K_ZLP", XC_LDA_K_ZLP}, /*kinetic energy version of ZLP */
#endif
#if defined(XC_LDA_C_MCWEENY)
        {"XC_LDA_C_MCWEENY", XC_LDA_C_MCWEENY}, /* McWeeny 76 */
#endif
#if defined(XC_LDA_C_BR78)
        {"XC_LDA_C_BR78", XC_LDA_C_BR78}, /* Brual & Rothstein 78 */
#endif
#if defined(XC_LDA_C_PK09)
        {"XC_LDA_C_PK09", XC_LDA_C_PK09}, /*Proynov and Kong 2009 */
#endif
#if defined(XC_LDA_C_OW_LYP)
        {"XC_LDA_C_OW_LYP", XC_LDA_C_OW_LYP}, /* Wigner with corresponding LYP parameters */
#endif
#if defined(XC_LDA_C_OW)
        {"XC_LDA_C_OW", XC_LDA_C_OW}, /* Optimized Wigner */
#endif
#if defined(XC_LDA_XC_GDSMFB)
        {"XC_LDA_XC_GDSMFB", XC_LDA_XC_GDSMFB}, /* Groth et al. parametrization */
#endif
#if defined(XC_LDA_C_GK72)
        {"XC_LDA_C_GK72", XC_LDA_C_GK72}, /*Gordon and Kim 1972 */
#endif
#if defined(XC_LDA_C_KARASIEV)
        {"XC_LDA_C_KARASIEV", XC_LDA_C_KARASIEV}, /* Karasiev reparameterization of Chachiyo */
#endif
#if defined(XC_LDA_K_LP96)
        {"XC_LDA_K_LP96", XC_LDA_K_LP96}, /* Liu-Parr kinetic */
#endif
#if defined(XC_GGA_X_GAM)
        {"XC_GGA_X_GAM", XC_GGA_X_GAM}, /* GAM functional from Minnesota */
#endif
#if defined(XC_GGA_C_GAM)
        {"XC_GGA_C_GAM", XC_GGA_C_GAM}, /* GAM functional from Minnesota */
#endif
#if defined(XC_GGA_X_HCTH_A)
        {"XC_GGA_X_HCTH_A", XC_GGA_X_HCTH_A}, /*HCTH-A */
#endif
#if defined(XC_GGA_X_EV93)
        {"XC_GGA_X_EV93", XC_GGA_X_EV93}, /*Engel and Vosko */
#endif
#if defined(XC_GGA_X_BCGP)
        {"XC_GGA_X_BCGP", XC_GGA_X_BCGP}, /* Burke, Cancio, Gould, and Pittalis */
#endif
#if defined(XC_GGA_C_BCGP)
        {"XC_GGA_C_BCGP", XC_GGA_C_BCGP}, /*Burke, Cancio, Gould, and Pittalis */
#endif
#if defined(XC_GGA_X_LAMBDA_OC2_N)
        {"XC_GGA_X_LAMBDA_OC2_N", XC_GGA_X_LAMBDA_OC2_N}, /* lambda_OC2(N) version of PBE */
#endif
#if defined(XC_GGA_X_B86_R)
        {"XC_GGA_X_B86_R", XC_GGA_X_B86_R}, /* Revised Becke 86 Xalpha,beta,gamma (with mod. grad. correction) */
#endif
#if defined(XC_GGA_X_LAMBDA_CH_N)
        {"XC_GGA_X_LAMBDA_CH_N", XC_GGA_X_LAMBDA_CH_N}, /* lambda_CH(N) version of PBE */
#endif
#if defined(XC_GGA_X_LAMBDA_LO_N)
        {"XC_GGA_X_LAMBDA_LO_N", XC_GGA_X_LAMBDA_LO_N}, /* lambda_LO(N) version of PBE */
#endif
#if defined(XC_GGA_X_HJS_B88_V2)
        {"XC_GGA_X_HJS_B88_V2", XC_GGA_X_HJS_B88_V2}, /*HJS screened exchange corrected B88 version */
#endif
#if defined(XC_GGA_C_Q2D)
        {"XC_GGA_C_Q2D", XC_GGA_C_Q2D}, /*Chiodo et al */
#endif
#if defined(XC_GGA_X_Q2D)
        {"XC_GGA_X_Q2D", XC_GGA_X_Q2D}, /*Chiodo et al */
#endif
#if defined(XC_GGA_X_PBE_MOL)
        {"XC_GGA_X_PBE_MOL", XC_GGA_X_PBE_MOL}, /* Del Campo, Gazquez, Trickey and Vela (PBE-like) */
#endif
#if defined(XC_GGA_K_TFVW)
        {"XC_GGA_K_TFVW", XC_GGA_K_TFVW}, /*Thomas-Fermi plus von Weiszaecker correction */
#endif
#if defined(XC_GGA_K_REVAPBEINT)
        {"XC_GGA_K_REVAPBEINT", XC_GGA_K_REVAPBEINT}, /* interpolated version of REVAPBE */
#endif
#if defined(XC_GGA_K_APBEINT)
        {"XC_GGA_K_APBEINT", XC_GGA_K_APBEINT}, /* interpolated version of APBE */
#endif
#if defined(XC_GGA_K_REVAPBE)
        {"XC_GGA_K_REVAPBE", XC_GGA_K_REVAPBE}, /* revised APBE */
#endif
#if defined(XC_GGA_X_AK13)
        {"XC_GGA_X_AK13", XC_GGA_X_AK13}, /*Armiento & Kuemmel 2013 */
#endif
#if defined(XC_GGA_K_MEYER)
        {"XC_GGA_K_MEYER", XC_GGA_K_MEYER}, /*Meyer, Wang, and Young */
#endif
#if defined(XC_GGA_X_LV_RPW86)
        {"XC_GGA_X_LV_RPW86", XC_GGA_X_LV_RPW86}, /*Berland and Hyldgaard */
#endif
#if defined(XC_GGA_X_PBE_TCA)
        {"XC_GGA_X_PBE_TCA", XC_GGA_X_PBE_TCA}, /* PBE revised by Tognetti et al */
#endif
#if defined(XC_GGA_X_PBEINT)
        {"XC_GGA_X_PBEINT", XC_GGA_X_PBEINT}, /*PBE for hybrid interfaces */
#endif
#if defined(XC_GGA_C_ZPBEINT)
        {"XC_GGA_C_ZPBEINT", XC_GGA_C_ZPBEINT}, /*spin-dependent gradient correction to PBEint */
#endif
#if defined(XC_GGA_C_PBEINT)
        {"XC_GGA_C_PBEINT", XC_GGA_C_PBEINT}, /* PBE for hybrid interfaces */
#endif
#if defined(XC_GGA_C_ZPBESOL)
        {"XC_GGA_C_ZPBESOL", XC_GGA_C_ZPBESOL}, /* spin-dependent gradient correction to PBEsol */
#endif
#if defined(XC_GGA_XC_OPBE_D)
        {"XC_GGA_XC_OPBE_D", XC_GGA_XC_OPBE_D}, /* oPBE_D functional of Goerigk and Grimme */
#endif
#if defined(XC_GGA_XC_OPWLYP_D)
        {"XC_GGA_XC_OPWLYP_D", XC_GGA_XC_OPWLYP_D}, /* oPWLYP-D functional of Goerigk and Grimme */
#endif
#if defined(XC_GGA_XC_OBLYP_D)
        {"XC_GGA_XC_OBLYP_D", XC_GGA_XC_OBLYP_D}, /*oBLYP-D functional of Goerigk and Grimme */
#endif
#if defined(XC_GGA_X_VMT84_GE)
        {"XC_GGA_X_VMT84_GE", XC_GGA_X_VMT84_GE}, /* VMT{8,4} with constraint satisfaction with mu = mu_GE */
#endif
#if defined(XC_GGA_X_VMT84_PBE)
        {"XC_GGA_X_VMT84_PBE", XC_GGA_X_VMT84_PBE}, /*VMT{8,4} with constraint satisfaction with mu = mu_PBE */
#endif
#if defined(XC_GGA_X_VMT_GE)
        {"XC_GGA_X_VMT_GE", XC_GGA_X_VMT_GE}, /* Vela, Medel, and Trickey with mu = mu_GE */
#endif
#if defined(XC_GGA_X_VMT_PBE)
        {"XC_GGA_X_VMT_PBE", XC_GGA_X_VMT_PBE}, /*Vela, Medel, and Trickey with mu = mu_PBE */
#endif
#if defined(XC_GGA_C_N12_SX)
        {"XC_GGA_C_N12_SX", XC_GGA_C_N12_SX}, /* N12-SX functional from Minnesota */
#endif
#if defined(XC_GGA_C_N12)
        {"XC_GGA_C_N12", XC_GGA_C_N12}, /*N12 functional from Minnesota */
#endif
#if defined(XC_GGA_X_N12)
        {"XC_GGA_X_N12", XC_GGA_X_N12}, /*N12 functional from Minnesota */
#endif
#if defined(XC_GGA_C_REGTPSS)
        {"XC_GGA_C_REGTPSS", XC_GGA_C_REGTPSS}, /*Regularized TPSS correlation (ex-VPBE) */
#endif
#if defined(XC_GGA_C_OP_XALPHA)
        {"XC_GGA_C_OP_XALPHA", XC_GGA_C_OP_XALPHA}, /*one-parameter progressive functional (XALPHA version) */
#endif
#if defined(XC_GGA_C_OP_G96)
        {"XC_GGA_C_OP_G96", XC_GGA_C_OP_G96}, /*one-parameter progressive functional (G96 version) */
#endif
#if defined(XC_GGA_C_OP_PBE)
        {"XC_GGA_C_OP_PBE", XC_GGA_C_OP_PBE}, /*one-parameter progressive functional (PBE version) */
#endif
#if defined(XC_GGA_C_OP_B88)
        {"XC_GGA_C_OP_B88", XC_GGA_C_OP_B88}, /*one-parameter progressive functional (B88 version) */
#endif
#if defined(XC_GGA_C_FT97)
        {"XC_GGA_C_FT97", XC_GGA_C_FT97}, /*Filatov & Thiel correlation */
#endif
#if defined(XC_GGA_C_SPBE)
        {"XC_GGA_C_SPBE", XC_GGA_C_SPBE}, /* PBE correlation to be used with the SSB exchange */
#endif
#if defined(XC_GGA_X_SSB_SW)
        {"XC_GGA_X_SSB_SW", XC_GGA_X_SSB_SW}, /*Swart, Sola and Bickelhaupt correction to PBE */
#endif
#if defined(XC_GGA_X_SSB)
        {"XC_GGA_X_SSB", XC_GGA_X_SSB}, /* Swart, Sola and Bickelhaupt */
#endif
#if defined(XC_GGA_X_SSB_D)
        {"XC_GGA_X_SSB_D", XC_GGA_X_SSB_D}, /* Swart, Sola and Bickelhaupt dispersion */
#endif
#if defined(XC_GGA_XC_HCTH_407P)
        {"XC_GGA_XC_HCTH_407P", XC_GGA_XC_HCTH_407P}, /* HCTH/407+ */
#endif
#if defined(XC_GGA_XC_HCTH_P76)
        {"XC_GGA_XC_HCTH_P76", XC_GGA_XC_HCTH_P76}, /* HCTH p=7/6 */
#endif
#if defined(XC_GGA_XC_HCTH_P14)
        {"XC_GGA_XC_HCTH_P14", XC_GGA_XC_HCTH_P14}, /* HCTH p=1/4 */
#endif
#if defined(XC_GGA_XC_B97_GGA1)
        {"XC_GGA_XC_B97_GGA1", XC_GGA_XC_B97_GGA1}, /* Becke 97 GGA-1 */
#endif
#if defined(XC_GGA_C_HCTH_A)
        {"XC_GGA_C_HCTH_A", XC_GGA_C_HCTH_A}, /*HCTH-A */
#endif
#if defined(XC_GGA_X_BPCCAC)
        {"XC_GGA_X_BPCCAC", XC_GGA_X_BPCCAC}, /*BPCCAC (GRAC for the energy) */
#endif
#if defined(XC_GGA_C_REVTCA)
        {"XC_GGA_C_REVTCA", XC_GGA_C_REVTCA}, /*Tognetti, Cortona, Adamo (revised) */
#endif
#if defined(XC_GGA_C_TCA)
        {"XC_GGA_C_TCA", XC_GGA_C_TCA}, /*Tognetti, Cortona, Adamo */
#endif
#if defined(XC_GGA_X_PBE)
        {"XC_GGA_X_PBE", XC_GGA_X_PBE}, /*Perdew, Burke & Ernzerhof exchange */
#endif
#if defined(XC_GGA_X_PBE_R)
        {"XC_GGA_X_PBE_R", XC_GGA_X_PBE_R}, /* Perdew, Burke & Ernzerhof exchange (revised) */
#endif
#if defined(XC_GGA_X_B86)
        {"XC_GGA_X_B86", XC_GGA_X_B86}, /*Becke 86 Xalpha,beta,gamma */
#endif
#if defined(XC_GGA_X_HERMAN)
        {"XC_GGA_X_HERMAN", XC_GGA_X_HERMAN}, /*Herman et al original GGA */
#endif
#if defined(XC_GGA_X_B86_MGC)
        {"XC_GGA_X_B86_MGC", XC_GGA_X_B86_MGC}, /* Becke 86 Xalpha,beta,gamma (with mod. grad. correction) */
#endif
#if defined(XC_GGA_X_B88)
        {"XC_GGA_X_B88", XC_GGA_X_B88}, /*Becke 88 */
#endif
#if defined(XC_GGA_X_G96)
        {"XC_GGA_X_G96", XC_GGA_X_G96}, /*Gill 96 */
#endif
#if defined(XC_GGA_X_PW86)
        {"XC_GGA_X_PW86", XC_GGA_X_PW86}, /*Perdew & Wang 86 */
#endif
#if defined(XC_GGA_X_PW91)
        {"XC_GGA_X_PW91", XC_GGA_X_PW91}, /*Perdew & Wang 91 */
#endif
#if defined(XC_GGA_X_OPTX)
        {"XC_GGA_X_OPTX", XC_GGA_X_OPTX}, /*Handy & Cohen OPTX 01 */
#endif
#if defined(XC_GGA_X_DK87_R1)
        {"XC_GGA_X_DK87_R1", XC_GGA_X_DK87_R1}, /*dePristo & Kress 87 (version R1) */
#endif
#if defined(XC_GGA_X_DK87_R2)
        {"XC_GGA_X_DK87_R2", XC_GGA_X_DK87_R2}, /* dePristo & Kress 87 (version R2) */
#endif
#if defined(XC_GGA_X_LG93)
        {"XC_GGA_X_LG93", XC_GGA_X_LG93}, /*Lacks & Gordon 93 */
#endif
#if defined(XC_GGA_X_FT97_A)
        {"XC_GGA_X_FT97_A", XC_GGA_X_FT97_A}, /*Filatov & Thiel 97 (version A) */
#endif
#if defined(XC_GGA_X_FT97_B)
        {"XC_GGA_X_FT97_B", XC_GGA_X_FT97_B}, /* Filatov & Thiel 97 (version B) */
#endif
#if defined(XC_GGA_X_PBE_SOL)
        {"XC_GGA_X_PBE_SOL", XC_GGA_X_PBE_SOL}, /* Perdew, Burke & Ernzerhof exchange (solids) */
#endif
#if defined(XC_GGA_X_RPBE)
        {"XC_GGA_X_RPBE", XC_GGA_X_RPBE}, /*Hammer, Hansen & Norskov (PBE-like) */
#endif
#if defined(XC_GGA_X_WC)
        {"XC_GGA_X_WC", XC_GGA_X_WC}, /*Wu & Cohen */
#endif
#if defined(XC_GGA_X_MPW91)
        {"XC_GGA_X_MPW91", XC_GGA_X_MPW91}, /* Modified form of PW91 by Adamo & Barone */
#endif
#if defined(XC_GGA_X_AM05)
        {"XC_GGA_X_AM05", XC_GGA_X_AM05}, /*Armiento & Mattsson 05 exchange */
#endif
#if defined(XC_GGA_X_PBEA)
        {"XC_GGA_X_PBEA", XC_GGA_X_PBEA}, /*Madsen (PBE-like) */
#endif
#if defined(XC_GGA_X_MPBE)
        {"XC_GGA_X_MPBE", XC_GGA_X_MPBE}, /*Adamo & Barone modification to PBE */
#endif
#if defined(XC_GGA_X_XPBE)
        {"XC_GGA_X_XPBE", XC_GGA_X_XPBE}, /* xPBE reparametrization by Xu & Goddard */
#endif
#if defined(XC_GGA_X_2D_B86_MGC)
        {"XC_GGA_X_2D_B86_MGC", XC_GGA_X_2D_B86_MGC}, /*Becke 86 MGC for 2D systems */
#endif
#if defined(XC_GGA_X_BAYESIAN)
        {"XC_GGA_X_BAYESIAN", XC_GGA_X_BAYESIAN}, /*Bayesian best fit for the enhancement factor */
#endif
#if defined(XC_GGA_X_PBE_JSJR)
        {"XC_GGA_X_PBE_JSJR", XC_GGA_X_PBE_JSJR}, /* JSJR reparametrization by Pedroza, Silva & Capelle */
#endif
#if defined(XC_GGA_X_2D_B88)
        {"XC_GGA_X_2D_B88", XC_GGA_X_2D_B88}, /*Becke 88 in 2D */
#endif
#if defined(XC_GGA_X_2D_B86)
        {"XC_GGA_X_2D_B86", XC_GGA_X_2D_B86}, /*Becke 86 Xalpha,beta,gamma */
#endif
#if defined(XC_GGA_X_2D_PBE)
        {"XC_GGA_X_2D_PBE", XC_GGA_X_2D_PBE}, /*Perdew, Burke & Ernzerhof exchange in 2D */
#endif
#if defined(XC_GGA_C_PBE)
        {"XC_GGA_C_PBE", XC_GGA_C_PBE}, /*Perdew, Burke & Ernzerhof correlation */
#endif
#if defined(XC_GGA_C_LYP)
        {"XC_GGA_C_LYP", XC_GGA_C_LYP}, /*Lee, Yang & Parr */
#endif
#if defined(XC_GGA_C_P86)
        {"XC_GGA_C_P86", XC_GGA_C_P86}, /*Perdew 86 */
#endif
#if defined(XC_GGA_C_PBE_SOL)
        {"XC_GGA_C_PBE_SOL", XC_GGA_C_PBE_SOL}, /* Perdew, Burke & Ernzerhof correlation SOL */
#endif
#if defined(XC_GGA_C_PW91)
        {"XC_GGA_C_PW91", XC_GGA_C_PW91}, /*Perdew & Wang 91 */
#endif
#if defined(XC_GGA_C_AM05)
        {"XC_GGA_C_AM05", XC_GGA_C_AM05}, /*Armiento & Mattsson 05 correlation */
#endif
#if defined(XC_GGA_C_XPBE)
        {"XC_GGA_C_XPBE", XC_GGA_C_XPBE}, /* xPBE reparametrization by Xu & Goddard */
#endif
#if defined(XC_GGA_C_LM)
        {"XC_GGA_C_LM", XC_GGA_C_LM}, /*Langreth and Mehl correlation */
#endif
#if defined(XC_GGA_C_PBE_JRGX)
        {"XC_GGA_C_PBE_JRGX", XC_GGA_C_PBE_JRGX}, /* JRGX reparametrization by Pedroza, Silva & Capelle */
#endif
#if defined(XC_GGA_X_OPTB88_VDW)
        {"XC_GGA_X_OPTB88_VDW",
         XC_GGA_X_OPTB88_VDW}, /* Becke 88 reoptimized to be used with vdW functional of Dion et al */
#endif
#if defined(XC_GGA_X_PBEK1_VDW)
        {"XC_GGA_X_PBEK1_VDW", XC_GGA_X_PBEK1_VDW}, /* PBE reparametrization for vdW */
#endif
#if defined(XC_GGA_X_OPTPBE_VDW)
        {"XC_GGA_X_OPTPBE_VDW", XC_GGA_X_OPTPBE_VDW}, /* PBE reparametrization for vdW */
#endif
#if defined(XC_GGA_X_RGE2)
        {"XC_GGA_X_RGE2", XC_GGA_X_RGE2}, /*Regularized PBE */
#endif
#if defined(XC_GGA_C_RGE2)
        {"XC_GGA_C_RGE2", XC_GGA_C_RGE2}, /* Regularized PBE */
#endif
#if defined(XC_GGA_X_RPW86)
        {"XC_GGA_X_RPW86", XC_GGA_X_RPW86}, /* refitted Perdew & Wang 86 */
#endif
#if defined(XC_GGA_X_KT1)
        {"XC_GGA_X_KT1", XC_GGA_X_KT1}, /*Exchange part of Keal and Tozer version 1 */
#endif
#if defined(XC_GGA_XC_KT2)
        {"XC_GGA_XC_KT2", XC_GGA_XC_KT2}, /* Keal and Tozer version 2 */
#endif
#if defined(XC_GGA_C_WL)
        {"XC_GGA_C_WL", XC_GGA_C_WL}, /*Wilson & Levy */
#endif
#if defined(XC_GGA_C_WI)
        {"XC_GGA_C_WI", XC_GGA_C_WI}, /* Wilson & Ivanov */
#endif
#if defined(XC_GGA_X_MB88)
        {"XC_GGA_X_MB88", XC_GGA_X_MB88}, /* Modified Becke 88 for proton transfer */
#endif
#if defined(XC_GGA_X_SOGGA)
        {"XC_GGA_X_SOGGA", XC_GGA_X_SOGGA}, /* Second-order generalized gradient approximation */
#endif
#if defined(XC_GGA_X_SOGGA11)
        {"XC_GGA_X_SOGGA11", XC_GGA_X_SOGGA11}, /*Second-order generalized gradient approximation 2011 */
#endif
#if defined(XC_GGA_C_SOGGA11)
        {"XC_GGA_C_SOGGA11", XC_GGA_C_SOGGA11}, /*Second-order generalized gradient approximation 2011 */
#endif
#if defined(XC_GGA_C_WI0)
        {"XC_GGA_C_WI0", XC_GGA_C_WI0}, /*Wilson & Ivanov initial version */
#endif
#if defined(XC_GGA_XC_TH1)
        {"XC_GGA_XC_TH1", XC_GGA_XC_TH1}, /* Tozer and Handy v. 1 */
#endif
#if defined(XC_GGA_XC_TH2)
        {"XC_GGA_XC_TH2", XC_GGA_XC_TH2}, /*Tozer and Handy v. 2 */
#endif
#if defined(XC_GGA_XC_TH3)
        {"XC_GGA_XC_TH3", XC_GGA_XC_TH3}, /*Tozer and Handy v. 3 */
#endif
#if defined(XC_GGA_XC_TH4)
        {"XC_GGA_XC_TH4", XC_GGA_XC_TH4}, /* Tozer and Handy v. 4 */
#endif
#if defined(XC_GGA_X_C09X)
        {"XC_GGA_X_C09X", XC_GGA_X_C09X}, /*C09x to be used with the VdW of Rutgers-Chalmers */
#endif
#if defined(XC_GGA_C_SOGGA11_X)
        {"XC_GGA_C_SOGGA11_X", XC_GGA_C_SOGGA11_X}, /* To be used with HYB_GGA_X_SOGGA11_X */
#endif
#if defined(XC_GGA_X_LB)
        {"XC_GGA_X_LB", XC_GGA_X_LB}, /*van Leeuwen & Baerends */
#endif
#if defined(XC_GGA_XC_HCTH_93)
        {"XC_GGA_XC_HCTH_93", XC_GGA_XC_HCTH_93}, /* HCTH functional fitted to 93 molecules */
#endif
#if defined(XC_GGA_XC_HCTH_120)
        {"XC_GGA_XC_HCTH_120", XC_GGA_XC_HCTH_120}, /* HCTH functional fitted to 120 molecules */
#endif
#if defined(XC_GGA_XC_HCTH_147)
        {"XC_GGA_XC_HCTH_147", XC_GGA_XC_HCTH_147}, /* HCTH functional fitted to 147 molecules */
#endif
#if defined(XC_GGA_XC_HCTH_407)
        {"XC_GGA_XC_HCTH_407", XC_GGA_XC_HCTH_407}, /* HCTH functional fitted to 407 molecules */
#endif
#if defined(XC_GGA_XC_EDF1)
        {"XC_GGA_XC_EDF1", XC_GGA_XC_EDF1}, /*Empirical functionals from Adamson, Gill, and Pople */
#endif
#if defined(XC_GGA_XC_XLYP)
        {"XC_GGA_XC_XLYP", XC_GGA_XC_XLYP}, /*XLYP functional */
#endif
#if defined(XC_GGA_XC_KT1)
        {"XC_GGA_XC_KT1", XC_GGA_XC_KT1}, /* Keal and Tozer version 1 */
#endif
#if defined(XC_GGA_XC_B97_D)
        {"XC_GGA_XC_B97_D", XC_GGA_XC_B97_D}, /*Grimme functional to be used with C6 vdW term */
#endif
#if defined(XC_GGA_XC_PBE1W)
        {"XC_GGA_XC_PBE1W", XC_GGA_XC_PBE1W}, /* Functionals fitted for water */
#endif
#if defined(XC_GGA_XC_MPWLYP1W)
        {"XC_GGA_XC_MPWLYP1W", XC_GGA_XC_MPWLYP1W}, /* Functionals fitted for water */
#endif
#if defined(XC_GGA_XC_PBELYP1W)
        {"XC_GGA_XC_PBELYP1W", XC_GGA_XC_PBELYP1W}, /* Functionals fitted for water */
#endif
#if defined(XC_GGA_X_LBM)
        {"XC_GGA_X_LBM", XC_GGA_X_LBM}, /* van Leeuwen & Baerends modified */
#endif
#if defined(XC_GGA_X_OL2)
        {"XC_GGA_X_OL2", XC_GGA_X_OL2}, /*Exchange form based on Ou-Yang and Levy v.2 */
#endif
#if defined(XC_GGA_X_APBE)
        {"XC_GGA_X_APBE", XC_GGA_X_APBE}, /* mu fixed from the semiclassical neutral atom */
#endif
#if defined(XC_GGA_K_APBE)
        {"XC_GGA_K_APBE", XC_GGA_K_APBE}, /* mu fixed from the semiclassical neutral atom */
#endif
#if defined(XC_GGA_C_APBE)
        {"XC_GGA_C_APBE", XC_GGA_C_APBE}, /* mu fixed from the semiclassical neutral atom */
#endif
#if defined(XC_GGA_K_TW1)
        {"XC_GGA_K_TW1", XC_GGA_K_TW1}, /* Tran and Wesolowski set 1 (Table II) */
#endif
#if defined(XC_GGA_K_TW2)
        {"XC_GGA_K_TW2", XC_GGA_K_TW2}, /* Tran and Wesolowski set 2 (Table II) */
#endif
#if defined(XC_GGA_K_TW3)
        {"XC_GGA_K_TW3", XC_GGA_K_TW3}, /* Tran and Wesolowski set 3 (Table II) */
#endif
#if defined(XC_GGA_K_TW4)
        {"XC_GGA_K_TW4", XC_GGA_K_TW4}, /* Tran and Wesolowski set 4 (Table II) */
#endif
#if defined(XC_GGA_X_HTBS)
        {"XC_GGA_X_HTBS", XC_GGA_X_HTBS}, /*Haas, Tran, Blaha, and Schwarz */
#endif
#if defined(XC_GGA_X_AIRY)
        {"XC_GGA_X_AIRY", XC_GGA_X_AIRY}, /*Constantin et al based on the Airy gas */
#endif
#if defined(XC_GGA_X_LAG)
        {"XC_GGA_X_LAG", XC_GGA_X_LAG}, /*Local Airy Gas */
#endif
#if defined(XC_GGA_XC_MOHLYP)
        {"XC_GGA_XC_MOHLYP", XC_GGA_XC_MOHLYP}, /* Functional for organometallic chemistry */
#endif
#if defined(XC_GGA_XC_MOHLYP2)
        {"XC_GGA_XC_MOHLYP2", XC_GGA_XC_MOHLYP2}, /* Functional for barrier heights */
#endif
#if defined(XC_GGA_XC_TH_FL)
        {"XC_GGA_XC_TH_FL", XC_GGA_XC_TH_FL}, /*Tozer and Handy v. FL */
#endif
#if defined(XC_GGA_XC_TH_FC)
        {"XC_GGA_XC_TH_FC", XC_GGA_XC_TH_FC}, /* Tozer and Handy v. FC */
#endif
#if defined(XC_GGA_XC_TH_FCFO)
        {"XC_GGA_XC_TH_FCFO", XC_GGA_XC_TH_FCFO}, /* Tozer and Handy v. FCFO */
#endif
#if defined(XC_GGA_XC_TH_FCO)
        {"XC_GGA_XC_TH_FCO", XC_GGA_XC_TH_FCO}, /* Tozer and Handy v. FCO */
#endif
#if defined(XC_GGA_C_OPTC)
        {"XC_GGA_C_OPTC", XC_GGA_C_OPTC}, /*Optimized correlation functional of Cohen and Handy */
#endif
#if defined(XC_GGA_C_PBELOC)
        {"XC_GGA_C_PBELOC", XC_GGA_C_PBELOC}, /*Semilocal dynamical correlation */
#endif
#if defined(XC_GGA_XC_VV10)
        {"XC_GGA_XC_VV10", XC_GGA_XC_VV10}, /*Vydrov and Van Voorhis */
#endif
#if defined(XC_GGA_C_PBEFE)
        {"XC_GGA_C_PBEFE", XC_GGA_C_PBEFE}, /* PBE for formation energies */
#endif
#if defined(XC_GGA_C_OP_PW91)
        {"XC_GGA_C_OP_PW91", XC_GGA_C_OP_PW91}, /*one-parameter progressive functional (PW91 version) */
#endif
#if defined(XC_GGA_X_PBEFE)
        {"XC_GGA_X_PBEFE", XC_GGA_X_PBEFE}, /* PBE for formation energies */
#endif
#if defined(XC_GGA_X_CAP)
        {"XC_GGA_X_CAP", XC_GGA_X_CAP}, /*Correct Asymptotic Potential */
#endif
#if defined(XC_GGA_X_EB88)
        {"XC_GGA_X_EB88", XC_GGA_X_EB88}, /* Non-empirical (excogitated) B88 functional of Becke and Elliott */
#endif
#if defined(XC_GGA_C_PBE_MOL)
        {"XC_GGA_C_PBE_MOL", XC_GGA_C_PBE_MOL}, /* Del Campo, Gazquez, Trickey and Vela (PBE-like) */
#endif
#if defined(XC_GGA_K_ABSP3)
        {"XC_GGA_K_ABSP3", XC_GGA_K_ABSP3}, /* gamma-TFvW form by Acharya et al [g = 1 - 1.513/N^0.35] */
#endif
#if defined(XC_GGA_K_ABSP4)
        {"XC_GGA_K_ABSP4", XC_GGA_K_ABSP4}, /* gamma-TFvW form by Acharya et al [g = l = 1/(1 + 1.332/N^(1/3))] */
#endif
#if defined(XC_GGA_C_BMK)
        {"XC_GGA_C_BMK", XC_GGA_C_BMK}, /* Boese-Martin for kinetics */
#endif
#if defined(XC_GGA_C_TAU_HCTH)
        {"XC_GGA_C_TAU_HCTH", XC_GGA_C_TAU_HCTH}, /* correlation part of tau-hcth */
#endif
#if defined(XC_GGA_C_HYB_TAU_HCTH)
        {"XC_GGA_C_HYB_TAU_HCTH", XC_GGA_C_HYB_TAU_HCTH}, /* correlation part of hyb_tau-hcth */
#endif
#if defined(XC_GGA_X_BEEFVDW)
        {"XC_GGA_X_BEEFVDW", XC_GGA_X_BEEFVDW}, /*BEEF-vdW exchange */
#endif
#if defined(XC_GGA_XC_BEEFVDW)
        {"XC_GGA_XC_BEEFVDW", XC_GGA_XC_BEEFVDW}, /* BEEF-vdW exchange-correlation */
#endif
#if defined(XC_GGA_X_PBETRANS)
        {"XC_GGA_X_PBETRANS", XC_GGA_X_PBETRANS}, /*Gradient-based interpolation between PBE and revPBE */
#endif
#if defined(XC_GGA_X_CHACHIYO)
        {"XC_GGA_X_CHACHIYO", XC_GGA_X_CHACHIYO}, /*Chachiyo exchange */
#endif
#if defined(XC_GGA_K_VW)
        {"XC_GGA_K_VW", XC_GGA_K_VW}, /* von Weiszaecker functional */
#endif
#if defined(XC_GGA_K_GE2)
        {"XC_GGA_K_GE2", XC_GGA_K_GE2}, /* Second-order gradient expansion (l = 1/9) */
#endif
#if defined(XC_GGA_K_GOLDEN)
        {"XC_GGA_K_GOLDEN", XC_GGA_K_GOLDEN}, /* TF-lambda-vW form by Golden (l = 13/45) */
#endif
#if defined(XC_GGA_K_YT65)
        {"XC_GGA_K_YT65", XC_GGA_K_YT65}, /* TF-lambda-vW form by Yonei and Tomishima (l = 1/5) */
#endif
#if defined(XC_GGA_K_BALTIN)
        {"XC_GGA_K_BALTIN", XC_GGA_K_BALTIN}, /* TF-lambda-vW form by Baltin (l = 5/9) */
#endif
#if defined(XC_GGA_K_LIEB)
        {"XC_GGA_K_LIEB", XC_GGA_K_LIEB}, /* TF-lambda-vW form by Lieb (l = 0.185909191) */
#endif
#if defined(XC_GGA_K_ABSP1)
        {"XC_GGA_K_ABSP1", XC_GGA_K_ABSP1}, /* gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)] */
#endif
#if defined(XC_GGA_K_ABSP2)
        {"XC_GGA_K_ABSP2", XC_GGA_K_ABSP2}, /* gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)] */
#endif
#if defined(XC_GGA_K_GR)
        {"XC_GGA_K_GR", XC_GGA_K_GR}, /* gamma-TFvW form by Gazquez and Robles */
#endif
#if defined(XC_GGA_K_LUDENA)
        {"XC_GGA_K_LUDENA", XC_GGA_K_LUDENA}, /* gamma-TFvW form by Ludena */
#endif
#if defined(XC_GGA_K_GP85)
        {"XC_GGA_K_GP85", XC_GGA_K_GP85}, /* gamma-TFvW form by Ghosh and Parr */
#endif
#if defined(XC_GGA_K_PEARSON)
        {"XC_GGA_K_PEARSON", XC_GGA_K_PEARSON}, /*Pearson */
#endif
#if defined(XC_GGA_K_OL1)
        {"XC_GGA_K_OL1", XC_GGA_K_OL1}, /*Ou-Yang and Levy v.1 */
#endif
#if defined(XC_GGA_K_OL2)
        {"XC_GGA_K_OL2", XC_GGA_K_OL2}, /* Ou-Yang and Levy v.2 */
#endif
#if defined(XC_GGA_K_FR_B88)
        {"XC_GGA_K_FR_B88", XC_GGA_K_FR_B88}, /* Fuentealba & Reyes (B88 version) */
#endif
#if defined(XC_GGA_K_FR_PW86)
        {"XC_GGA_K_FR_PW86", XC_GGA_K_FR_PW86}, /* Fuentealba & Reyes (PW86 version) */
#endif
#if defined(XC_GGA_K_DK)
        {"XC_GGA_K_DK", XC_GGA_K_DK}, /*DePristo and Kress */
#endif
#if defined(XC_GGA_K_PERDEW)
        {"XC_GGA_K_PERDEW", XC_GGA_K_PERDEW}, /* Perdew */
#endif
#if defined(XC_GGA_K_VSK)
        {"XC_GGA_K_VSK", XC_GGA_K_VSK}, /* Vitos, Skriver, and Kollar */
#endif
#if defined(XC_GGA_K_VJKS)
        {"XC_GGA_K_VJKS", XC_GGA_K_VJKS}, /* Vitos, Johansson, Kollar, and Skriver */
#endif
#if defined(XC_GGA_K_ERNZERHOF)
        {"XC_GGA_K_ERNZERHOF", XC_GGA_K_ERNZERHOF}, /* Ernzerhof */
#endif
#if defined(XC_GGA_K_LC94)
        {"XC_GGA_K_LC94", XC_GGA_K_LC94}, /* Lembarki & Chermette */
#endif
#if defined(XC_GGA_K_LLP)
        {"XC_GGA_K_LLP", XC_GGA_K_LLP}, /* Lee, Lee & Parr */
#endif
#if defined(XC_GGA_K_THAKKAR)
        {"XC_GGA_K_THAKKAR", XC_GGA_K_THAKKAR}, /*Thakkar 1992 */
#endif
#if defined(XC_GGA_X_WPBEH)
        {"XC_GGA_X_WPBEH", XC_GGA_X_WPBEH}, /*short-range version of the PBE */
#endif
#if defined(XC_GGA_X_HJS_PBE)
        {"XC_GGA_X_HJS_PBE", XC_GGA_X_HJS_PBE}, /*HJS screened exchange PBE version */
#endif
#if defined(XC_GGA_X_HJS_PBE_SOL)
        {"XC_GGA_X_HJS_PBE_SOL", XC_GGA_X_HJS_PBE_SOL}, /* HJS screened exchange PBE_SOL version */
#endif
#if defined(XC_GGA_X_HJS_B88)
        {"XC_GGA_X_HJS_B88", XC_GGA_X_HJS_B88}, /* HJS screened exchange B88 version */
#endif
#if defined(XC_GGA_X_HJS_B97X)
        {"XC_GGA_X_HJS_B97X", XC_GGA_X_HJS_B97X}, /* HJS screened exchange B97x version */
#endif
#if defined(XC_GGA_X_ITYH)
        {"XC_GGA_X_ITYH", XC_GGA_X_ITYH}, /*short-range recipe for exchange GGA functionals */
#endif
#if defined(XC_GGA_X_SFAT)
        {"XC_GGA_X_SFAT", XC_GGA_X_SFAT}, /*short-range recipe for exchange GGA functionals */
#endif
#if defined(XC_GGA_X_SG4)
        {"XC_GGA_X_SG4", XC_GGA_X_SG4}, /*Semiclassical GGA at fourth order */
#endif
#if defined(XC_GGA_C_SG4)
        {"XC_GGA_C_SG4", XC_GGA_C_SG4}, /*Semiclassical GGA at fourth order */
#endif
#if defined(XC_GGA_X_GG99)
        {"XC_GGA_X_GG99", XC_GGA_X_GG99}, /*Gilbert and Gill 1999 */
#endif
#if defined(XC_GGA_X_PBEpow)
        {"XC_GGA_X_PBEpow", XC_GGA_X_PBEpow}, /*PBE power */
#endif
#if defined(XC_GGA_X_KGG99)
        {"XC_GGA_X_KGG99", XC_GGA_X_KGG99}, /* Gilbert and Gill 1999 (mixed) */
#endif
#if defined(XC_GGA_XC_HLE16)
        {"XC_GGA_XC_HLE16", XC_GGA_XC_HLE16}, /* high local exchange 2016 */
#endif
#if defined(XC_GGA_C_SCAN_E0)
        {"XC_GGA_C_SCAN_E0", XC_GGA_C_SCAN_E0}, /*GGA component of SCAN */
#endif
#if defined(XC_GGA_C_GAPC)
        {"XC_GGA_C_GAPC", XC_GGA_C_GAPC}, /*GapC */
#endif
#if defined(XC_GGA_C_GAPLOC)
        {"XC_GGA_C_GAPLOC", XC_GGA_C_GAPLOC}, /*Gaploc */
#endif
#if defined(XC_GGA_C_ZVPBEINT)
        {"XC_GGA_C_ZVPBEINT", XC_GGA_C_ZVPBEINT}, /*another spin-dependent correction to PBEint */
#endif
#if defined(XC_GGA_C_ZVPBESOL)
        {"XC_GGA_C_ZVPBESOL", XC_GGA_C_ZVPBESOL}, /* another spin-dependent correction to PBEsol */
#endif
#if defined(XC_GGA_C_TM_LYP)
        {"XC_GGA_C_TM_LYP", XC_GGA_C_TM_LYP}, /* Takkar and McCarthy reparametrization */
#endif
#if defined(XC_GGA_C_TM_PBE)
        {"XC_GGA_C_TM_PBE", XC_GGA_C_TM_PBE}, /* Thakkar and McCarthy reparametrization */
#endif
#if defined(XC_GGA_C_W94)
        {"XC_GGA_C_W94", XC_GGA_C_W94}, /*Wilson 94 (Eq. 25) */
#endif
#if defined(XC_GGA_C_CS1)
        {"XC_GGA_C_CS1", XC_GGA_C_CS1}, /*A dynamical correlation functional */
#endif
#if defined(XC_GGA_X_B88M)
        {"XC_GGA_X_B88M", XC_GGA_X_B88M}, /* Becke 88 reoptimized to be used with mgga_c_tau1 */
#endif
#if defined(XC_GGA_K_PBE3)
        {"XC_GGA_K_PBE3", XC_GGA_K_PBE3}, /* Three parameter PBE-like expansion */
#endif
#if defined(XC_GGA_K_PBE4)
        {"XC_GGA_K_PBE4", XC_GGA_K_PBE4}, /* Four parameter PBE-like expansion */
#endif
#if defined(XC_GGA_K_EXP4)
        {"XC_GGA_K_EXP4", XC_GGA_K_EXP4}, /*Intermediate form between PBE3 and PBE4 */
#endif
#if defined(XC_HYB_GGA_X_N12_SX)
        {"XC_HYB_GGA_X_N12_SX", XC_HYB_GGA_X_N12_SX}, /*N12-SX functional from Minnesota */
#endif
#if defined(XC_HYB_GGA_XC_B97_1p)
        {"XC_HYB_GGA_XC_B97_1p", XC_HYB_GGA_XC_B97_1p}, /* version of B97 by Cohen and Handy */
#endif
#if defined(XC_HYB_GGA_XC_PBE_MOL0)
        {"XC_HYB_GGA_XC_PBE_MOL0", XC_HYB_GGA_XC_PBE_MOL0}, /* PBEmol0 */
#endif
#if defined(XC_HYB_GGA_XC_PBE_SOL0)
        {"XC_HYB_GGA_XC_PBE_SOL0", XC_HYB_GGA_XC_PBE_SOL0}, /* PBEsol0 */
#endif
#if defined(XC_HYB_GGA_XC_PBEB0)
        {"XC_HYB_GGA_XC_PBEB0", XC_HYB_GGA_XC_PBEB0}, /* PBEbeta0 */
#endif
#if defined(XC_HYB_GGA_XC_PBE_MOLB0)
        {"XC_HYB_GGA_XC_PBE_MOLB0", XC_HYB_GGA_XC_PBE_MOLB0}, /* PBEmolbeta0 */
#endif
#if defined(XC_HYB_GGA_XC_PBE50)
        {"XC_HYB_GGA_XC_PBE50", XC_HYB_GGA_XC_PBE50}, /* PBE0 with 50% exx */
#endif
#if defined(XC_HYB_GGA_XC_B3PW91)
        {"XC_HYB_GGA_XC_B3PW91", XC_HYB_GGA_XC_B3PW91}, /*The original (ACM) hybrid of Becke */
#endif
#if defined(XC_HYB_GGA_XC_B3LYP)
        {"XC_HYB_GGA_XC_B3LYP", XC_HYB_GGA_XC_B3LYP}, /* The (in)famous B3LYP */
#endif
#if defined(XC_HYB_GGA_XC_B3P86)
        {"XC_HYB_GGA_XC_B3P86", XC_HYB_GGA_XC_B3P86}, /* Perdew 86 hybrid similar to B3PW91 */
#endif
#if defined(XC_HYB_GGA_XC_O3LYP)
        {"XC_HYB_GGA_XC_O3LYP", XC_HYB_GGA_XC_O3LYP}, /*hybrid using the optx functional */
#endif
#if defined(XC_HYB_GGA_XC_MPW1K)
        {"XC_HYB_GGA_XC_MPW1K", XC_HYB_GGA_XC_MPW1K}, /* mixture of mPW91 and PW91 optimized for kinetics */
#endif
#if defined(XC_HYB_GGA_XC_PBEH)
        {"XC_HYB_GGA_XC_PBEH", XC_HYB_GGA_XC_PBEH}, /*aka PBE0 or PBE1PBE */
#endif
#if defined(XC_HYB_GGA_XC_B97)
        {"XC_HYB_GGA_XC_B97", XC_HYB_GGA_XC_B97}, /*Becke 97 */
#endif
#if defined(XC_HYB_GGA_XC_B97_1)
        {"XC_HYB_GGA_XC_B97_1", XC_HYB_GGA_XC_B97_1}, /* Becke 97-1 */
#endif
#if defined(XC_HYB_GGA_XC_B97_2)
        {"XC_HYB_GGA_XC_B97_2", XC_HYB_GGA_XC_B97_2}, /* Becke 97-2 */
#endif
#if defined(XC_HYB_GGA_XC_X3LYP)
        {"XC_HYB_GGA_XC_X3LYP", XC_HYB_GGA_XC_X3LYP}, /* hybrid by Xu and Goddard */
#endif
#if defined(XC_HYB_GGA_XC_B1WC)
        {"XC_HYB_GGA_XC_B1WC", XC_HYB_GGA_XC_B1WC}, /*Becke 1-parameter mixture of WC and PBE */
#endif
#if defined(XC_HYB_GGA_XC_B97_K)
        {"XC_HYB_GGA_XC_B97_K", XC_HYB_GGA_XC_B97_K}, /* Boese-Martin for Kinetics */
#endif
#if defined(XC_HYB_GGA_XC_B97_3)
        {"XC_HYB_GGA_XC_B97_3", XC_HYB_GGA_XC_B97_3}, /* Becke 97-3 */
#endif
#if defined(XC_HYB_GGA_XC_MPW3PW)
        {"XC_HYB_GGA_XC_MPW3PW", XC_HYB_GGA_XC_MPW3PW}, /* mixture with the mPW functional */
#endif
#if defined(XC_HYB_GGA_XC_B1LYP)
        {"XC_HYB_GGA_XC_B1LYP", XC_HYB_GGA_XC_B1LYP}, /* Becke 1-parameter mixture of B88 and LYP */
#endif
#if defined(XC_HYB_GGA_XC_B1PW91)
        {"XC_HYB_GGA_XC_B1PW91", XC_HYB_GGA_XC_B1PW91}, /* Becke 1-parameter mixture of B88 and PW91 */
#endif
#if defined(XC_HYB_GGA_XC_MPW1PW)
        {"XC_HYB_GGA_XC_MPW1PW", XC_HYB_GGA_XC_MPW1PW}, /* Becke 1-parameter mixture of mPW91 and PW91 */
#endif
#if defined(XC_HYB_GGA_XC_MPW3LYP)
        {"XC_HYB_GGA_XC_MPW3LYP", XC_HYB_GGA_XC_MPW3LYP}, /* mixture of mPW and LYP */
#endif
#if defined(XC_HYB_GGA_XC_SB98_1a)
        {"XC_HYB_GGA_XC_SB98_1a", XC_HYB_GGA_XC_SB98_1a}, /* Schmider-Becke 98 parameterization 1a */
#endif
#if defined(XC_HYB_GGA_XC_SB98_1b)
        {"XC_HYB_GGA_XC_SB98_1b", XC_HYB_GGA_XC_SB98_1b}, /* Schmider-Becke 98 parameterization 1b */
#endif
#if defined(XC_HYB_GGA_XC_SB98_1c)
        {"XC_HYB_GGA_XC_SB98_1c", XC_HYB_GGA_XC_SB98_1c}, /* Schmider-Becke 98 parameterization 1c */
#endif
#if defined(XC_HYB_GGA_XC_SB98_2a)
        {"XC_HYB_GGA_XC_SB98_2a", XC_HYB_GGA_XC_SB98_2a}, /* Schmider-Becke 98 parameterization 2a */
#endif
#if defined(XC_HYB_GGA_XC_SB98_2b)
        {"XC_HYB_GGA_XC_SB98_2b", XC_HYB_GGA_XC_SB98_2b}, /* Schmider-Becke 98 parameterization 2b */
#endif
#if defined(XC_HYB_GGA_XC_SB98_2c)
        {"XC_HYB_GGA_XC_SB98_2c", XC_HYB_GGA_XC_SB98_2c}, /* Schmider-Becke 98 parameterization 2c */
#endif
#if defined(XC_HYB_GGA_X_SOGGA11_X)
        {"XC_HYB_GGA_X_SOGGA11_X", XC_HYB_GGA_X_SOGGA11_X}, /*Hybrid based on SOGGA11 form */
#endif
#if defined(XC_HYB_GGA_XC_HSE03)
        {"XC_HYB_GGA_XC_HSE03", XC_HYB_GGA_XC_HSE03}, /*the 2003 version of the screened hybrid HSE */
#endif
#if defined(XC_HYB_GGA_XC_HSE06)
        {"XC_HYB_GGA_XC_HSE06", XC_HYB_GGA_XC_HSE06}, /* the 2006 version of the screened hybrid HSE */
#endif
#if defined(XC_HYB_GGA_XC_HJS_PBE)
        {"XC_HYB_GGA_XC_HJS_PBE", XC_HYB_GGA_XC_HJS_PBE}, /* HJS hybrid screened exchange PBE version */
#endif
#if defined(XC_HYB_GGA_XC_HJS_PBE_SOL)
        {"XC_HYB_GGA_XC_HJS_PBE_SOL", XC_HYB_GGA_XC_HJS_PBE_SOL}, /* HJS hybrid screened exchange PBE_SOL version */
#endif
#if defined(XC_HYB_GGA_XC_HJS_B88)
        {"XC_HYB_GGA_XC_HJS_B88", XC_HYB_GGA_XC_HJS_B88}, /* HJS hybrid screened exchange B88 version */
#endif
#if defined(XC_HYB_GGA_XC_HJS_B97X)
        {"XC_HYB_GGA_XC_HJS_B97X", XC_HYB_GGA_XC_HJS_B97X}, /* HJS hybrid screened exchange B97x version */
#endif
#if defined(XC_HYB_GGA_XC_CAM_B3LYP)
        {"XC_HYB_GGA_XC_CAM_B3LYP", XC_HYB_GGA_XC_CAM_B3LYP}, /*CAM version of B3LYP */
#endif
#if defined(XC_HYB_GGA_XC_TUNED_CAM_B3LYP)
        {"XC_HYB_GGA_XC_TUNED_CAM_B3LYP",
         XC_HYB_GGA_XC_TUNED_CAM_B3LYP}, /* CAM version of B3LYP tuned for excitations */
#endif
#if defined(XC_HYB_GGA_XC_BHANDH)
        {"XC_HYB_GGA_XC_BHANDH", XC_HYB_GGA_XC_BHANDH}, /* Becke half-and-half */
#endif
#if defined(XC_HYB_GGA_XC_BHANDHLYP)
        {"XC_HYB_GGA_XC_BHANDHLYP", XC_HYB_GGA_XC_BHANDHLYP}, /* Becke half-and-half with B88 exchange */
#endif
#if defined(XC_HYB_GGA_XC_MB3LYP_RC04)
        {"XC_HYB_GGA_XC_MB3LYP_RC04", XC_HYB_GGA_XC_MB3LYP_RC04}, /* B3LYP with RC04 LDA */
#endif
#if defined(XC_HYB_GGA_XC_MPWLYP1M)
        {"XC_HYB_GGA_XC_MPWLYP1M", XC_HYB_GGA_XC_MPWLYP1M}, /* MPW with 1 par. for metals/LYP */
#endif
#if defined(XC_HYB_GGA_XC_REVB3LYP)
        {"XC_HYB_GGA_XC_REVB3LYP", XC_HYB_GGA_XC_REVB3LYP}, /* Revised B3LYP */
#endif
#if defined(XC_HYB_GGA_XC_CAMY_BLYP)
        {"XC_HYB_GGA_XC_CAMY_BLYP", XC_HYB_GGA_XC_CAMY_BLYP}, /*BLYP with yukawa screening */
#endif
#if defined(XC_HYB_GGA_XC_PBE0_13)
        {"XC_HYB_GGA_XC_PBE0_13", XC_HYB_GGA_XC_PBE0_13}, /* PBE0-1/3 */
#endif
#if defined(XC_HYB_GGA_XC_B3LYPs)
        {"XC_HYB_GGA_XC_B3LYPs", XC_HYB_GGA_XC_B3LYPs}, /* B3LYP* functional */
#endif
#if defined(XC_HYB_GGA_XC_WB97)
        {"XC_HYB_GGA_XC_WB97", XC_HYB_GGA_XC_WB97}, /*Chai and Head-Gordon */
#endif
#if defined(XC_HYB_GGA_XC_WB97X)
        {"XC_HYB_GGA_XC_WB97X", XC_HYB_GGA_XC_WB97X}, /* Chai and Head-Gordon */
#endif
#if defined(XC_HYB_GGA_XC_LRC_WPBEH)
        {"XC_HYB_GGA_XC_LRC_WPBEH", XC_HYB_GGA_XC_LRC_WPBEH}, /* Long-range corrected functional by Rorhdanz et al */
#endif
#if defined(XC_HYB_GGA_XC_WB97X_V)
        {"XC_HYB_GGA_XC_WB97X_V", XC_HYB_GGA_XC_WB97X_V}, /* Mardirossian and Head-Gordon */
#endif
#if defined(XC_HYB_GGA_XC_LCY_PBE)
        {"XC_HYB_GGA_XC_LCY_PBE", XC_HYB_GGA_XC_LCY_PBE}, /*PBE with yukawa screening */
#endif
#if defined(XC_HYB_GGA_XC_LCY_BLYP)
        {"XC_HYB_GGA_XC_LCY_BLYP", XC_HYB_GGA_XC_LCY_BLYP}, /*BLYP with yukawa screening */
#endif
#if defined(XC_HYB_GGA_XC_LC_VV10)
        {"XC_HYB_GGA_XC_LC_VV10", XC_HYB_GGA_XC_LC_VV10}, /*Vydrov and Van Voorhis */
#endif
#if defined(XC_HYB_GGA_XC_CAMY_B3LYP)
        {"XC_HYB_GGA_XC_CAMY_B3LYP", XC_HYB_GGA_XC_CAMY_B3LYP}, /*B3LYP with Yukawa screening */
#endif
#if defined(XC_HYB_GGA_XC_WB97X_D)
        {"XC_HYB_GGA_XC_WB97X_D", XC_HYB_GGA_XC_WB97X_D}, /* Chai and Head-Gordon */
#endif
#if defined(XC_HYB_GGA_XC_HPBEINT)
        {"XC_HYB_GGA_XC_HPBEINT", XC_HYB_GGA_XC_HPBEINT}, /* hPBEint */
#endif
#if defined(XC_HYB_GGA_XC_LRC_WPBE)
        {"XC_HYB_GGA_XC_LRC_WPBE", XC_HYB_GGA_XC_LRC_WPBE}, /* Long-range corrected functional by Rorhdanz et al */
#endif
#if defined(XC_HYB_GGA_XC_B3LYP5)
        {"XC_HYB_GGA_XC_B3LYP5", XC_HYB_GGA_XC_B3LYP5}, /* B3LYP with VWN functional 5 instead of RPA */
#endif
#if defined(XC_HYB_GGA_XC_EDF2)
        {"XC_HYB_GGA_XC_EDF2", XC_HYB_GGA_XC_EDF2}, /*Empirical functional from Lin, George and Gill */
#endif
#if defined(XC_HYB_GGA_XC_CAP0)
        {"XC_HYB_GGA_XC_CAP0", XC_HYB_GGA_XC_CAP0}, /*Correct Asymptotic Potential hybrid */
#endif
#if defined(XC_HYB_GGA_XC_LC_WPBE)
        {"XC_HYB_GGA_XC_LC_WPBE", XC_HYB_GGA_XC_LC_WPBE}, /* Long-range corrected functional by Vydrov and Scuseria */
#endif
#if defined(XC_HYB_GGA_XC_HSE12)
        {"XC_HYB_GGA_XC_HSE12", XC_HYB_GGA_XC_HSE12}, /* HSE12 by Moussa, Schultz and Chelikowsky */
#endif
#if defined(XC_HYB_GGA_XC_HSE12S)
        {"XC_HYB_GGA_XC_HSE12S", XC_HYB_GGA_XC_HSE12S}, /* Short-range HSE12 by Moussa, Schultz, and Chelikowsky */
#endif
#if defined(XC_HYB_GGA_XC_HSE_SOL)
        {"XC_HYB_GGA_XC_HSE_SOL", XC_HYB_GGA_XC_HSE_SOL}, /* HSEsol functional by Schimka, Harl, and Kresse */
#endif
#if defined(XC_HYB_GGA_XC_CAM_QTP_01)
        {"XC_HYB_GGA_XC_CAM_QTP_01",
         XC_HYB_GGA_XC_CAM_QTP_01}, /* CAM-QTP(01): CAM-B3LYP retuned using ionization potentials of water */
#endif
#if defined(XC_HYB_GGA_XC_MPW1LYP)
        {"XC_HYB_GGA_XC_MPW1LYP", XC_HYB_GGA_XC_MPW1LYP}, /* Becke 1-parameter mixture of mPW91 and LYP */
#endif
#if defined(XC_HYB_GGA_XC_MPW1PBE)
        {"XC_HYB_GGA_XC_MPW1PBE", XC_HYB_GGA_XC_MPW1PBE}, /* Becke 1-parameter mixture of mPW91 and PBE */
#endif
#if defined(XC_HYB_GGA_XC_KMLYP)
        {"XC_HYB_GGA_XC_KMLYP", XC_HYB_GGA_XC_KMLYP}, /* Kang-Musgrave hybrid */
#endif
#if defined(XC_HYB_GGA_XC_B5050LYP)
        {"XC_HYB_GGA_XC_B5050LYP", XC_HYB_GGA_XC_B5050LYP}, /* Like B3LYP but more exact exchange */
#endif
#if defined(XC_MGGA_C_DLDF)
        {"XC_MGGA_C_DLDF", XC_MGGA_C_DLDF}, /* Dispersionless Density Functional */
#endif
#if defined(XC_MGGA_XC_ZLP)
        {"XC_MGGA_XC_ZLP", XC_MGGA_XC_ZLP}, /*Zhao, Levy & Parr, Eq. (21) */
#endif
#if defined(XC_MGGA_XC_OTPSS_D)
        {"XC_MGGA_XC_OTPSS_D", XC_MGGA_XC_OTPSS_D}, /*oTPSS_D functional of Goerigk and Grimme */
#endif
#if defined(XC_MGGA_C_CS)
        {"XC_MGGA_C_CS", XC_MGGA_C_CS}, /*Colle and Salvetti */
#endif
#if defined(XC_MGGA_C_MN12_SX)
        {"XC_MGGA_C_MN12_SX", XC_MGGA_C_MN12_SX}, /* MN12-SX correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_MN12_L)
        {"XC_MGGA_C_MN12_L", XC_MGGA_C_MN12_L}, /* MN12-L correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M11_L)
        {"XC_MGGA_C_M11_L", XC_MGGA_C_M11_L}, /* M11-L correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M11)
        {"XC_MGGA_C_M11", XC_MGGA_C_M11}, /* M11 correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M08_SO)
        {"XC_MGGA_C_M08_SO", XC_MGGA_C_M08_SO}, /* M08-SO correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M08_HX)
        {"XC_MGGA_C_M08_HX", XC_MGGA_C_M08_HX}, /*M08-HX correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_X_LTA)
        {"XC_MGGA_X_LTA", XC_MGGA_X_LTA}, /*Local tau approximation of Ernzerhof & Scuseria */
#endif
#if defined(XC_MGGA_X_TPSS)
        {"XC_MGGA_X_TPSS", XC_MGGA_X_TPSS}, /*Tao, Perdew, Staroverov & Scuseria exchange */
#endif
#if defined(XC_MGGA_X_M06_L)
        {"XC_MGGA_X_M06_L", XC_MGGA_X_M06_L}, /*M06-L exchange functional from Minnesota */
#endif
#if defined(XC_MGGA_X_GVT4)
        {"XC_MGGA_X_GVT4", XC_MGGA_X_GVT4}, /*GVT4 from Van Voorhis and Scuseria */
#endif
#if defined(XC_MGGA_X_TAU_HCTH)
        {"XC_MGGA_X_TAU_HCTH", XC_MGGA_X_TAU_HCTH}, /*tau-HCTH from Boese and Handy */
#endif
#if defined(XC_MGGA_X_BR89)
        {"XC_MGGA_X_BR89", XC_MGGA_X_BR89}, /*Becke-Roussel 89 */
#endif
#if defined(XC_MGGA_X_BJ06)
        {"XC_MGGA_X_BJ06", XC_MGGA_X_BJ06}, /* Becke & Johnson correction to Becke-Roussel 89 */
#endif
#if defined(XC_MGGA_X_TB09)
        {"XC_MGGA_X_TB09", XC_MGGA_X_TB09}, /* Tran & Blaha correction to Becke & Johnson */
#endif
#if defined(XC_MGGA_X_RPP09)
        {"XC_MGGA_X_RPP09", XC_MGGA_X_RPP09}, /* Rasanen, Pittalis, and Proetto correction to Becke & Johnson */
#endif
#if defined(XC_MGGA_X_2D_PRHG07)
        {"XC_MGGA_X_2D_PRHG07", XC_MGGA_X_2D_PRHG07}, /*Pittalis, Rasanen, Helbig, Gross Exchange Functional */
#endif
#if defined(XC_MGGA_X_2D_PRHG07_PRP10)
        {"XC_MGGA_X_2D_PRHG07_PRP10", XC_MGGA_X_2D_PRHG07_PRP10}, /* PRGH07 with PRP10 correction */
#endif
#if defined(XC_MGGA_X_REVTPSS)
        {"XC_MGGA_X_REVTPSS", XC_MGGA_X_REVTPSS}, /* revised Tao, Perdew, Staroverov & Scuseria exchange */
#endif
#if defined(XC_MGGA_X_PKZB)
        {"XC_MGGA_X_PKZB", XC_MGGA_X_PKZB}, /*Perdew, Kurth, Zupan, and Blaha */
#endif
#if defined(XC_MGGA_X_MS0)
        {"XC_MGGA_X_MS0", XC_MGGA_X_MS0}, /*MS exchange of Sun, Xiao, and Ruzsinszky */
#endif
#if defined(XC_MGGA_X_MS1)
        {"XC_MGGA_X_MS1", XC_MGGA_X_MS1}, /* MS1 exchange of Sun, et al */
#endif
#if defined(XC_MGGA_X_MS2)
        {"XC_MGGA_X_MS2", XC_MGGA_X_MS2}, /* MS2 exchange of Sun, et al */
#endif
#if defined(XC_MGGA_X_M11_L)
        {"XC_MGGA_X_M11_L", XC_MGGA_X_M11_L}, /*M11-L exchange functional from Minnesota */
#endif
#if defined(XC_MGGA_X_MN12_L)
        {"XC_MGGA_X_MN12_L", XC_MGGA_X_MN12_L}, /*MN12-L exchange functional from Minnesota */
#endif
#if defined(XC_MGGA_XC_CC06)
        {"XC_MGGA_XC_CC06", XC_MGGA_XC_CC06}, /*Cancio and Chou 2006 */
#endif
#if defined(XC_MGGA_X_MK00)
        {"XC_MGGA_X_MK00", XC_MGGA_X_MK00}, /*Exchange for accurate virtual orbital energies */
#endif
#if defined(XC_MGGA_C_TPSS)
        {"XC_MGGA_C_TPSS", XC_MGGA_C_TPSS}, /*Tao, Perdew, Staroverov & Scuseria correlation */
#endif
#if defined(XC_MGGA_C_VSXC)
        {"XC_MGGA_C_VSXC", XC_MGGA_C_VSXC}, /*VSxc from Van Voorhis and Scuseria (correlation part) */
#endif
#if defined(XC_MGGA_C_M06_L)
        {"XC_MGGA_C_M06_L", XC_MGGA_C_M06_L}, /*M06-L correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M06_HF)
        {"XC_MGGA_C_M06_HF", XC_MGGA_C_M06_HF}, /* M06-HF correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M06)
        {"XC_MGGA_C_M06", XC_MGGA_C_M06}, /* M06 correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M06_2X)
        {"XC_MGGA_C_M06_2X", XC_MGGA_C_M06_2X}, /* M06-2X correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M05)
        {"XC_MGGA_C_M05", XC_MGGA_C_M05}, /*M05 correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_M05_2X)
        {"XC_MGGA_C_M05_2X", XC_MGGA_C_M05_2X}, /* M05-2X correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_C_PKZB)
        {"XC_MGGA_C_PKZB", XC_MGGA_C_PKZB}, /*Perdew, Kurth, Zupan, and Blaha */
#endif
#if defined(XC_MGGA_C_BC95)
        {"XC_MGGA_C_BC95", XC_MGGA_C_BC95}, /*Becke correlation 95 */
#endif
#if defined(XC_MGGA_C_REVTPSS)
        {"XC_MGGA_C_REVTPSS", XC_MGGA_C_REVTPSS}, /*revised TPSS correlation */
#endif
#if defined(XC_MGGA_XC_TPSSLYP1W)
        {"XC_MGGA_XC_TPSSLYP1W", XC_MGGA_XC_TPSSLYP1W}, /*Functionals fitted for water */
#endif
#if defined(XC_MGGA_X_MK00B)
        {"XC_MGGA_X_MK00B", XC_MGGA_X_MK00B}, /* Exchange for accurate virtual orbital energies (v. B) */
#endif
#if defined(XC_MGGA_X_BLOC)
        {"XC_MGGA_X_BLOC", XC_MGGA_X_BLOC}, /* functional with balanced localization */
#endif
#if defined(XC_MGGA_X_MODTPSS)
        {"XC_MGGA_X_MODTPSS", XC_MGGA_X_MODTPSS}, /* Modified Tao, Perdew, Staroverov & Scuseria exchange */
#endif
#if defined(XC_MGGA_C_TPSSLOC)
        {"XC_MGGA_C_TPSSLOC", XC_MGGA_C_TPSSLOC}, /*Semilocal dynamical correlation */
#endif
#if defined(XC_MGGA_X_MBEEF)
        {"XC_MGGA_X_MBEEF", XC_MGGA_X_MBEEF}, /*mBEEF exchange */
#endif
#if defined(XC_MGGA_X_MBEEFVDW)
        {"XC_MGGA_X_MBEEFVDW", XC_MGGA_X_MBEEFVDW}, /*mBEEF-vdW exchange */
#endif
#if defined(XC_MGGA_XC_B97M_V)
        {"XC_MGGA_XC_B97M_V", XC_MGGA_XC_B97M_V}, /*Mardirossian and Head-Gordon */
#endif
#if defined(XC_MGGA_X_MVS)
        {"XC_MGGA_X_MVS", XC_MGGA_X_MVS}, /*MVS exchange of Sun, Perdew, and Ruzsinszky */
#endif
#if defined(XC_MGGA_X_MN15_L)
        {"XC_MGGA_X_MN15_L", XC_MGGA_X_MN15_L}, /* MN15-L exhange functional from Minnesota */
#endif
#if defined(XC_MGGA_C_MN15_L)
        {"XC_MGGA_C_MN15_L", XC_MGGA_C_MN15_L}, /* MN15-L correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_X_SCAN)
        {"XC_MGGA_X_SCAN", XC_MGGA_X_SCAN}, /*SCAN exchange of Sun, Ruzsinszky, and Perdew */
#endif
#if defined(XC_MGGA_C_SCAN)
        {"XC_MGGA_C_SCAN", XC_MGGA_C_SCAN}, /*SCAN correlation */
#endif
#if defined(XC_MGGA_C_MN15)
        {"XC_MGGA_C_MN15", XC_MGGA_C_MN15}, /* MN15 correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_X_B00)
        {"XC_MGGA_X_B00", XC_MGGA_X_B00}, /* Becke 2000 */
#endif
#if defined(XC_MGGA_XC_HLE17)
        {"XC_MGGA_XC_HLE17", XC_MGGA_XC_HLE17}, /*high local exchange 2017 */
#endif
#if defined(XC_MGGA_C_SCAN_RVV10)
        {"XC_MGGA_C_SCAN_RVV10", XC_MGGA_C_SCAN_RVV10}, /* SCAN correlation + rVV10 correlation */
#endif
#if defined(XC_MGGA_X_REVM06_L)
        {"XC_MGGA_X_REVM06_L", XC_MGGA_X_REVM06_L}, /* revised M06-L exchange functional from Minnesota */
#endif
#if defined(XC_MGGA_C_REVM06_L)
        {"XC_MGGA_C_REVM06_L", XC_MGGA_C_REVM06_L}, /* Revised M06-L correlation functional from Minnesota */
#endif
#if defined(XC_MGGA_X_TM)
        {"XC_MGGA_X_TM", XC_MGGA_X_TM}, /*Tao and Mo 2016 */
#endif
#if defined(XC_MGGA_X_VT84)
        {"XC_MGGA_X_VT84", XC_MGGA_X_VT84}, /*meta-GGA version of VT{8,4} GGA */
#endif
#if defined(XC_MGGA_X_SA_TPSS)
        {"XC_MGGA_X_SA_TPSS", XC_MGGA_X_SA_TPSS}, /*TPSS with correct surface asymptotics */
#endif
#if defined(XC_MGGA_K_PC07)
        {"XC_MGGA_K_PC07", XC_MGGA_K_PC07}, /*Perdew and Constantin 2007 */
#endif
#if defined(XC_MGGA_C_KCIS)
        {"XC_MGGA_C_KCIS", XC_MGGA_C_KCIS}, /*Krieger, Chen, Iafrate, and Savin */
#endif
#if defined(XC_MGGA_XC_LP90)
        {"XC_MGGA_XC_LP90", XC_MGGA_XC_LP90}, /*Lee & Parr, Eq. (56) */
#endif
#if defined(XC_MGGA_C_B88)
        {"XC_MGGA_C_B88", XC_MGGA_C_B88}, /*Meta-GGA correlation by Becke */
#endif
#if defined(XC_MGGA_X_GX)
        {"XC_MGGA_X_GX", XC_MGGA_X_GX}, /*GX functional of Loos */
#endif
#if defined(XC_MGGA_X_PBE_GX)
        {"XC_MGGA_X_PBE_GX", XC_MGGA_X_PBE_GX}, /*PBE-GX functional of Loos */
#endif
#if defined(XC_MGGA_X_REVSCAN)
        {"XC_MGGA_X_REVSCAN", XC_MGGA_X_REVSCAN}, /* revised SCAN */
#endif
#if defined(XC_MGGA_C_REVSCAN)
        {"XC_MGGA_C_REVSCAN", XC_MGGA_C_REVSCAN}, /*revised SCAN correlation */
#endif
#if defined(XC_MGGA_C_SCAN_VV10)
        {"XC_MGGA_C_SCAN_VV10", XC_MGGA_C_SCAN_VV10}, /* SCAN correlation + VV10 correlation */
#endif
#if defined(XC_MGGA_C_REVSCAN_VV10)
        {"XC_MGGA_C_REVSCAN_VV10", XC_MGGA_C_REVSCAN_VV10}, /* revised SCAN correlation */
#endif
#if defined(XC_MGGA_X_BR89_EXPLICIT)
        {"XC_MGGA_X_BR89_EXPLICIT", XC_MGGA_X_BR89_EXPLICIT}, /*Becke-Roussel 89 with an explicit inversion of x(y) */
#endif
#if defined(XC_HYB_MGGA_X_DLDF)
        {"XC_HYB_MGGA_X_DLDF", XC_HYB_MGGA_X_DLDF}, /*Dispersionless Density Functional */
#endif
#if defined(XC_HYB_MGGA_X_MS2H)
        {"XC_HYB_MGGA_X_MS2H", XC_HYB_MGGA_X_MS2H}, /*MS2 hybrid exchange of Sun, et al */
#endif
#if defined(XC_HYB_MGGA_X_MN12_SX)
        {"XC_HYB_MGGA_X_MN12_SX", XC_HYB_MGGA_X_MN12_SX}, /*MN12-SX hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_SCAN0)
        {"XC_HYB_MGGA_X_SCAN0", XC_HYB_MGGA_X_SCAN0}, /*SCAN hybrid exchange */
#endif
#if defined(XC_HYB_MGGA_X_MN15)
        {"XC_HYB_MGGA_X_MN15", XC_HYB_MGGA_X_MN15}, /* MN15 hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_BMK)
        {"XC_HYB_MGGA_X_BMK", XC_HYB_MGGA_X_BMK}, /*Boese-Martin for kinetics */
#endif
#if defined(XC_HYB_MGGA_X_TAU_HCTH)
        {"XC_HYB_MGGA_X_TAU_HCTH", XC_HYB_MGGA_X_TAU_HCTH}, /* Hybrid version of tau-HCTH */
#endif
#if defined(XC_HYB_MGGA_X_M08_HX)
        {"XC_HYB_MGGA_X_M08_HX", XC_HYB_MGGA_X_M08_HX}, /*M08-HX exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_M08_SO)
        {"XC_HYB_MGGA_X_M08_SO", XC_HYB_MGGA_X_M08_SO}, /* M08-SO exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_M11)
        {"XC_HYB_MGGA_X_M11", XC_HYB_MGGA_X_M11}, /*M11 hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_M05)
        {"XC_HYB_MGGA_X_M05", XC_HYB_MGGA_X_M05}, /*M05 hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_M05_2X)
        {"XC_HYB_MGGA_X_M05_2X", XC_HYB_MGGA_X_M05_2X}, /* M05-2X hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_XC_B88B95)
        {"XC_HYB_MGGA_XC_B88B95", XC_HYB_MGGA_XC_B88B95}, /*Mixture of B88 with BC95 (B1B95) */
#endif
#if defined(XC_HYB_MGGA_XC_B86B95)
        {"XC_HYB_MGGA_XC_B86B95", XC_HYB_MGGA_XC_B86B95}, /* Mixture of B86 with BC95 */
#endif
#if defined(XC_HYB_MGGA_XC_PW86B95)
        {"XC_HYB_MGGA_XC_PW86B95", XC_HYB_MGGA_XC_PW86B95}, /* Mixture of PW86 with BC95 */
#endif
#if defined(XC_HYB_MGGA_XC_BB1K)
        {"XC_HYB_MGGA_XC_BB1K", XC_HYB_MGGA_XC_BB1K}, /* Mixture of B88 with BC95 from Zhao and Truhlar */
#endif
#if defined(XC_HYB_MGGA_X_M06_HF)
        {"XC_HYB_MGGA_X_M06_HF", XC_HYB_MGGA_X_M06_HF}, /*M06-HF hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_XC_MPW1B95)
        {"XC_HYB_MGGA_XC_MPW1B95", XC_HYB_MGGA_XC_MPW1B95}, /* Mixture of mPW91 with BC95 from Zhao and Truhlar */
#endif
#if defined(XC_HYB_MGGA_XC_MPWB1K)
        {"XC_HYB_MGGA_XC_MPWB1K", XC_HYB_MGGA_XC_MPWB1K}, /* Mixture of mPW91 with BC95 for kinetics */
#endif
#if defined(XC_HYB_MGGA_XC_X1B95)
        {"XC_HYB_MGGA_XC_X1B95", XC_HYB_MGGA_XC_X1B95}, /* Mixture of X with BC95 */
#endif
#if defined(XC_HYB_MGGA_XC_XB1K)
        {"XC_HYB_MGGA_XC_XB1K", XC_HYB_MGGA_XC_XB1K}, /* Mixture of X with BC95 for kinetics */
#endif
#if defined(XC_HYB_MGGA_X_M06)
        {"XC_HYB_MGGA_X_M06", XC_HYB_MGGA_X_M06}, /* M06 hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_X_M06_2X)
        {"XC_HYB_MGGA_X_M06_2X", XC_HYB_MGGA_X_M06_2X}, /* M06-2X hybrid exchange functional from Minnesota */
#endif
#if defined(XC_HYB_MGGA_XC_PW6B95)
        {"XC_HYB_MGGA_XC_PW6B95", XC_HYB_MGGA_XC_PW6B95}, /* Mixture of PW91 with BC95 from Zhao and Truhlar */
#endif
#if defined(XC_HYB_MGGA_XC_PWB6K)
        {"XC_HYB_MGGA_XC_PWB6K",
         XC_HYB_MGGA_XC_PWB6K}, /* Mixture of PW91 with BC95 from Zhao and Truhlar for kinetics */
#endif
#if defined(XC_HYB_MGGA_XC_TPSSH)
        {"XC_HYB_MGGA_XC_TPSSH", XC_HYB_MGGA_XC_TPSSH}, /*TPSS hybrid */
#endif
#if defined(XC_HYB_MGGA_XC_REVTPSSH)
        {"XC_HYB_MGGA_XC_REVTPSSH", XC_HYB_MGGA_XC_REVTPSSH}, /* revTPSS hybrid */
#endif
#if defined(XC_HYB_MGGA_X_MVSH)
        {"XC_HYB_MGGA_X_MVSH", XC_HYB_MGGA_X_MVSH}, /*MVSh hybrid */
#endif
#if defined(XC_HYB_MGGA_XC_WB97M_V)
        {"XC_HYB_MGGA_XC_WB97M_V", XC_HYB_MGGA_XC_WB97M_V}, /*Mardirossian and Head-Gordon */
#endif
#if defined(XC_HYB_MGGA_XC_B0KCIS)
        {"XC_HYB_MGGA_XC_B0KCIS", XC_HYB_MGGA_XC_B0KCIS}, /*Hybrid based on KCIS */
#endif
#if defined(XC_HYB_MGGA_XC_MPW1KCIS)
        {"XC_HYB_MGGA_XC_MPW1KCIS", XC_HYB_MGGA_XC_MPW1KCIS}, /*Modified Perdew-Wang + KCIS hybrid */
#endif
#if defined(XC_HYB_MGGA_XC_MPWKCIS1K)
        {"XC_HYB_MGGA_XC_MPWKCIS1K",
         XC_HYB_MGGA_XC_MPWKCIS1K}, /* Modified Perdew-Wang + KCIS hybrid with more exact exchange */
#endif
#if defined(XC_HYB_MGGA_XC_PBE1KCIS)
        {"XC_HYB_MGGA_XC_PBE1KCIS", XC_HYB_MGGA_XC_PBE1KCIS}, /* Perdew-Burke-Ernzerhof + KCIS hybrid */
#endif
#if defined(XC_HYB_MGGA_XC_TPSS1KCIS)
        {"XC_HYB_MGGA_XC_TPSS1KCIS", XC_HYB_MGGA_XC_TPSS1KCIS}, /* TPSS hybrid with KCIS correlation */
#endif
#if defined(XC_HYB_MGGA_X_REVSCAN0)
        {"XC_HYB_MGGA_X_REVSCAN0", XC_HYB_MGGA_X_REVSCAN0}, /* revised SCAN hybrid exchange */
#endif
#if defined(XC_HYB_MGGA_XC_B98)
        {"XC_HYB_MGGA_XC_B98", XC_HYB_MGGA_XC_B98} /*Becke 98 */
#endif
};

/// Interface class to Libxc.
class XC_functional_base
{
  protected:
    std::string libxc_name_;

    int num_spins_;

    std::unique_ptr<xc_func_type> handler_{nullptr};

    bool libxc_initialized_{false};

  private:
    /* forbid copy constructor */
    XC_functional_base(const XC_functional_base& src) = delete;

    /* forbid assignment operator */
    XC_functional_base&
    operator=(const XC_functional_base& src) = delete;

  public:
    XC_functional_base(const std::string libxc_name__, int num_spins__)
        : libxc_name_(libxc_name__)
        , num_spins_(num_spins__)
    {
        /* check if functional name is in list */
        if (libxc_functionals.count(libxc_name_) == 0 && libxc_name_ != "XC_GGA_DEBUG" &&
            libxc_name_ != "XC_LDA_DEBUG") {
            /* if not just return since van der walls functionals can be
             * used */
            libxc_initialized_ = false;
            return;
        }

        auto ns = (num_spins__ == 1) ? XC_UNPOLARIZED : XC_POLARIZED;

        if (libxc_name_ != "XC_GGA_DEBUG" && libxc_name_ != "XC_LDA_DEBUG") {
            handler_ = std::make_unique<xc_func_type>();

            /* init xc functional handler */
            if (xc_func_init(handler_.get(), libxc_functionals.at(libxc_name_), ns) != 0) {
                RTE_THROW("xc_func_init() failed");
            }
        }

        libxc_initialized_ = true;
    }

    XC_functional_base(XC_functional_base&& src__)
    {
        this->libxc_name_        = src__.libxc_name_;
        this->num_spins_         = src__.num_spins_;
        this->handler_           = std::move(src__.handler_);
        this->libxc_initialized_ = src__.libxc_initialized_;
        src__.libxc_initialized_ = false;
    }

    ~XC_functional_base()
    {
        if (handler_) {
            xc_func_end(handler_.get());
        }
    }

    const std::string
    name() const
    {
        if (handler_) {
            return std::string(handler_->info->name);
        } else {
            return libxc_name_;
        }
    }

    const std::string
    refs() const
    {
        if (handler_) {
            std::stringstream s;
            for (int i = 0; handler_->info->refs[i] != NULL; i++) {
                s << std::string(handler_->info->refs[i]->ref);
                if (strlen(handler_->info->refs[i]->doi) > 0) {
                    s << " (" << std::string(handler_->info->refs[i]->doi) << ")";
                }
                s << std::endl;
            }
            return s.str();
        } else {
            return "";
        }
    }

    int
    family() const
    {
        if (handler_) {
            return handler_->info->family;
        } else {
            if (libxc_name_ == "XC_GGA_DEBUG") {
                return XC_FAMILY_GGA;
            } else {
                return XC_FAMILY_LDA;
            }
        }
    }

    xc_func_type*
    handler()
    {
        if (handler_) {
            return handler_.get();
        }

        throw std::runtime_error("attempt to access nullptr in xc_functional_base::handler");
    }

    bool
    is_lda() const
    {
        return family() == XC_FAMILY_LDA;
    }

    bool
    is_gga() const
    {
        return family() == XC_FAMILY_GGA;
    }

    int
    kind() const
    {
        if (handler_) {
            return handler_->info->kind;
        } else {
            return XC_EXCHANGE_CORRELATION;
        }
    }

    bool
    is_exchange() const
    {
        return kind() == XC_EXCHANGE;
    }

    bool
    is_correlation() const
    {
        return kind() == XC_CORRELATION;
    }

    bool
    is_exchange_correlation() const
    {
        return kind() == XC_EXCHANGE_CORRELATION;
    }

    /// Get LDA contribution.
    void
    get_lda(const int size, const double* rho, double* v, double* e) const
    {
        if (family() != XC_FAMILY_LDA) {
            RTE_THROW("wrong XC");
        }

        /* check density */
        for (int i = 0; i < size; i++) {
            if (rho[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho[i]);
                RTE_THROW(s);
            }
        }

        if (handler_) {
            xc_lda_exc_vxc(handler_.get(), size, rho, e, v);
        } else {
            for (int i = 0; i < size; i++) {
                /* E = \int e * rho * dr */
                e[i] = -0.001 * (rho[i] * rho[i]);
                /* var E / var rho = (de/drho) rho + e */
                v[i] = -0.002 * rho[i] * rho[i] + e[i];
            }
        }
    }

    /// Get LSDA contribution.
    void
    get_lda(const int size, const double* rho_up, const double* rho_dn, double* v_up, double* v_dn, double* e) const
    {
        if (family() != XC_FAMILY_LDA) {
            RTE_THROW("wrong XC");
        }

        std::vector<double> rho_ud(size * 2);
        /* check and rearrange density */
        for (int i = 0; i < size; i++) {
            if (rho_up[i] < 0 || rho_dn[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho_up[i]) << " " << double_to_string(rho_dn[i]);
                RTE_THROW(s);
            }

            rho_ud[2 * i]     = rho_up[i];
            rho_ud[2 * i + 1] = rho_dn[i];
        }

        if (handler_) {
            std::vector<double> v_ud(size * 2);

            xc_lda_exc_vxc(handler_.get(), size, &rho_ud[0], &e[0], &v_ud[0]);

            /* extract potential */
            for (int i = 0; i < size; i++) {
                v_up[i] = v_ud[2 * i];
                v_dn[i] = v_ud[2 * i + 1];
            }
        } else {
            for (int i = 0; i < size; i++) {
                e[i]    = -0.001 * (rho_up[i] * rho_up[i] + rho_dn[i] * rho_dn[i]);
                v_up[i] = -0.002 * rho_up[i] * (rho_up[i] + rho_dn[i]) + e[i];
                v_dn[i] = -0.002 * rho_dn[i] * (rho_up[i] + rho_dn[i]) + e[i];
            }
        }
    }

    /// Get GGA contribution.
    void
    get_gga(const int size, const double* rho, const double* sigma, double* vrho, double* vsigma, double* e) const
    {
        if (family() != XC_FAMILY_GGA)
            RTE_THROW("wrong XC");

        /* check density */
        for (int i = 0; i < size; i++) {
            if (rho[i] < 0.0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho[i]);
                RTE_THROW(s);
            }
        }

        if (handler_) {
            xc_gga_exc_vxc(handler_.get(), size, rho, sigma, e, vrho, vsigma);
        } else {
            for (int i = 0; i < size; i++) {
                e[i]      = -0.001 * (rho[i] * sigma[i]);
                vrho[i]   = -0.001 * sigma[i];
                vsigma[i] = -0.001 * rho[i];
            }
        }
    }

    /// Get spin-resolved GGA contribution.
    void
    get_gga(const int size, const double* rho_up, const double* rho_dn, const double* sigma_uu, const double* sigma_ud,
            const double* sigma_dd, double* vrho_up, double* vrho_dn, double* vsigma_uu, double* vsigma_ud,
            double* vsigma_dd, double* e) const
    {
        if (family() != XC_FAMILY_GGA) {
            RTE_THROW("wrong XC");
        }

        std::vector<double> rho(2 * size);
        std::vector<double> sigma(3 * size);
        /* check and rearrange density */
        /* rearrange sigma as well */
        for (int i = 0; i < size; i++) {
            if (rho_up[i] < 0 || rho_dn[i] < 0) {
                std::stringstream s;
                s << "rho is negative : " << double_to_string(rho_up[i]) << " " << double_to_string(rho_dn[i]);
                RTE_THROW(s);
            }

            rho[2 * i]     = rho_up[i];
            rho[2 * i + 1] = rho_dn[i];

            sigma[3 * i]     = sigma_uu[i];
            sigma[3 * i + 1] = sigma_ud[i];
            sigma[3 * i + 2] = sigma_dd[i];
        }

        std::vector<double> vrho(2 * size);
        std::vector<double> vsigma(3 * size);

        if (handler_) {
            xc_gga_exc_vxc(handler_.get(), size, &rho[0], &sigma[0], e, &vrho[0], &vsigma[0]);

            /* extract vrho and vsigma */
            for (int i = 0; i < size; i++) {
                vrho_up[i] = vrho[2 * i];
                vrho_dn[i] = vrho[2 * i + 1];

                vsigma_uu[i] = vsigma[3 * i];
                vsigma_ud[i] = vsigma[3 * i + 1];
                vsigma_dd[i] = vsigma[3 * i + 2];
            }
        } else {
            auto h1 = std::make_unique<xc_func_type>();

            /* init xc functional handler */
            if (xc_func_init(h1.get(), XC_LDA_C_PZ, 2) != 0) {
                RTE_THROW("xc_func_init() failed");
            }

            xc_lda_exc_vxc(h1.get(), size, &rho[0], e, &vrho[0]);
            /* extract vrho and vsigma */
            for (int i = 0; i < size; i++) {
                vrho_up[i] = vrho[2 * i];
                vrho_dn[i] = vrho[2 * i + 1];

                vsigma_uu[i] = 0;
                vsigma_ud[i] = 0;
                vsigma_dd[i] = 0;
            }

            for (int i = 0; i < size; i++) {
                e[i] += 0.001 * (sigma_uu[i] + sigma_ud[i] + sigma_dd[i]);
                vsigma_uu[i] = -0.001;
                vsigma_ud[i] = -0.001;
                vsigma_dd[i] = -0.001;
            }
        }
    }

    /// set density threshold of libxc, if density is below tre, all xc output will be set to 0.
    void
    set_dens_threshold(double tre)
    {
#if XC_MAJOR_VERSION >= 4
        xc_func_set_dens_threshold(this->handler(), tre);
#else
        std::cout << "set_dens_threshold not available in old libxc versions, install at least 4.2.3"
                  << "\n";
#endif
    }
};
} // namespace sirius
#endif // __XC_FUNCTIONAL_BASE_H__
