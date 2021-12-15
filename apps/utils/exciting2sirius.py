#!/usr/bin/env python3

from typing import List
from decimal import Decimal
import xml.etree.ElementTree as ET
import pathlib
import json
import argparse
import subprocess
import tempfile
import sys
from typing import IO, Union, Optional

import pydantic


class BaseModel(pydantic.BaseModel):
    class Config:
        extra = "forbid"


class MuffinTin(BaseModel):
    rmin: Decimal
    radius: Decimal
    rinf: Decimal
    radialmeshPoints: pydantic.PositiveInt


class AtomicState(BaseModel):
    n: int
    l: int
    kappa: int
    occ: Decimal
    core: bool


class BasisDefault(BaseModel):
    btype: str = pydantic.Field("lapw", alias="type")
    trialEnergy: Decimal
    searchE: bool

    def to_sirius_dict(self):
        assert self.btype == "lapw", "Only LAPW basis supported for default"
        return {
            "basis": [
                {"auto": int(self.searchE), "dme": dme, "enu": float(self.trialEnergy)}
                for dme in range(2)
            ]
        }


class BasisCustom(BaseModel):
    btype: str = pydantic.Field(
        "apw+lo", alias="type"
    )  # species.xsd has a different default here than for 'default'
    trialEnergy: Decimal
    searchE: bool
    l: int

    def to_sirius_dict(self, n: int) -> dict:
        assert self.btype == "lapw", "Only LAPW basis supported for custom"
        assert n >= (self.l + 1), "n quantum number must be larger than l"
        return {
            "basis": [
                {"auto": int(self.searchE), "dme": dme, "enu": float(self.trialEnergy)}
                for dme in range(2)
            ],
            "l": self.l,
            "n": n,
        }


class BasisWF(BaseModel):
    matchingOrder: int
    trialEnergy: Decimal
    searchE: bool


class BasisLO(BaseModel):
    l: int
    wf: List[BasisWF]
    btype: str = pydantic.Field("lapw", alias="type")

    def to_sirius_dict(self):
        assert all(
            wf.searchE == False for wf in self.wf
        ), "local-orbitals with searchE enabled are currently not supported"
        assert self.btype == "lapw"

        return {
            "basis": [
                {
                    "auto": int(wf.searchE),
                    "dme": wf.matchingOrder,
                    "enu": float(wf.trialEnergy),
                    "n": self.l + 1,
                }
                for wf in self.wf
            ],
            "l": self.l,
        }


class Basis(BaseModel):
    default: BasisDefault
    custom: List[BasisCustom]
    lo: List[BasisLO]

    def to_sirius_dict(self):
        if self.custom:
            max_l = max(c.l for c in self.custom)
            custom = [c.to_sirius_dict(max_l + 1) for c in self.custom]
        else:
            custom = []

        return {
            "lo": [lo.to_sirius_dict() for lo in self.lo],
            "valence": [self.default.to_sirius_dict(), *custom],
        }


L2ORB = ["s", "p", "d", "f"]


class Species(BaseModel):
    muffinTin: MuffinTin
    atomicState: List[AtomicState]
    basis: Basis
    chemicalSymbol: str
    name: str
    z: Decimal
    mass: Decimal

    def to_sirius_dict(self):
        return {
            "core": "".join(
                f"{st.n}{L2ORB[st.l]}" for st in self.atomicState if st.core == True
            ),
            "free_atom": {
                "density": [],
                "radial_grid": [],
            },
            **self.basis.to_sirius_dict(),
            "mass": float(self.mass) / 1.82288848426455e03,  # [u] to [a.u.]
            "name": self.name,
            "nrmt": self.muffinTin.radialmeshPoints,
            "number": -int(self.z.to_integral_exact()),
            "rinf": float(self.muffinTin.rinf),
            "rmin": float(self.muffinTin.rmin),
            "rmt": float(self.muffinTin.radius),
            "symbol": self.chemicalSymbol,
        }


def parse_exciting_species(fhandle: IO[str]) -> Species:
    tree = ET.parse(fhandle)
    xmlsp = tree.getroot()[0]

    assert (
        xmlsp.tag == "sp"
    ), "Did not find species, are you sure this is an Exciting species file?"
    xmlbasis = xmlsp.find("basis")

    return Species(
        muffinTin=MuffinTin(**xmlsp.find("muffinTin").attrib),
        atomicState=[
            AtomicState(**state.attrib) for state in xmlsp.findall("atomicState")
        ],
        basis=Basis(
            lo=[
                BasisLO(
                    wf=[BasisWF(**wf.attrib) for wf in lo.findall("wf")], **lo.attrib
                )
                for lo in xmlbasis.findall("lo")
            ],
            default=BasisDefault(**xmlbasis.find("default").attrib),
            custom=[BasisCustom(**c.attrib) for c in xmlbasis.findall("custom")],
        ),
        **xmlsp.attrib,
    )


def gen_sirius_species(symbol: str, core: Optional[Union[float, Decimal]]):
    atom_args = ["atom", f"--symbol={symbol}"]

    if core:
        atom_args.append(f"--core={core}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        subprocess.run(atom_args, cwd=tmpdirname, check=True, capture_output=True)
        with pathlib.Path(tmpdirname).joinpath(f"{symbol}.json").open() as fhandle:
            return json.load(fhandle)


def main():
    parser = argparse.ArgumentParser(
        description="Convert an Exciting to a SIRIUS species. The 'atom' executable has to be available and located in a path in $PATH."
    )
    parser.add_argument("exciting_species_file", type=argparse.FileType("r"))
    parser.add_argument(
        "--core",
        type=str,
        help=(
            "The RMT for the free atom (default is to take it from the Exciting species)."
            " Use 'SIRIUS' to take the SIRIUS default value instead, or specify a numerical value."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=argparse.FileType("w"), default=sys.stdout
    )
    args = parser.parse_args()

    print("Parsing Exciting species file...", file=sys.stderr)
    exciting = parse_exciting_species(args.exciting_species_file)
    rmt = exciting.muffinTin.radius
    print(f"Found a species for '{exciting.chemicalSymbol}', and rmt={rmt}", file=sys.stderr)

    print("Converting Exciting structure to SIRIUS...", file=sys.stderr)
    sirius = exciting.to_sirius_dict()

    if args.core:
        if args.core == "SIRIUS":
            rmt = None
        else:
            rmt = float(args.core)

    try:
        print("Generating temporary SIRIUS species file...", file=sys.stderr)
        sirius_temp = gen_sirius_species(exciting.chemicalSymbol, rmt)
    except subprocess.CalledProcessError as exc:
        print("ERROR: calling SIRIUS' atom command failed", file=sys.stderr)
        print(exc.stdout)
        print(exc.stderr, file=sys.stderr)

    # fill-in the required values, overwrite others to match the free_atom data:
    sirius["free_atom"] = sirius_temp["free_atom"]
    sirius["nrmt"] = sirius_temp["nrmt"]
    sirius["rinf"] = sirius_temp["rinf"]
    sirius["rmin"] = sirius_temp["rmin"]

    json.dump(sirius, args.output, indent=4)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
