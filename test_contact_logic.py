import tempfile
import unittest
from pathlib import Path

from analyze_nanobody_epitope import (
    DEFAULT_CONFIG,
    NumberingMapper,
    analyze_contacts,
    parse_structure,
    select_cdr_residues,
    select_chain_residues,
)


MINI_PDB = """\
ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ASP A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ASP A   1       2.000  -1.400   0.000  1.00 20.00           C
ATOM      4  O   ASP A   1       1.300  -2.400   0.000  1.00 20.00           O
ATOM      5  CB  ASP A   1       2.000   1.300   0.000  1.00 20.00           C
ATOM      6  CG  ASP A   1       3.480   1.300   0.000  1.00 20.00           C
ATOM      7  OD1 ASP A   1       4.300   2.200   0.000  1.00 20.00           O
ATOM      8  OD2 ASP A   1       3.900   0.100   0.000  1.00 20.00           O
ATOM      9  N   LYS C  30       5.800  -1.000   0.000  1.00 20.00           N
ATOM     10  CA  LYS C  30       5.800   0.300   0.000  1.00 20.00           C
ATOM     11  C   LYS C  30       6.900   1.100   0.000  1.00 20.00           C
ATOM     12  O   LYS C  30       7.900   0.500   0.000  1.00 20.00           O
ATOM     13  CB  LYS C  30       4.500   1.100   0.000  1.00 20.00           C
ATOM     14  CG  LYS C  30       4.700   2.600   0.000  1.00 20.00           C
ATOM     15  CD  LYS C  30       3.500   3.400   0.000  1.00 20.00           C
ATOM     16  CE  LYS C  30       3.700   4.900   0.000  1.00 20.00           C
ATOM     17  NZ  LYS C  30       4.700   5.700   0.000  1.00 20.00           N
TER
END
"""


class ContactLogicTest(unittest.TestCase):
    def test_basic_contact_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "mini.pdb"
            pdb_path.write_text(MINI_PDB, encoding="utf-8")

            model = parse_structure(str(pdb_path))
            antigen_residues, _ = select_chain_residues(model, ["A"])
            cdr_residues, _ = select_cdr_residues(model, "C", {"cdr1": (30, 30), "cdr2": (31, 31), "cdr3": (32, 32)})

            warnings = []
            mapper = NumberingMapper({}, ["A"], warnings)
            records = analyze_contacts(
                antigen_residues=antigen_residues,
                cdr_residues=cdr_residues,
                cutoffs=DEFAULT_CONFIG["contact_cutoffs"],
                mapper=mapper,
                warnings=warnings,
            )

            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertGreater(rec["total_contact_count"], 0)
            self.assertGreater(rec["min_heavy_atom_distance"], 0.0)
            self.assertGreater(rec["salt_bridge_count"], 0)


if __name__ == "__main__":
    unittest.main()
