AF2_NICKNAME = ["af2", "alphafold2", "monomer"]  # The nickname of AF2

BASE_SEQ2PATH_NAME = (
    "msa_seq2path"  # The mapping file's name. Use it to create the mapping file.
)

BASE_FOLDER_NAME = "MSAs"  # project file's name, Use it to create a directory.

JACKHMMER_UNIREF90_FILE_NAME = "jackhmmer_uniref90_hits.sto"

JACKHMMER_MGNIFY_FILE_NAME = "jackhmmer_mgnify_hits.sto"

HHBLITS_BFD_UNIClUST30_FILE_NAME = "hhblits_bfd_uniclust30_hits.a3m"

MULTIMER_NICKNAME = ["multimer", "alphafold2_multimer", "alphafold2-multimer"]

MULTIMER_BASE_FOLDER_NAME = "multimer"

JACKHMMER_UNIPROT_FILE_NAME = "jackhmmer_uniprot_hits.sto"

HMMSEARCH_PDB_SEQRES_FILE_NAME = "hmmsearch_pdb_seqres_hits.sto"

HHSEARCH_PDB70_FILE_NAME = "hhsearch_pdb70_hits.hhr"

HHBLITS_BFD_FILE_NAME = "hhblits_bfd_hits.a3m"

HHBLITS_UNICLUST30_FILE_NAME = "hhblits_uniclust30_hits.a3m"

MMDB_UNIREF30_FILE_NAME = "mmdb_uniref30_hits.a3m"

MMDB_ENVDB_FILE_NAME = "mmdb_envdb_hits.a3m"

MMDB_PDB70_FILE_NAME = "mmdb_pdb70_hits.a8"

MMSEQS_JHU_NICKNAME = ["mmseqs_jackhmmer_uniprot", "mmseqs_jhu"]

MMSEQS_JHU_BASE_FOLDER_NAME = "mmseqs_jackhmmer_uniprot"

MSA_COMPONENTS = {
    "AF2_0df8042a5fe9a829368522d695328450": [
        JACKHMMER_UNIREF90_FILE_NAME,
        JACKHMMER_MGNIFY_FILE_NAME,
        HHBLITS_BFD_FILE_NAME,
        HHBLITS_BFD_UNIClUST30_FILE_NAME,
        HHBLITS_UNICLUST30_FILE_NAME,
        HHSEARCH_PDB70_FILE_NAME,
    ],
    "multimer_f89063cbe31815418d57ceaba36caeb0": [
        JACKHMMER_UNIREF90_FILE_NAME,
        JACKHMMER_MGNIFY_FILE_NAME,
        HHBLITS_BFD_UNIClUST30_FILE_NAME,
        JACKHMMER_UNIPROT_FILE_NAME,
        HMMSEARCH_PDB_SEQRES_FILE_NAME,
    ],
    "mmseqs_jackhmmer_uniprot_886cff956097ef901040c3a6bbc68942": [
        MMDB_UNIREF30_FILE_NAME,
        MMDB_ENVDB_FILE_NAME,
        MMDB_PDB70_FILE_NAME,
        JACKHMMER_UNIPROT_FILE_NAME,
    ],
}
