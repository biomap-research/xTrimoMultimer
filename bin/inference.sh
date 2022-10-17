#!/bin/bash

python inference.py data/pdb_mmcif/mmcif_files/ \
    --fasta_paths target.fasta \
    --model_preset multimer \
    --output_dir ./tmp/ \
    --param_dir ./data/params/ \
    --gpu_nums 1 \
    --cpus 12 \
    --use_fastfold_optimize \
    --use_precomputed_alignments ./tmp/alignments/ \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniclust30_database_path data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --pdb_seqres_database_path data/pdb_seqres/pdb_seqres.txt \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --jackhmmer_binary_path `which jackhmmer` \
    --hhblits_binary_path `which hhblits` \
    --hhsearch_binary_path `which hhsearch` \
    --kalign_binary_path `which kalign`
