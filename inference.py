# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
# Copyright 2022 HPC-AI Technology Inc.
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import sys
import os
import time
import argparse
import glob
import pickle
import pathlib
import torch
import numpy as np
import torch.multiprocessing as mp

from datetime import date

env_path = "%s/../" % os.path.dirname(os.path.abspath(__file__))
sys.path.append(env_path)

import xtrimomultimer.np.relax.relax as relax

from xtrimomultimer.config import model_config
from xtrimomultimer.data import (
    templates,
    feature_pipeline,
    data_pipeline as pipeline,
    data_pipeline_multimer as multimer_pipeline,
)
from xtrimomultimer.data.tools import hhsearch, hmmsearch
from xtrimomultimer.model_acc.inject_fastnn import inject_fastnn
from xtrimomultimer.utils.io import parse_fasta, save_feature, save_ranked_results
from xtrimomultimer.utils.seed import set_device_seed
from xtrimomultimer.model.model import AlphaFold
from xtrimomultimer.np import residue_constants, protein
from xtrimomultimer.utils.import_weights import import_jax_weights_
from xtrimomultimer.utils.tensor_utils import tensor_tree_map
from xtrimomultimer.utils.general_utils import warmup_gpu_for_pytorch
from xtrimomultimer.model_acc.distributed import init_dap
from xtrimomultimer.model_acc import set_chunk_size

from xtrimomultimer.utils.logger import Logger

logger = Logger.logger

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        if args.use_precomputed_alignments is None and not os.path.isdir(
            local_alignment_dir
        ):
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir)

            alignment_runner = pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                hhsearch_binary_path=args.hhsearch_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                pdb70_database_path=args.pdb70_database_path,
                no_cpus=args.cpus,
            )
            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        else:
            logger.info(f"Using precomputed alignments for {tag} at {alignment_dir}...")

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def add_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--uniref90_database_path", type=str, default=None)
    parser.add_argument("--mgnify_database_path", type=str, default=None)
    parser.add_argument("--pdb70_database_path", type=str, default=None)
    parser.add_argument("--uniclust30_database_path", type=str, default=None)
    parser.add_argument("--bfd_database_path", type=str, default=None)
    parser.add_argument(
        "--pdb_seqres_database_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--uniprot_database_path",
        type=str,
        default=None,
    )
    parser.add_argument("--jackhmmer_binary_path", type=str, default="jackhmmer")
    parser.add_argument("--hhblits_binary_path", type=str, default="hhblits")
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument(
        "--max_template_date", type=str, default=date.today().strftime("%Y-%m-%d")
    )
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_paths",
        type=str,
        default=None,
        help="Paths to FASTA files, could be a directory or a fasta file path or "
        "a list of fasta file splitting by comma",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to a directory that will store the results.",
    )
    parser.add_argument(
        "--model_preset",
        type=str,
        default="monomer",
        choices=["monomer", "multimer"],
        help="Choose preset model configuration - the monomer model, the monomer model with "
        "extra ensembling, monomer model with pTM head, or multimer model",
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation
                    is skipped and database path arguments are ignored.""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_1,model_2,model_3,model_4,model_5",
        choices=[
            "model_1,model_2,model_3,model_4,model_5",
            "model_1_multimer,model_2_multimer,model_3_multimer,model_4_multimer,model_5_multimer",
        ],
        help="Name of a model config. Choose one of model_{1-5} or model_{1-5}_ptm.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="""Number of CPUs with which to run alignment tools""",
    )
    parser.add_argument(
        "--gpu_nums", type=int, default=1, help="number of NVIDIA GPU cuda use"
    )
    parser.add_argument(
        "--preset", type=str, default="full_dbs", choices=("reduced_dbs", "full_dbs")
    )

    parser.add_argument(
        "--release_dates_path", type=str, default=None, help="pdb release dates path"
    )  # todo, not test
    parser.add_argument("template_mmcif_dir", type=str)

    # msa searcher tools and database
    parser.add_argument(
        "--param_dir",
        type=str,
        help="Path to param database",
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default="kalign", help="Kalign command path"
    )
    parser.add_argument(
        "--hhsearch_binary_path", type=str, default="hhsearch", help="path to hhsearch"
    )

    # other params
    parser.add_argument(
        "--warmup",
        default=False,
        action="store_true",
        help="develop or product version",
    )

    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--cuda_for_amber", default=False, action="store_true")
    parser.add_argument("--seed", help="Random generator seed.", type=int, default=0)
    parser.add_argument(
        "--recompute",
        default=False,
        action="store_true",
        help="whether recompute for existed results",
    )
    parser.add_argument(
        "--skip_amber_relax",
        default=False,
        action="store_true",
        help="whether skip amber relax",
    )
    parser.add_argument(
        "--reproduce",
        default=False,
        action="store_true",
        help="reproduce all the results",
    )
    parser.add_argument(
        "--skip_template",
        default=False,
        action="store_true",
        help="whether skip template",
    )
    parser.add_argument(
        "--use_fastfold_optimize",
        default=False,
        action="store_true",
        help="enable fastfold optimize kernel",
    )
    parser.add_argument(
        "--center_atom_positions",
        default=False,
        action="store_true",
        help="center atom positions",
    )

    """
    R: number of residues
    A: number of atoms, A=37
    O: number of aminod acid, O=21
    M: number of msa
    D1, D2, D3: hidden embedding size for evoformer, pair and structure module
    msa, shape = (M, R, D1), D1 = 256, D2 = 128, D3 = 384
    msa_first_row, shape = (r, D1)
    pair, shape = (R, R, D2)
    single, shape = (R, D3)
    structure_module, shape = (R, D3)
    """
    parser.add_argument(
        "--saved_emds",
        type=str,
        default="msa_first_row,single,structure_module",
        help="saved embedding lists: msa_first_row,single,structure_module,msa,pair; "
        "set saved_emds=None to skip saving embeddings",
    )

    add_data_args(parser)

    args = parser.parse_args()
    if args.saved_emds.lower() == "none":
        args.saved_emds = list()
    else:
        saved_emds = list()
        for value in args.saved_emds.strip().split(","):
            if value.strip() in ("msa_first_row", "structure_module"):
                logger.warning(
                    f"saved_emds current version does NOT support {value.strip()}."
                )
            else:
                saved_emds.append(value.strip())
        args.saved_emds = saved_emds

    if (args.model_preset == "monomer" and "multimer" in args.model_name) or (args.model_preset == "multimer" 
        and any(["multimer" not in name for name in args.model_name.split(',')])):
        raise ValueError(f"{args.model_preset} is set as preset but model_name contains model from another preset")
    elif args.model_preset not in ["monomer", "multimer"]:
        logger.error(f"unknown model_preset={args.model_preset}")
        raise NameError

    return args


def check_inference_result_exits(output_dir, num_model):
    """check whether there exists the expected results"""
    features_path = os.path.join(output_dir, f"features.pkl")
    if not os.path.isfile(features_path):
        return False

    for idx in range(num_model):
        ranked_unrelax_output_path = os.path.join(
            output_dir, f"ranked_unrelax_{idx}.pdb"
        )

        if os.path.isfile(ranked_unrelax_output_path):
            pass
        else:
            return False
    return True


def get_fasta_file_list(fasta_paths: str, sp: str = ","):
    if fasta_paths.endswith(".fasta"):
        fasta_list = [file_path.strip() for file_path in fasta_paths.split(sp)]
    else:
        fasta_list = list(glob.iglob(fasta_paths + "**/*.fasta", recursive=True))
    if len(fasta_list) != len(set(fasta_list)):
        raise ValueError("All FASTA paths must have a unique basename.")

    # TODO, specify eukaryotic or prokaryotic for each fasta in the future
    # These values determine the pairing method for the MSA.
    is_prokaryote_list = [False] * len(fasta_list)
    logger.warning(
        "set default is_prokaryote_list=[False]. "
        "Specify eukaryotic or prokaryotic for each fasta in the future"
    )

    return fasta_list, is_prokaryote_list


def predict_structure(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    tmp_file: str,
    model_name: str,
    fasta_path: str,
    processed_feature_dict: pipeline.FeatureDict,
    is_fastfold_optimize: bool,
):

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # init distributed for Dynamic Axial Parallelism

    init_dap()
    if world_size > 1 and not is_fastfold_optimize:
        logger.error("Cannot use multiGPU without enable FASTFOLD_OPTIMIZE")
        assert False, "Multithread without optimize is useless"

    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    param_path = f"{args.param_dir}/params_{model_name}.npz"
    # make config according to specified model
    config = model_config(model_name)
    model = AlphaFold(config)
    import_jax_weights_(model, param_path, version=model_name)
    if is_fastfold_optimize:
        inject_fastnn(model)

    model = model.eval()
    model = model.to(device)

    fasta_name = pathlib.Path(fasta_path).stem
    output_dir = os.path.join(args.output_dir, fasta_name)
    os.makedirs(output_dir, exist_ok=True)

    if is_fastfold_optimize:
        model.globals.chunk_size = 64
        set_chunk_size(64)

    batch = processed_feature_dict
    with torch.no_grad():
        batch = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        torch.cuda.synchronize()
        t = time.perf_counter()
        output_dict = model(batch)
        torch.cuda.synchronize()
        logger.info(f"Inference time: {time.perf_counter() - t}")

    output_dict = tensor_tree_map(lambda x: np.array(x.cpu()), output_dict)
    if rank == 0:
        pickle.dump(output_dict, open(tmp_file, "wb"))
    torch.cuda.synchronize()
    torch.distributed.barrier()


def main(args):
    fasta_list, is_prokaryote_list = get_fasta_file_list(args.fasta_paths)
    # we use model class to sepicfy monomer or multimer
    global_is_multimer = True if args.model_preset == "multimer" else False
    model_names = [model_name.strip() for model_name in args.model_name.split(",")]
    num_model = len(model_names)

    if args.skip_template:
        template_featurizer = None

    predict_max_templates = 4

    if global_is_multimer:
        if not args.use_precomputed_alignments:
            template_searcher = hmmsearch.Hmmsearch(
                binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                database_path=args.pdb_seqres_database_path,
            )
        else:
            template_searcher = None

        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=predict_max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )
    else:
        if not args.use_precomputed_alignments:
            template_searcher = hhsearch.HHSearch(
                binary_path=args.hhsearch_binary_path,
                databases=[args.pdb70_database_path],
            )
        else:
            template_searcher = None

        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=predict_max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )

    if not args.use_precomputed_alignments:
        alignment_runner = pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            uniprot_database_path=args.uniprot_database_path,
            template_searcher=template_searcher,
            use_small_bfd=(args.bfd_database_path is None),
            no_cpus=args.cpus,
        )
    else:
        alignment_runner = None

    data_processor = pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    if global_is_multimer:
        data_processor = multimer_pipeline.DataPipeline(
            monomer_data_pipeline=data_processor,
        )

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if not args.use_precomputed_alignments:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    logger.info(f"total number of fasta: {len(fasta_list)}")
    success = 0
    for fasta_path in fasta_list:
        if args.reproduce:
            _ = set_device_seed(args)  # reset seed for reproduce

        ranking_confidences = {}
        unrelaxed_pdbs = {}
        amber_outputs = {}
        predicted_results = {}
        amber_status = [False if args.skip_amber_relax else True]
        fasta_name = pathlib.Path(fasta_path).stem
        output_dir = os.path.join(args.output_dir, fasta_name)

        # Check whether the result has been computed
        if (
            check_inference_result_exits(output_dir, num_model=num_model)
            and not args.recompute
        ):
            logger.info(f"expected results exist in {output_dir}")
            success += 1
            continue
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Gather input sequences
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)
        if (not global_is_multimer) and len(tags) != 1:
            print(
                f"{fasta_path} contains more than one sequence but "
                f"multimer mode is not enabled. Skipping..."
            )
            continue

        for tag, seq in zip(tags, seqs):
            local_alignment_dir = os.path.join(alignment_dir, tag)
            if args.use_precomputed_alignments is None:
                if not os.path.exists(local_alignment_dir):
                    os.makedirs(local_alignment_dir)

                chain_fasta_str = f">chain_{tag}\n{seq}\n"
                with multimer_pipeline.temp_fasta_file(
                    chain_fasta_str
                ) as chain_fasta_path:
                    alignment_runner.run(chain_fasta_path, local_alignment_dir)

        if global_is_multimer:
            local_alignment_dir = alignment_dir
        else:
            local_alignment_dir = os.path.join(
                alignment_dir,
                tags[0],
            )

        if os.path.exists(os.path.join(output_dir, "features.pkl")) and not args.recompute:
            feature_dict = pickle.load(
                open(os.path.join(output_dir, "features.pkl"), "rb")
            )
        else:
            feature_dict = data_processor.process_fasta(
                fasta_path=fasta_path, alignment_dir=local_alignment_dir
            )
            save_feature(feature_dict, fasta_path, output_dir)

        for model_name in model_names:
            model_name = model_name.strip()

            # second stage feature extraction
            logger.info(f"process features {model_name} on {fasta_path}")
            processed_feature_dict = dict()
            feature_processor = feature_pipeline.FeaturePipeline(
                model_config(model_name).data
            )
            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode="predict",
                is_multimer=global_is_multimer,
            )
            save_feature(
                processed_feature_dict, fasta_path, args.output_dir, postfix="processed"
            )

            tmp_file = tempfile.NamedTemporaryFile()
            logger.info(f"inference model {model_name} on {fasta_path}")
            mp.spawn(
                predict_structure,
                nprocs=args.gpu_nums,
                args=(
                    args.gpu_nums,
                    args,
                    tmp_file.name,
                    model_name,
                    fasta_path,
                    processed_feature_dict,
                    args.use_fastfold_optimize,
                ),
            )

            output_dict = pickle.load(open(tmp_file.name, "rb"))
            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
            )

            plddt = output_dict["plddt"]
            ranking_confidence = np.mean(
                plddt
            )  # monomer used mean_plddt as confidence ranking
            plddt_b_factors = np.repeat(
                plddt[..., None], residue_constants.atom_type_num, axis=-1
            )
            unrelaxed_protein = protein.from_prediction(
                features=batch, result=output_dict, b_factors=plddt_b_factors
            )
            unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)

            amber_output = {
                "relaxed_pdb": None,
                "debug_data": None,
                "violations": None,
            }

            this_amber_status = False
            if amber_status[-1]:
                # run amber relax if previous trail is successful.
                try:
                    config = model_config(model_name)
                    amber_relaxer = relax.AmberRelaxation(**config.relax)
                    relaxed_pdb, debug_data, violations = amber_relaxer.process(
                        prot=unrelaxed_protein
                    )
                    amber_output["relaxed_pdb"] = relaxed_pdb
                    amber_output["debug_data"] = debug_data
                    amber_output["violations"] = violations
                    this_amber_status = True
                except Exception as e:
                    logger.exception("Amber relax failed.")

            ranking_confidences[model_name] = ranking_confidence
            unrelaxed_pdbs[model_name] = unrelaxed_pdb
            amber_outputs[model_name] = amber_output
            predicted_results[model_name] = output_dict
            # if one instance is failed, set saved_amber = False
            amber_status.append(this_amber_status)

        is_save_amber = all(amber_status)
        save_ranked_results(
            fasta_path,
            ranking_confidences,
            unrelaxed_pdbs,
            amber_outputs,
            predicted_results,
            args.output_dir,
            is_save_amber,
            args.saved_emds,
        )
        success += 1

    logger.info(f"tot={len(fasta_list)}, success={success}")


if __name__ == "__main__":
    args = get_args()
    device = set_device_seed(args)
    if args.reproduce:
        # warm-up gpu device, measure gpu running time correctly.
        warmup_gpu_for_pytorch(device)
    main(args)
    logger.info("End.")
