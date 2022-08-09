# Copyright 2022 BioMap (Beijing) Intelligence Technology Limited
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

import json
import os
import re
import pathlib
import pickle
from typing import Dict, List, Tuple, Union
from xtrimomultimer.data.data_pipeline import FeatureDict

from xtrimomultimer.utils.logger import Logger

logger = Logger.logger


def parse_fasta(data: str) -> Tuple[List[str], List[str]]:
    """function to parse fasta file

    Args:
        data (str): fasta file content

    Returns:
        Tuple[List[str], List[str]]:
            Tuple of tags list and sequence list parsed from fasta file
    """
    data = re.sub(">$", "", data, flags=re.M)
    lines = [
        l.replace("\n", "")
        for prot in data.split(">")
        for l in prot.strip().split("\n", 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def save_feature(
    feature_dict: FeatureDict,
    fasta_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    postfix: str = "",
):
    """function for saving feature dict

    Args
        feature_dict (FeatureDict): Feature diction with format: {feature_name: feature_value}
        fasta_path (Union[str, os.PathLike]): path to corresponding fasta that generate the feature
        output_dir (Union[str, os.PathLike]): path to base directory to save
        postfix (str, optional): postfix of saved feature file. Defaults to "".
    """
    fasta_name = pathlib.Path(fasta_path).stem
    logger.info("predicting structure for %s", fasta_name)
    output_dir = os.path.join(output_dir, fasta_name)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"features.pkl" if postfix == "" else f"features_{postfix}.pkl"
    feature_path = os.path.join(output_dir, filename)
    with open(feature_path, "wb") as f:
        pickle.dump(feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_ranked_results(
    fasta_path: Union[str, os.PathLike],
    ranking_confidences: Dict,
    unrelaxed_pdbs: Dict,
    amber_outputs: Dict,
    predicted_results: Dict,
    output_dir: Union[str, os.PathLike],
    saved_amber: bool = False,
    saved_embeddings: bool = False,
):
    """
    save model by its ranking confidence
    """
    fasta_name = pathlib.Path(fasta_path).stem
    logger.info("predicting structure for %s", fasta_name)
    output_dir = os.path.join(output_dir, fasta_name)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (model_name, _) in enumerate(
        sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)
    ):
        # always save ranked_unrelax results
        if len(saved_embeddings) > 0:
            # Save the model outputs.
            result_output_path = os.path.join(output_dir, f"emds_{idx}.pkl")
            with open(result_output_path, "wb") as f:
                saved_emd_results = dict()
                # only save embedding of interest
                for key in saved_embeddings:
                    key = key.strip()
                    saved_emd_results[key] = predicted_results[model_name][key]
                pickle.dump(saved_emd_results, f)

        ranked_output_path = os.path.join(output_dir, f"ranked_unrelax_{idx}.pdb")
        with open(ranked_output_path, "w") as f:
            f.write(unrelaxed_pdbs[model_name])

        if saved_amber:
            ranked_output_path = os.path.join(output_dir, f"ranked_relax_{idx}.pdb")
            with open(ranked_output_path, "w") as f:
                f.write(amber_outputs[model_name]["relaxed_pdb"])
            amber_output_path = os.path.join(output_dir, f"amber_output_{idx}.json")
            with open(amber_output_path, "w") as f:
                f.write(json.dumps(amber_outputs[model_name]["debug_data"], indent=4))
