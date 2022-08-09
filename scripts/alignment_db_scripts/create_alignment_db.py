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

import argparse
import json
import os


def main(args):
    db_path = os.path.join(args.output_db_path, f"{args.output_db_name}.db")
    index_path = os.path.join(args.output_db_path, f"{args.output_db_name}.index")
    db_fp = open(db_path, "wb")
    index = {}
    db_offset = 0
    for chain_alignment_dir in os.listdir(args.alignment_dir):
        cad_path = os.path.join(args.alignment_dir, chain_alignment_dir)
        for f in os.listdir(cad_path):
            f_path = os.path.join(cad_path, f)
            with open(f_path, "rb") as fp:
                file_bytes = fp.read()

            l = len(file_bytes)
            file_list = index.setdefault(chain_alignment_dir, [])
            file_list.append((f, db_offset, l))

            db_fp.write(file_bytes)
            db_offset += l

    db_fp.close()

    with open(index_path, "w") as fp:
        json.dump(index, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alignment_dir",
        type=str,
        help="""Path to precomputed alignment directory, with one subdirectory
                per chain.""",
    )
    parser.add_argument("output_db_path", type=str)
    parser.add_argument("output_db_name", type=str)

    args = parser.parse_args()

    main(args)
