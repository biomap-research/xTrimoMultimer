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


""" Unifies databases created with create_alignment_db.py """


def main(args):
    super_index = {}
    for f in os.listdir(args.alignment_db_dir):
        if not os.path.splitext(f)[-1] == ".index":
            continue

        with open(os.path.join(args.alignment_db_dir, f), "r") as fp:
            index = json.load(fp)

        db_name = f"{os.path.splitext(f)[0]}.db"

        for k in index:
            super_index[k] = {
                "db": db_name,
                "files": index[k],
            }

    with open(os.path.join(args.output_dir, "super.index"), "w") as fp:
        json.dump(super_index, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alignment_db_dir", type=str, help="Path to directory containing alignment_dbs"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path in which to output super index"
    )

    args = parser.parse_args()

    main(args)
