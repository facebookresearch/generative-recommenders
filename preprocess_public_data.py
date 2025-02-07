# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

"""
Usage: mkdir -p tmp/ && python3 preprocess_public_data.py
"""

from generative_recommenders.research.data.preprocessor import get_common_preprocessors


def main() -> None:
    get_common_preprocessors()["ml-1m"].preprocess_rating()
    get_common_preprocessors()["ml-20m"].preprocess_rating()
    # get_common_preprocessors()["ml-1b"].preprocess_rating()
    get_common_preprocessors()["amzn-books"].preprocess_rating()


if __name__ == "__main__":
    main()
