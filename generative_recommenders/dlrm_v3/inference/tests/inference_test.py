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

import os
import unittest

from generative_recommenders.common import gpu_unavailable
from generative_recommenders.dlrm_v3.inference.main import main
from hypothesis import given, settings, strategies as st, Verbosity


class DLRMV3InferenceTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        world_size=st.sampled_from([1]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=1,
        deadline=None,
    )
    def test_e2e(self, world_size: int) -> None:
        os.environ["WORLD_SIZE"] = str(world_size)
        main()


if __name__ == "__main__":
    unittest.main()
