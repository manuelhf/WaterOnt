import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def built():
    from wateront.pipeline import build_knowledge_graph
    return build_knowledge_graph("Net3", verbose=False)
