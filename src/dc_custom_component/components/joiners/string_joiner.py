from typing import Dict, List

from haystack import component, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


@component
class SebStringJoiner:
    """
    Component to join strings from different components to a list of strings
    """

    @component.output_types(strings=List[str])
    def run(self, strings: Variadic[str]) -> Dict[str, List[str]]:
        """
        Joins strings into a list of strings
        :param strings: strings from different components

        """
        strings = list(strings)
        return {"strings": strings}
