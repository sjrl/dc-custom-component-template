from typing import List, Dict, Any

from haystack import Document, component


@component
class DocToSourceIDExtender:
    def __init__(self, topK: int):
        self.topK = topK

    @component.output_types(filters=Dict[str, Any])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """ """

        source_ids = []
        filters_list = []
        for doc in documents:
            if "source_id" in doc.meta:
                if doc.meta["source_id"] not in source_ids:
                    source_ids.append(doc.meta["source_id"])
                    filters_list.append({"field": "_id", "operator": "==", "value": doc.meta["source_id"]})

                    if len(filters_list) >= self.topK:
                        break

        filters = {"operator": "OR", "conditions": filters_list}

        return {"filters": filters}
