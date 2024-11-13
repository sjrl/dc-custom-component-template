import json
from typing import Dict, List

from haystack import component, logging
from haystack.dataclasses import ChatMessage, ChatRole


logger = logging.getLogger(__name__)


@component
class SebChatHistorySplitter:
    @component.output_types(history=str, query=str)
    def run(self, chat_history_and_query: str) -> Dict[str, str]:
        if (
            "Chat History: " not in chat_history_and_query
            and "\n\nCurrent Question:" not in chat_history_and_query
        ):
            logger.info(
                "Chat History not found in input so returning empty history and treating the input as the query."
            )
            return {"history": "", "query": chat_history_and_query}
        items = chat_history_and_query.split("\n\nCurrent Question:")
        query = items[-1].strip()
        history = items[0].split("Chat History: ")[1]
        return {"history": history, "query": query}


@component
class SebChatHistorySplitterChatMessages:
    @component.output_types(history=List[ChatMessage], query=str)
    def run(self, chat_history_and_query: str) -> Dict[str, str]:
        if (
            "Chat History: " not in chat_history_and_query
            and "\n\nCurrent Question:" not in chat_history_and_query
        ):
            logger.info(
                "Chat History not found in input so returning empty history and treating the input as the query."
            )
            return {"history": [], "query": chat_history_and_query}
        items = chat_history_and_query.split("\n\nCurrent Question:")
        query = items[-1].strip()
        history = items[0].split("Chat History: ")[1]
        history = json.loads(history)
        history_messages = []
        for item in history:
            history_messages.append(ChatMessage(
                role=ChatRole(item["role"]),
                content=item["content"],
                name=item.get("name"),
                meta=item.get("meta", {}),
            ))
        return {"history": history_messages, "query": query}
