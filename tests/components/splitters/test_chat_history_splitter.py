from haystack.dataclasses import ChatMessage, ChatRole

from dc_custom_component.components.splitters.chat_history_splitter import SebChatHistorySplitterChatMessages


class TestSebChatHistorySplitterChatMessages:
    def test_init(self):
        splitter = SebChatHistorySplitterChatMessages()
        assert splitter is not None

    def test_run(self):
        splitter = SebChatHistorySplitterChatMessages()
        chat_history_and_query = 'Chat History: [{"role": "user", "content": "Hello!"}]\n\nCurrent Question: How are you?'
        result = splitter.run(chat_history_and_query)
        assert result == {
            "history": [
                ChatMessage(
                    role=ChatRole("user"),
                    content="Hello!",
                    name=None,
                    meta={},
                ),
            ],
            "query": "How are you?",
        }

    def test_run_no_chat_history(self):
        splitter = SebChatHistorySplitterChatMessages()
        chat_history_and_query = "How are you?"
        result = splitter.run(chat_history_and_query)
        assert result == {
            "history": [],
            "query": "How are you?",
        }
