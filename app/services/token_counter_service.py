import tiktoken

class TokenCounterService:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Using gpt-4 encoding for 4o-mini

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.encoding.encode(text)) 