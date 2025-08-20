from strands import Agent, tool
from strands_tools import current_time, python_repl

@tool
def letter_counter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())

agent = Agent(tools=[current_time, python_repl, letter_counter])

message = """
いくつか質問があります。

1. いま日本は何時？
2. その時刻表示の中に「1」はいくつある？
3. ここまで話した内容をPyhtonコードにして。テストしたあとにコード内容を表示して。
"""

agent(message)
