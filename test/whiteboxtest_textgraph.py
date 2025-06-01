import pytest
import os
import sys
import sys
sys.path.append(r'F:\sourcepycharm\SoftwareEnginingLab1\src')
from textgraph import TextGraph

@pytest.fixture
def text_graph():
    """测试夹具，初始化TextGraph并加载测试文本"""
    graph = TextGraph()
    test_text = ("The scientist carefully analyzed the data, "
                 "wrote a detailed report, and shared the report with the team, "
                 "but the team requested more data, so the scientist analyzed it again.")

    # 手动构建图结构（确保顺序正确）
    words = [
        "the", "scientist", "carefully", "analyzed", "the", "data",
        "wrote", "a", "detailed", "report",
        "and", "shared", "the", "report", "with", "the", "team",
        "but", "the", "team", "requested", "more", "data",
        "so", "the", "scientist", "analyzed", "it", "again"
    ]

    # 构建邻接表
    for i in range(len(words) - 1):
        current = words[i]
        next_word = words[i + 1]
        graph.adj_list[current][next_word] = graph.adj_list[current].get(next_word, 0) + 1
        graph.nodes.update([current, next_word])
        graph.word_freq[current] += 1
        graph.total_words += 1

    # 处理最后一个词
    if words:
        graph.word_freq[words[-1]] += 1
        graph.total_words += 1

    return graph


def test_query_bridge_words_existing_pair(text_graph):
    """测试存在的单词对"""
    result = text_graph.query_bridge_words("but", "team")
    # 根据实际实现调整断言（注意单复数）
    assert result == "The bridge word from but to team is: the"


def test_query_bridge_words_reverse_pair(text_graph):
    """测试反向单词对"""
    result = text_graph.query_bridge_words("team", "but")
    assert result == "No bridge words from team to but!"


def test_query_bridge_words_nonexistent_word(text_graph):
    """测试包含不存在的单词"""
    result = text_graph.query_bridge_words("but", "teams")
    assert result == "No teams in the graph!"



def test_query_bridge_words_no_bridge(text_graph):
    """测试没有桥接词的情况"""
    result = text_graph.query_bridge_words("data", "wrote")
    assert result == "No bridge words from data to wrote!"