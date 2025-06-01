import pytest
import os
import tempfile
import sys

# 添加src目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from textgraph import TextGraph

# 测试内容
TEST_CONTENT = """
The scientist carefully analyzed the data, wrote a detailed report,
and shared the report with the team. But the team requested more data,
so the scientist analyzed it again.
"""


@pytest.fixture(scope="module")
def sample_text_graph():
    """创建带有测试数据的TextGraph实例"""
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tmp_file:
        tmp_file.write(TEST_CONTENT)
        file_path = tmp_file.name

    # 初始化并加载图数据
    graph = TextGraph()
    graph.read_file(file_path)

    yield graph

    # 清理临时文件
    os.unlink(file_path)


def print_success(desc, input_text, output):
    """打印测试成功信息"""
    print(f"\n✓ {desc} 测试通过")
    print(f"   输入: '{input_text}'")
    print(f"   输出: '{output}'")


def test_generate_new_text_normal(sample_text_graph):
    """测试正常输入"""
    input_text = "Scientist wrote a report"
    result = sample_text_graph.generate_new_text(input_text)

    # 验证结果
    assert "scientist" in result.lower()
    assert "report" in result.lower()
    assert len(result.split()) >= 4

    # 打印成功信息
    print_success("正常输入", input_text, result)


def test_generate_new_text_special_chars(sample_text_graph):
    """测试特殊字符输入"""
    input_text = "@@113"
    result = sample_text_graph.generate_new_text(input_text)

    assert result == "", f"预期空字符串，实际得到: '{result}'"

    # 打印成功信息
    print_success("特殊字符输入", input_text, result)


def test_generate_new_text_mixed_chars(sample_text_graph):
    """测试混合字符输入"""
    input_text = "Scientist@ wrote %% a report"
    result = sample_text_graph.generate_new_text(input_text)

    # 验证特殊字符被移除
    assert "@" not in result, f"特殊字符@仍在输出中: {result}"
    assert "%%" not in result, f"特殊字符%%仍在输出中: {result}"

    # 验证核心词汇被保留
    words = result.lower().split()
    assert "scientist" in words, f"'scientist' 未在输出中发现: {result}"
    assert "wrote" in words, f"'wrote' 未在输出中发现: {result}"
    assert "a" in words, f"'a' 未在输出中发现: {result}"
    assert "report" in words, f"'report' 未在输出中发现: {result}"

    # 打印成功信息
    print_success("混合字符输入", input_text, result)


def test_generate_new_text_empty(sample_text_graph):
    """测试空输入"""
    input_text = ""
    result = sample_text_graph.generate_new_text(input_text)

    assert result == "", f"预期空字符串，实际得到: '{result}'"

    # 打印成功信息
    print_success("空输入", input_text, result)


# 额外的测试用例
def test_generate_new_text_single_word(sample_text_graph):
    """测试单单词输入"""
    input_text = "@@@Scientist###"
    result = sample_text_graph.generate_new_text(input_text)

    assert result == "scientist", f"预期'scientist'，实际得到: '{result}'"

    # 打印成功信息
    print_success("单单词输入", input_text, result)


def test_generate_new_text_only_special_characters(sample_text_graph):
    """测试全特殊字符输入"""
    input_text = "@#$ ^&*()"
    result = sample_text_graph.generate_new_text(input_text)

    assert result == "", f"预期空字符串，实际得到: '{result}'"

    # 打印成功信息
    print_success("全特殊字符输入", input_text, result)


# 可选：添加测试总结
@pytest.fixture(scope="session", autouse=True)
def test_summary():
    """在所有测试结束后打印总结信息"""
    yield
    print("\n" + "=" * 60)
    print("所有测试用例已完成 - 全部通过!")
    print("=" * 60)