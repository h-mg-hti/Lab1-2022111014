import os
import ast
import subprocess
from collections import defaultdict
import platform


def run_flake8(file_path):
    """使用flake8检查PEP8规范"""
    try:
        # Windows路径处理
        if platform.system() == 'Windows':
            file_path = file_path.replace('\\', '/')

        result = subprocess.run(
            ["flake8", "--select=E,W", file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.splitlines()
    except Exception as e:
        return [f"Flake8检查失败: {str(e)}"]


def run_pylint(file_path):
    """使用pylint检查代码质量"""
    try:
        # Windows路径处理
        if platform.system() == 'Windows':
            file_path = file_path.replace('\\', '/')

        result = subprocess.run(
            ["pylint", "--score=n", file_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        return [line for line in result.stdout.splitlines() if ":" in line]
    except Exception as e:
        return [f"Pylint检查失败: {str(e)}"]


def custom_code_review(file_path):
    """自定义代码检查规则"""
    issues = []
    if not os.path.exists(file_path):
        return [f"文件不存在: {file_path}"]

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            code = f.read()
            tree = ast.parse(code)

            # 检查变量命名规范
            issues += check_naming_conventions(tree)

            # 检查函数长度
            issues += check_function_lengths(code)

            # 检查硬编码值
            issues += check_hardcoded_values(tree)

            # 检查异常处理
            issues += check_exception_handling(tree)

            # 检查注释比例
            issues += check_comment_ratio(code)

        except SyntaxError as e:
            issues.append(f"语法错误: {str(e)}")
        except Exception as e:
            issues.append(f"分析失败: {str(e)}")

    return issues


def check_naming_conventions(tree):
    """检查命名是否符合蛇形命名法"""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.islower() and not node.name.startswith('_'):
                issues.append(f"函数命名不规范: '{node.name}' 应使用蛇形命名法(line:{node.lineno})")
        elif isinstance(node, ast.ClassDef):
            if not node.name[0].isupper():
                issues.append(f"类命名不规范: '{node.name}' 应使用驼峰式(line:{node.lineno})")
        elif isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name) and not node.targets[0].id.islower() and not node.targets[
                0].id.startswith('_'):
                issues.append(f"变量命名不规范: '{node.targets[0].id}' 应使用蛇形命名法(line:{node.lineno})")
    return issues


def check_function_lengths(code):
    """检查函数是否过长（>40行）"""
    issues = []
    lines = code.split('\n')
    function_line = 0
    function_name = None
    count = 0
    in_function = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 检测函数定义
        if stripped.startswith("def "):
            if in_function and function_name and count > 40:
                issues.append(f"函数过长: '{function_name}' ({count}行), 建议重构(line:{function_line})")

            # 开始新的函数
            function_name = stripped.split('(')[0].split()[-1]
            function_line = i + 1
            count = 0
            in_function = True

        elif stripped.startswith("class "):
            # 类定义会重置函数计数
            in_function = False
            function_name = None

        elif in_function:
            # 统计非空行且非注释行
            if stripped and not stripped.startswith('#'):
                count += 1

    # 检查最后一个函数
    if in_function and function_name and count > 40:
        issues.append(f"函数过长: '{function_name}' ({count}行), 建议重构(line:{function_line})")

    return issues


def check_hardcoded_values(tree):
    """检查可能的硬编码值"""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 50:  # 长字符串可能应该放在配置文件
                issues.append(f"可疑的硬编码长字符串(line:{node.lineno})")
            elif any(phrase in node.value.lower() for phrase in ["password", "secret", "key", "token"]):
                issues.append(f"潜在的敏感信息硬编码: 字符串包含敏感词汇(line:{node.lineno})")

        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value > 1000:  # 大数值可能应该设为常量
                issues.append(f"可疑的硬编码数值: {node.value}(line:{node.lineno})")

    return issues


def check_exception_handling(tree):
    """检查异常处理是否具体"""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(f"裸异常捕获: 应该指定具体异常类型(line:{node.lineno})")
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                issues.append(f"过于宽泛的异常捕获: 应使用具体异常类型而非全局Exception(line:{node.lineno})")
    return issues


def check_comment_ratio(code):
    """检查代码的注释比例（至少10%）"""
    issues = []
    lines = code.split('\n')
    code_lines = 0
    comment_lines = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        elif stripped.startswith('#'):
            comment_lines += 1
        else:
            code_lines += 1

    total_lines = code_lines + comment_lines
    if total_lines == 0:
        return issues

    ratio = comment_lines / total_lines * 100
    if ratio < 10:
        issues.append(f"注释比例不足: 仅 {ratio:.1f}% (建议至少10%)")

    return issues


def code_review_report(file_path):
    """生成完整的代码审查报告"""
    print(f"\n{'=' * 40}")
    print(f"代码审查报告: {os.path.basename(file_path)}")
    print(f"文件路径: {file_path}")
    print(f"{'=' * 40}")

    # 分类存储问题
    issues = defaultdict(list)

    # 运行各检查
    issues["PEP 8规范"].extend(run_flake8(file_path))
    issues["Pylint代码质量"].extend(run_pylint(file_path))
    issues["自定义代码问题"].extend(custom_code_review(file_path))

    # 按类别打印问题
    found_issues = False
    for category, problem_list in issues.items():
        if not problem_list:
            continue

        found_issues = True
        print(f"\n【{category}】")
        for i, problem in enumerate(problem_list, 1):
            print(f"{i}. {problem}")

    # 总结统计
    total = sum(len(v) for v in issues.values())

    if not found_issues:
        print("\n👍 未发现问题 - 代码符合基本规范")
    else:
        print(f"\n总问题数: {total}个")
        print("\n改进建议:")
        print("- 使用flake8/pylint可自动修复PEP8和基本代码质量问题")
        print("- 使用snake_case命名函数和变量，CamelCase命名类")
        print("- 拆分超过40行的函数")
        print("- 提取硬编码值到配置或常量")

    print("=" * 40)


if __name__ == "__main__":
    # 你的文件路径
    file_to_review = r"F:\sourcepycharm\SoftwareEnginingLab1\src\textgraph.py"

    # 检查文件是否存在
    if not os.path.exists(file_to_review):
        print(f"错误: 文件不存在 - {file_to_review}")
        # 尝试可能的替代路径
        possible_paths = [
            r"F:\sourcepycharm\SoftwareEnginingLab1\src\textgraph.py",
            r"F:/sourcepycharm/SoftwareEnginingLab1/src/textgraph.py",
            "./src/textgraph.py",
            "textgraph.py"
        ]

        print("\n尝试查找文件...")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到文件: {path}")
                file_to_review = path
                break
        else:
            print("未找到文件，请确认路径")
            exit(1)

    print(f"开始审查: {file_to_review}")
    code_review_report(file_to_review)