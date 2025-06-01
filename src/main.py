from textgraph import TextGraph, DEFAULT_GRAPH_FILE, RANDOM_WALK_FILE, PAGE_RANK_INITIALIZATION
from textgraph import BRIDGE_WORDS_NOT_FOUND, BRIDGE_WORDS_SINGLE, BRIDGE_WORDS_MULTIPLE
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义（特定于主程序）
FILE_NOT_FOUND_MSG = "File not found: {}"
ENTER_CHOICE_PROMPT = "Enter choice (0-6): "


def show_menu() -> None:
    """显示主菜单"""
    print("\n文本图处理器菜单:")
    print("1. 显示图可视化")
    print("2. 查询桥接词")
    print("3. 生成新文本")
    print("4. 计算最短路径")
    print("5. 计算PageRank")
    print("6. 随机游走")
    print("0. 退出")


def handle_visualization(graph: TextGraph) -> None:
    """处理可视化选项"""
    out_file = input("输出文件名(默认graph.png): ").strip()
    out_file = out_file or DEFAULT_GRAPH_FILE
    graph.visualize_graph(out_file)


def handle_bridge_words(graph: TextGraph) -> None:
    """处理桥接词查询"""
    w1 = input("第一个词: ").strip()
    w2 = input("第二个词: ").strip()
    print(graph.query_bridge_words(w1, w2))


def handle_text_generation(graph: TextGraph) -> None:
    """处理新文本生成"""
    text = input("输入文本: ").strip()
    print("新文本:", graph.generate_new_text(text))


def handle_shortest_path(graph: TextGraph) -> None:
    """处理最短路径计算"""
    w1 = input("起始词: ").strip()
    w2 = input("目标词 (选填): ").strip()

    if w2:
        # 单一路径
        paths = graph.calc_shortest_path(w1, w2)
        target = graph._clean_word(w2)
        result = paths.get(target)

        if result:
            dist, path = result
            print(f"最短路径 ({dist}): {' → '.join(path)}")
        else:
            print("未找到路径")
    else:
        # 所有路径
        paths = graph.calc_shortest_path(w1)

        if paths:
            print(f"从 {w1} 出发的最短路径:")
            for target, (dist, path) in sorted(paths.items()):
                print(f"到 {target} ({dist}): {' → '.join(path)}")
        else:
            print("未找到路径")


def handle_page_rank(graph: TextGraph) -> None:
    """处理PageRank计算"""
    use_tfidf = input(PAGE_RANK_INITIALIZATION).strip().lower() == 'y'

    try:
        graph.calculate_page_rank(use_tfidf=use_tfidf)
        print("PageRank计算完成")

        word = input("输入要查询的单词(选填): ").strip()
        if word:
            cleaned = graph._clean_word(word)
            value = graph.page_rank.get(cleaned, 0)
            print(f"{word} 的PageRank值: {value:.6f}")
    except Exception as e:
        logger.error(f"PageRank calculation failed: {e}")
        print(f"PageRank计算失败: {e}")


def handle_random_walk(graph: TextGraph) -> None:
    """处理随机游走"""
    result = graph.random_walk()
    print("随机游走结果:", result)
    print(f"路径已保存到{RANDOM_WALK_FILE}")


def main():
    """主程序函数"""
    print("文本图处理器")

    # 创建图实例
    graph = TextGraph()

    # 读取文件
    while True:
        try:
            filename = input("输入文本文件路径: ").strip()
            if os.path.isfile(filename):
                graph.read_file(filename)
                break
            print(FILE_NOT_FOUND_MSG.format(filename))
        except Exception as e:
            logger.error(f"Error during file loading: {e}")
            print(f"文件加载失败: {e}. 请重试.")

    # 主菜单循环
    while True:
        try:
            show_menu()
            choice = input(ENTER_CHOICE_PROMPT).strip()

            if choice == "0":
                print("程序已退出")
                break

            elif choice == "1":
                handle_visualization(graph)

            elif choice == "2":
                handle_bridge_words(graph)

            elif choice == "3":
                handle_text_generation(graph)

            elif choice == "4":
                handle_shortest_path(graph)

            elif choice == "5":
                handle_page_rank(graph)

            elif choice == "6":
                handle_random_walk(graph)

            else:
                print("无效选项，请重试")

        except KeyboardInterrupt:
            print("\n操作已取消")
        except Exception as e:
            logger.error(f"处理错误: {e}")
            print(f"发生错误: {e}. 请重试.")


if __name__ == "__main__":
    main()