import re
import random
import secrets
import heapq
import math
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
DEFAULT_GRAPH_FILE = "graph.png"
RANDOM_WALK_FILE = "random_walk.txt"
PAGE_RANK_INITIALIZATION = "Use TF-IDF for PageRank initialization? (y/n): "
BRIDGE_WORDS_NOT_FOUND = "No bridge words from {} to {}!"
BRIDGE_WORDS_SINGLE = "The bridge word from {} to {} is: {}"
BRIDGE_WORDS_MULTIPLE = "The bridge words from {} to {} are: {} and {}"


class TextGraph:
    """文本图处理器，用于处理文本并构建词关系图

    Attributes:
        adj_list (Dict[str, Dict[str, int]]): 邻接表表示图结构
        nodes (Set[str]): 图中所有节点集合
        page_rank (Dict[str, float]): 节点的PageRank值
        word_freq (Dict[str, int]): 单词频率统计
        total_words (int): 总单词数
        documents (List[List[str]]): 文档集合
    """

    def __init__(self):
        """初始化文本图处理器"""
        self.adj_list: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.nodes: Set[str] = set()
        self.page_rank: Dict[str, float] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.total_words = 0
        self.documents: List[List[str]] = []

    def _clean_word(self, word: str) -> str:
        """清洗单词：小写化并移除非字母字符

        Args:
            word (str): 原始单词

        Returns:
            str: 清洗后的单词
        """
        return re.sub(r'[^a-zA-Z]', '', word).lower()

    def read_file(self, filename: str) -> None:
        """读取文本文件并构建图结构

        Args:
            filename (str): 文本文件路径

        Raises:
            IOError: 文件读取错误时抛出
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    # 作为TF-IDF文档处理
                    cleaned_words = [
                        self._clean_word(w) for w in line.split()
                    ]
                    cleaned_words = [w for w in cleaned_words if w]

                    if cleaned_words:
                        self.documents.append(cleaned_words)

                    # 构建图边
                    for i in range(len(cleaned_words) - 1):
                        current = cleaned_words[i]
                        next_word = cleaned_words[i + 1]

                        # 更新邻接表
                        self.adj_list[current][next_word] = (
                                self.adj_list[current].get(next_word, 0) + 1
                        )

                        # 更新节点集合
                        self.nodes.update([current, next_word])

                        # 更新词频统计
                        self.word_freq[current] += 1
                        self.total_words += 1

                # 统计最后一个词
                if cleaned_words:
                    self.word_freq[cleaned_words[-1]] += 1
                    self.total_words += 1

            logger.info(f"Successfully loaded file: {filename}")

        except IOError as e:
            logger.error(f"Error reading file: {e}")
            raise

    def visualize_graph(self, output_file: str = DEFAULT_GRAPH_FILE) -> None:
        """可视化图结构并保存到文件

        Args:
            output_file (str): 输出文件路径

        Raises:
            nx.NetworkXException: 网络图处理错误时抛出
            RuntimeError: Matplotlib错误时抛出
        """
        try:
            # 命令行文本展示
            print("\n【命令行文本格式展示】")
            print("节点:", ", ".join(sorted(self.nodes)))
            print("\n边及权重:")

            for from_node in sorted(self.adj_list):
                for to_node, weight in sorted(
                        self.adj_list[from_node].items()
                ):
                    print(f"  {from_node} → {to_node} (权重:{weight})")

            # 图形文件生成
            print("\n【生成图形文件中...】")
            graph = nx.DiGraph()

            # 添加节点和边
            for from_node, edges in self.adj_list.items():
                for to_node, weight in edges.items():
                    graph.add_edge(from_node, to_node, weight=weight)

            # 绘图设置
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph, seed=42)

            # 绘制节点和边
            nx.draw_networkx_nodes(
                graph, pos,
                node_size=800,
                node_color='lightblue'
            )

            edge_widths = [d['weight'] * 0.5 for _, _, d in graph.edges(data=True)]
            nx.draw_networkx_edges(
                graph, pos,
                width=edge_widths,
                edge_color='gray',
                arrows=True
            )
            nx.draw_networkx_labels(graph, pos, font_size=10)

            # 边权重标签
            edge_labels = {
                (u, v): d["weight"]
                for u, v, d in graph.edges(data=True)
            }

            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels,
                font_size=8
            )

            plt.title("Text Graph Visualization")
            plt.axis('off')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图形文件已保存到: {output_file}")

        except (nx.NetworkXException, RuntimeError) as e:
            logger.error(f"Graph visualization failed: {e}")
            print(f"图形生成失败: {e}")

    def query_bridge_words(self, word1: str, word2: str) -> str:
        """查询两个词之间的桥接词

        Args:
            word1 (str): 第一个单词
            word2 (str): 第二个单词

        Returns:
            str: 查询结果信息
        """
        w1 = self._clean_word(word1)
        w2 = self._clean_word(word2)

        # 检查单词是否在图中
        if w1 not in self.nodes or w2 not in self.nodes:
            missing = []
            if w1 not in self.nodes:
                missing.append(w1)
            if w2 not in self.nodes:
                missing.append(w2)
            return f"No {' or '.join(missing)} in the graph!"

        bridges = []
        for neighbor in self.adj_list.get(w1, {}):
            if w2 in self.adj_list.get(neighbor, {}):
                bridges.append(neighbor)

        if not bridges:
            return BRIDGE_WORDS_NOT_FOUND.format(w1, w2)

        if len(bridges) == 1:
            return BRIDGE_WORDS_SINGLE.format(w1, w2, bridges[0])
        else:
            return BRIDGE_WORDS_MULTIPLE.format(
                w1, w2,
                ', '.join(bridges[:-1]),
                bridges[-1]
            )

    def generate_new_text(self, input_text: str) -> str:
        # 处理空输入
        if not input_text.strip():
            return ""

        # 清理并过滤单词
        cleaned_words = self._clean_and_filter_words(input_text)

        # 处理有效单词不足的情况
        if not cleaned_words:
            return ""
        if len(cleaned_words) == 1:
            return cleaned_words[0]

        # 处理多个单词
        return self._insert_bridge_words(cleaned_words)

    def _clean_and_filter_words(self, input_text: str) -> List[str]:
        return [cleaned for word in input_text.split()
                if (cleaned := self._clean_word(word))]

    def _insert_bridge_words(self, words: List[str]) -> str:
        new_text = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            new_text.append(w1)

            bridges = self._find_bridge_words(w1, w2)
            if bridges:
                new_text.append(random.choice(bridges))

        new_text.append(words[-1])
        return " ".join(new_text)

    def _find_bridge_words(self, word1: str, word2: str) -> List[str]:
        if word1 not in self.nodes or word2 not in self.nodes:
            return []

        return [
            neighbor for neighbor in self.adj_list.get(word1, {})
            if word2 in self.adj_list.get(neighbor, {})
        ]

    def calc_shortest_path(self, word1: str, word2: Optional[str] = None
                           ) -> Dict[str, Tuple[int, List[str]]]:
        """计算最短路径 (Dijkstra算法)

        Args:
            word1 (str): 起始单词
            word2 (Optional[str]): 目标单词（可选）

        Returns:
            Dict[str, Tuple[int, List[str]]]: 最短路径字典
        """
        w1 = self._clean_word(word1)
        if w1 not in self.nodes:
            return {}

        # 初始化
        distances = {node: float('inf') for node in self.nodes}
        predecessors = {node: None for node in self.nodes}
        distances[w1] = 0
        heap = [(0, w1)]

        # Dijkstra算法
        while heap:
            current_dist, current = heapq.heappop(heap)
            if current_dist > distances[current]:
                continue

            for neighbor, weight in self.adj_list.get(current, {}).items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(heap, (distance, neighbor))

        # 构建路径
        paths = {}
        targets = [self._clean_word(word2)] if word2 else self.nodes

        for target in targets:
            if distances.get(target, float('inf')) == float('inf'):
                continue

            path = []
            node = target
            while node is not None:
                path.append(node)
                node = predecessors[node]
            path.reverse()
            paths[target] = (distances[target], path)

        return paths

    def _calculate_tf_idf(self) -> Dict[str, float]:
        """计算TF-IDF分数

        Returns:
            Dict[str, float]: 单词的TF-IDF分数字典
        """
        tf_idf = {}
        num_docs = len(self.documents)

        for word in self.nodes:
            # 词频 (TF)
            tf = self.word_freq[word] / self.total_words

            # 文档频率 (DF)
            df = sum(1 for doc in self.documents if word in doc)

            # 逆文档频率 (IDF) 平滑处理
            idf = math.log((num_docs + 1) / (df + 1)) + 1

            tf_idf[word] = tf * idf

        return tf_idf

    def calculate_page_rank(
            self,
            damping: float = 0.85,
            iterations: int = 100,
            use_tfidf: bool = False
    ) -> None:
        """计算PageRank (可选TF-IDF初始化)

        Args:
            damping (float): 阻尼系数 (default: 0.85)
            iterations (int): 迭代次数 (default: 100)
            use_tfidf (bool): 是否使用TF-IDF初始化(default: False)
        """
        node_count = len(self.nodes)
        if node_count == 0:
            logger.warning("No nodes to calculate PageRank")
            return

        # 初始化PageRank值
        if use_tfidf:
            try:
                tf_idf = self._calculate_tf_idf()
                tf_idf_sum = sum(tf_idf.values())
                self.page_rank = {
                    node: tf_idf[node] / tf_idf_sum
                    for node in self.nodes
                }
            except Exception as e:
                logger.error(f"TF-IDF initialization failed: {e}")
                # 回退到均匀初始化
                self.page_rank = {node: 1.0 / node_count for node in self.nodes}
        else:
            self.page_rank = {node: 1.0 / node_count for node in self.nodes}

        # 处理悬挂节点
        dangling_nodes = [
            n for n in self.nodes
            if not self.adj_list.get(n)
        ]

        for _ in range(iterations):
            new_pr = {}
            dangling_pr = sum(self.page_rank[n] for n in dangling_nodes)

            for node in self.nodes:
                # 来自其他节点的贡献
                pr_sum = 0.0
                for src in self.nodes:
                    if node in self.adj_list.get(src, {}):
                        out_degree = len(self.adj_list[src])
                        pr_sum += self.page_rank[src] / out_degree

                # 更新PageRank值
                new_pr[node] = (
                        (1 - damping) / node_count +
                        damping * (pr_sum + dangling_pr / node_count)
                )

            # 归一化
            total = sum(new_pr.values())
            self.page_rank = {k: v / total for k, v in new_pr.items()}

    def random_walk(self) -> str:
        """执行随机游走

        Returns:
            str: 游走路径字符串
        """
        if not self.nodes:
            return ""

        current = secrets.choice(list(self.nodes))
        path = [current]
        visited_edges = set()

        while True:
            neighbors = list(self.adj_list.get(current, {}).keys())
            if not neighbors:
                break

            next_node = secrets.choice(neighbors)
            edge = (current, next_node)

            if edge in visited_edges:
                break

            visited_edges.add(edge)
            path.append(next_node)
            current = next_node

        # 保存到文件
        try:
            with open(RANDOM_WALK_FILE, "w", encoding="utf-8") as f:
                f.write(" ".join(path))
            logger.info(f"Random walk path saved to {RANDOM_WALK_FILE}")
        except IOError as e:
            logger.error(f"Error saving random walk: {e}")

        return " ".join(path)