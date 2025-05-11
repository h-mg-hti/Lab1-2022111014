import re
import random
import heapq
import math
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import os


class TextGraph:
    def __init__(self):
        self.adj_list: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.nodes: Set[str] = set()
        self.page_rank: Dict[str, float] = {}
        self.word_freq: Dict[str, int] = defaultdict(int)
        self.total_words = 0
        self.documents: List[List[str]] = []

    def _clean_word(self, word: str) -> str:
        """Clean word: lowercase and remove non-alphabetic characters"""
        return re.sub(r'[^a-zA-Z]', '', word).lower()

    def read_file(self, filename: str) -> None:
        """Read text file and build graph"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    # Process each line as a document for TF-IDF
                    cleaned_words = [self._clean_word(w) for w in line.split()]
                    cleaned_words = [w for w in cleaned_words if w]
                    if cleaned_words:
                        self.documents.append(cleaned_words)

                    # Build graph edges
                    for i in range(len(cleaned_words) - 1):
                        current, next_word = cleaned_words[i], cleaned_words[i + 1]
                        self.adj_list[current][next_word] = self.adj_list[current].get(next_word, 0) + 1
                        self.nodes.update([current, next_word])
                        self.word_freq[current] += 1
                        self.total_words += 1
                # Count last word
                if cleaned_words:
                    self.word_freq[cleaned_words[-1]] += 1
                    self.total_words += 1

        except IOError as e:
            print(f"Error reading file: {e}")

    def visualize_graph(self, output_file: str = "graph.png") -> None:
        """可视化有向图：先命令行展示，再生成图片"""
        try:
            # 第一步：命令行文本展示
            print("\n【命令行文本格式展示】")
            print("节点:", ", ".join(sorted(self.nodes)))
            print("\n边及权重:")
            for from_node in sorted(self.adj_list):
                for to_node, weight in sorted(self.adj_list[from_node].items()):
                    print(f"  {from_node} → {to_node} (权重:{weight})")

            # 第二步：生成图片文件
            print("\n【生成图形文件中...】")
            G = nx.DiGraph()

            # 添加节点和边
            for from_node, edges in self.adj_list.items():
                for to_node, weight in edges.items():
                    G.add_edge(from_node, to_node, weight=weight)

            # 绘制设置
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)

            # 绘制节点和边
            nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
            nx.draw_networkx_edges(
                G, pos,
                width=[d['weight'] * 0.5 for _, _, d in G.edges(data=True)],
                edge_color='gray',
                arrows=True
            )
            nx.draw_networkx_labels(G, pos, font_size=10)

            # 边权重标签
            edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            plt.title("Text Graph Visualization")
            plt.axis('off')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 图形文件已保存到: {output_file}")

        except Exception as e:
            print(f"可视化失败: {e}")

    def query_bridge_words(self, word1: str, word2: str) -> str:
        """Find bridge words between two words"""
        w1, w2 = self._clean_word(word1), self._clean_word(word2)

        if w1 not in self.nodes or w2 not in self.nodes:
            missing = []
            if w1 not in self.nodes: missing.append(w1)
            if w2 not in self.nodes: missing.append(w2)
            return f"No {' or '.join(missing)} in the graph!"

        bridges = []
        for neighbor in self.adj_list.get(w1, {}):
            if w2 in self.adj_list.get(neighbor, {}):
                bridges.append(neighbor)

        if not bridges:
            return f"No bridge words from {w1} to {w2}!"

        if len(bridges) == 1:
            return f"The bridge word from {w1} to {w2} is: {bridges[0]}"
        else:
            return f"The bridge words from {w1} to {w2} are: {', '.join(bridges[:-1])} and {bridges[-1]}"

    def generate_new_text(self, input_text: str) -> str:
        """Generate new text by inserting bridge words"""
        words = [self._clean_word(w) for w in input_text.split() if self._clean_word(w)]
        if len(words) < 2:
            return input_text

        new_text = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            new_text.append(w1)

            if w1 in self.nodes and w2 in self.nodes:
                bridges = []
                for neighbor in self.adj_list.get(w1, {}):
                    if w2 in self.adj_list.get(neighbor, {}):
                        bridges.append(neighbor)

                if bridges:
                    new_text.append(random.choice(bridges))

        new_text.append(words[-1])
        return ' '.join(new_text)

    def calc_shortest_path(self, word1: str, word2: Optional[str] = None) -> Dict[str, Tuple[int, List[str]]]:
        """Calculate shortest path(s) using Dijkstra's algorithm"""
        w1 = self._clean_word(word1)
        if w1 not in self.nodes:
            return {}

        # Initialize
        distances = {node: float('inf') for node in self.nodes}
        predecessors = {node: None for node in self.nodes}
        distances[w1] = 0
        heap = [(0, w1)]

        # Dijkstra's algorithm
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

        # Build paths
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
        """Calculate TF-IDF scores for words"""
        tf_idf = {}
        num_docs = len(self.documents)

        for word in self.nodes:
            # Term Frequency
            tf = self.word_freq[word] / self.total_words

            # Document Frequency
            df = sum(1 for doc in self.documents if word in doc)

            # Inverse Document Frequency with smoothing
            idf = math.log((num_docs + 1) / (df + 1)) + 1

            tf_idf[word] = tf * idf

        return tf_idf

    def calculate_page_rank(self, damping: float = 0.85, iterations: int = 100,
                            use_tfidf: bool = False) -> None:
        """Calculate PageRank with optional TF-IDF initialization"""
        N = len(self.nodes)
        if N == 0:
            return

        # Initialize PR values
        if use_tfidf:
            tf_idf = self._calculate_tf_idf()
            tf_idf_sum = sum(tf_idf.values())
            self.page_rank = {node: tf_idf[node] / tf_idf_sum for node in self.nodes}
        else:
            self.page_rank = {node: 1.0 / N for node in self.nodes}

        # Handle dangling nodes
        dangling_nodes = [n for n in self.nodes if not self.adj_list.get(n)]

        for _ in range(iterations):
            new_pr = {}
            dangling_pr = sum(self.page_rank[n] for n in dangling_nodes)

            for node in self.nodes:
                # Contribution from other nodes
                pr_sum = 0.0
                for src in self.nodes:
                    if node in self.adj_list.get(src, {}):
                        out_degree = len(self.adj_list[src])
                        pr_sum += self.page_rank[src] / out_degree

                # Update PR value
                new_pr[node] = (1 - damping) / N + damping * (pr_sum + dangling_pr / N)

            # Normalize
            total = sum(new_pr.values())
            self.page_rank = {k: v / total for k, v in new_pr.items()}

    def random_walk(self) -> str:
        """Perform random walk until repeat edge or no out edges"""
        if not self.nodes:
            return ""

        current = random.choice(list(self.nodes))
        path = [current]
        visited_edges = set()

        while True:
            neighbors = list(self.adj_list.get(current, {}).keys())
            if not neighbors:
                break

            next_node = random.choice(neighbors)
            edge = (current, next_node)

            if edge in visited_edges:
                break

            visited_edges.add(edge)
            path.append(next_node)
            current = next_node

        # Save to file
        with open("random_walk.txt", "w") as f:
            f.write(" ".join(path))

        return " ".join(path)


def main():
    graph = TextGraph()
    print("Text Graph Processor")

    # Read file
    while True:
        filename = input("Enter text file path: ").strip()
        if os.path.isfile(filename):
            graph.read_file(filename)
            break
        print(f"File not found: {filename}")

    while True:
        print("\nMenu:")
        print("1. Show graph visualization")
        print("2. Query bridge words")
        print("3. Generate new text")
        print("4. Calculate shortest path")
        print("5. Calculate PageRank") # 可以选择tfidf进行优化
        print("6. Random walk")
        print("0. Exit")

        choice = input("Enter choice: ").strip()

        if choice == "0":
            break


        elif choice == "1":  # 可视化菜单
            out_file = input("输出文件名(默认graph.png): ").strip() or "graph.png"
            graph.visualize_graph(out_file)  # 自动执行两步操作

        elif choice == "2":
            w1 = input("First word: ").strip()
            w2 = input("Second word: ").strip()
            print(graph.query_bridge_words(w1, w2))

        elif choice == "3":
            text = input("Enter text: ").strip()
            print("New text:", graph.generate_new_text(text))

        elif choice == "4":
            w1 = input("Start word: ").strip()
            w2 = input("End word (optional, leave blank for all): ").strip()
            paths = graph.calc_shortest_path(w1, w2 if w2 else None)

            if not paths:
                print("No paths found")
            elif w2:
                dist, path = paths.get(graph._clean_word(w2), (0, []))
                print(f"Shortest path ({dist}): {' → '.join(path)}")
            else:
                print(f"Shortest paths from {w1}:")
                for target, (dist, path) in sorted(paths.items()):
                    print(f"To {target} ({dist}): {' → '.join(path)}")

        elif choice == "5":
            use_tfidf = input("Use TF-IDF initialization? (y/n): ").strip().lower() == 'y'
            graph.calculate_page_rank(use_tfidf=use_tfidf)
            word = input("Enter word to check PR (or leave blank): ").strip()
            if word:
                print(f"PageRank of {word}: {graph.page_rank.get(graph._clean_word(word), 0)}")
            else:
                print("PageRank calculated")

        elif choice == "6":
            print("Random walk result:", graph.random_walk())
            print("Path saved to random_walk.txt")

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()