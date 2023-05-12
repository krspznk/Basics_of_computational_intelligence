import sys


class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        Цей метод забезпечує симетричність графу
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Повертає вузли графу"
        return self.nodes

    def get_outgoing_edges(self, node):
        "Повертає сусідів графу"
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Повертає значення ребра між двома графами"
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # Словник для оновлення руху по графу
    shortest_path = {}

    # словник для збереження найкоротшого шляху до вузлв
    previous_nodes = {}

    # ініціалізація ребер між вузлами, які ще не відвідали
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # Ініціалізація початкового вузла
    shortest_path[start_node] = 0

    # Виконуємо алгоритм, поки не відвідаємо всі вузли
    while unvisited_nodes:
        # Знаходження вузла з найменши значенням ребра
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # Витягуємо сусідів поточного вузла та оновлюмо їх відстані
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # Після відвідування всіх сусідів відмічаємо вузол, чк відвіданий
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Додаємо початковий вузол
    path.append(start_node)

    print("Знайдений найкращим маршрут зі значенням = {}.".format(shortest_path[target_node]))
    print(" -> ".join(reversed(path)))


nodes = ["1","2", "3", "4", "5", "6", "7", "8"]

init_graph = {}
for node in nodes:
    init_graph[node] = {}

init_graph["1"]["2"] = 8
init_graph["1"]["3"] = 7
init_graph["2"]["3"] = 5
init_graph["2"]["4"] = 9
init_graph["2"]["5"] = 12
init_graph["3"]["4"] = 9
init_graph["4"]["5"] = 6
init_graph["4"]["6"] = 11
init_graph["5"]["7"] = 4
init_graph["6"]["7"] = 8
init_graph["6"]["8"] = 7
init_graph["7"]["8"] = 11


graph = Graph(nodes, init_graph)

previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node="3")

print_result(previous_nodes, shortest_path, start_node="3", target_node="8")

