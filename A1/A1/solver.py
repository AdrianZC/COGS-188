import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
        - self includes node in self.neighbors
        - node includes self in node.neighbors (undirected)
        """

        if node not in self.neighbors:
            self.neighbors.append(node)
            node.neighbors.append(self)
        pass

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    if maze is None or maze.size == 0:
        return {}, None, None
    
    rows, cols = maze.shape
    if rows == 0 or cols == 0:
        return {}, None, None
    
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                nodes_dict[(r, c)] = Node((r, c))

    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:
                if r > 0 and maze[r-1][c] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r-1,c)])
                if r < rows-1 and maze[r+1][c] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r+1,c)])
                if c > 0 and maze[r][c-1] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r,c-1)])
                if c < cols-1 and maze[r][c+1] == 0:
                    nodes_dict[(r,c)].add_neighbor(nodes_dict[(r,c+1)])

    start_node = nodes_dict.get((0, 0))
    goal_node = nodes_dict.get((rows-1, cols-1))

    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
    1. Use a queue (collections.deque) to hold nodes to explore.
    2. Track visited nodes so you don’t revisit.
    3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    if not (start_node and goal_node):
        return None
    
    queue = deque([start_node])
    visited = {start_node}
    parent_map = {start_node: None}

    while queue:
        current = queue.popleft()
        if current == goal_node:
            return reconstruct_path(current, parent_map)
        
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                queue.append(neighbor)

    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
    1. Use a stack (Python list) to hold nodes to explore.
    2. Keep track of visited nodes to avoid cycles.
    3. Reconstruct path via parent_map if goal_node is found.
    """
    if not (start_node and goal_node):
        return None
    
    stack = [start_node]
    visited = {start_node}
    parent_map = {start_node: None}

    while stack:
        current = stack.pop()
        if current == goal_node:
            return reconstruct_path(current, parent_map)
        
        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = current
                stack.append(neighbor)
    
    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
    1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
    2. f_score[node] = g_score[node] + heuristic(node, goal_node).
    3. g_score[node] is the cost from start_node to node.
    4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    if not (start_node and goal_node):
        return None
    
    frontier = [(0, start_node)]
    g_score = {start_node: 0}
    parent_map = {start_node: None}
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        if current == goal_node:
            return reconstruct_path(current, parent_map)
        
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                parent_map[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(neighbor, goal_node)
                heapq.heappush(frontier, (f_score, neighbor))
    
    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    r1, c1 = node_a.value
    r2, c2 = node_b.value

    return abs(r1 - r2) + abs(c1 - c2)


###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
    1. Maintain two frontiers (queues), one from start_node, one from goal_node.
    2. Alternate expansions between these two queues.
    3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    if not (start_node and goal_node):
        return None
    
    start_queue = deque([start_node])
    goal_queue = deque([goal_node])
    start_visited = {start_node: None}
    goal_visited = {goal_node: None}

    while start_queue and goal_queue:
        current_start = start_queue.popleft()
        current_goal = goal_queue.popleft()

        if current_start in goal_visited:
            return reconstruct_path(current_start, start_visited) + reconstruct_path(current_start, goal_visited)[::-1][1:]

        if current_goal in start_visited:
            return reconstruct_path(current_goal, goal_visited) + reconstruct_path(current_goal, start_visited)[::-1][1:]

        for neighbor in current_start.neighbors:
            if neighbor not in start_visited:
                start_visited[neighbor] = current_start
                start_queue.append(neighbor)

        for neighbor in current_goal.neighbors:
            if neighbor not in goal_visited:
                goal_visited[neighbor] = current_goal
                goal_queue.append(neighbor)

    return None


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
    1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
    2. Pick a random neighbor. Compute next_cost.
    3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
    4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    if not (start_node and goal_node):
        return None
    
    current = start_node
    path = [current.value]

    while temperature > min_temperature and current != goal_node:
        if not current.neighbors:
            return None
        
        next_node = random.choice(current.neighbors)
        current_cost = manhattan_distance(current, goal_node)
        next_cost = manhattan_distance(next_node, goal_node)
        cost_diff = next_cost - current_cost

        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current = next_node
            path.append(current.value)

        temperature *= cooling_rate

    return path if current == goal_node else None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
    1. Start with end_node, follow parent_map[node] until None.
    2. Collect node.value, reverse the list, return it.
    """
    if not end_node or not parent_map:
        return None
    
    path = []
    current = end_node

    while current:
        path.append(current.value)
        current = parent_map[current]

    return path[::-1]


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
