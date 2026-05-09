from __future__ import annotations
from .node import Node
from .trace import Trace
from .trajectory import TrajectoryStep
from typing import Sequence
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Graph:
    def __init__(
            self, 
            root_url: str, 
            exp_dir: str, 
            allowlist_patterns: Sequence[str] = tuple(), 
            denylist_patterns: Sequence[str] = tuple(), 
            resume: bool=False
        ):

        self.nodes = {}
        self.explored_nodes = []
        self.unexplored_nodes = []
        self.exp_dir = os.path.join(exp_dir, "graph")
        self.allowlist_patterns = allowlist_patterns
        self.denylist_patterns = denylist_patterns

        if not resume:
            self.root = self.add_url(root_url, None, [])

            # Save graph info
            graph_info = {
                "root_url": self.root.url,
                "allowlist_patterns": self.allowlist_patterns,
                "denylist_patterns": self.denylist_patterns,
            }
            with open(os.path.join(self.exp_dir, "graph_info.json"), "w") as f:
                json.dump(graph_info, f, indent=4)
        
    def get_node(self, url: str) -> Node | None:
        return self.nodes.get(url, None)
    
    def add_url(self, url: str, parent: Node, prefixes: list[Trace], node_misc: dict = None) -> Node:
        
        if url in self.nodes:
            logger.warning(f"In Graph.add_url: Node {url} already exists in the graph.")
            return self.nodes[url]
        
        node_exp_dir = os.path.join(self.exp_dir, f"node_{len(self.nodes)}")
        node = Node(url, {}, {}, [], "", prefixes, False, node_exp_dir, misc=node_misc)
        if parent:
            parent.children.append(node.url)
            parent.update_save(save_prefix=False)
        self.nodes[url] = node
        self.unexplored_nodes.append(node)
        return node
    
    def add_to_explored(self, node: Node):
        self.explored_nodes.append(node)
        self.unexplored_nodes.remove(node)
        node.visited = True
        node.update_save(save_prefix=False)
        logger.info(f"Node {node.url} has been explored.")

    def get_next_node(self) -> Node | None:
        if len(self.unexplored_nodes) == 0:
            logger.info("No nodes left to explore.")
            return None
        return self.unexplored_nodes[0] #TODO: Can add user-defined priortization here.

    
    def check_if_url_allowed(self, url: str) -> bool:
        for pattern in self.allowlist_patterns:
            if re.match(pattern, url):
                return True
        for pattern in self.denylist_patterns:
            if re.match(pattern, url):
                return False
        return True


    @staticmethod
    def load(path: str, load_steps: bool=True, load_prefixes: bool=True, load_images: bool=True, max_nodes=-1) -> Graph:
        nodes = {}
        explored_nodes = []
        unexplored_nodes = []

        logger.info(f"Loading graph from {path}")
        
        if max_nodes == -1:
            max_nodes = len(os.listdir(path)) - 1
        else:
            max_nodes = min(max_nodes, len(os.listdir(path)) - 1)
            
        for i in range(max_nodes):
            logger.info(f"Loading node {i} from {path}")
            node_load_dir = os.path.join(path, f"node_{i}")
            node = Node.load(node_load_dir, load_steps=load_steps, load_prefix=load_prefixes, load_images=load_images)
            nodes[node.url] = node
            if node.visited:
                explored_nodes.append(node)
            else:
                unexplored_nodes.append(node)
        
        graph_info = {}
        with open(os.path.join(path, "graph_info.json"), "r") as f:
            graph_info = json.load(f)
        
        graph = Graph(graph_info["root_url"], path, graph_info["allowlist_patterns"], graph_info["denylist_patterns"], resume=True)
        graph.root = nodes[graph_info["root_url"]]
        graph.nodes = nodes
        graph.explored_nodes = explored_nodes
        graph.unexplored_nodes = unexplored_nodes
        graph.exp_dir = path

        logger.info(f"Loaded graph with {len(nodes)} nodes, {len(explored_nodes)} explored nodes, and {len(unexplored_nodes)} unexplored nodes.")
        
        return graph
