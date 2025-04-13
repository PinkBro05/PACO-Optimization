import random

# File paths
input_file = 'Data/Modified_TSP/Email-EuALL.txt'   # This should contain lines like "0 1"
output_file = 'Data/Modified_TSP/test_30.txt'

# Step 1: Read the input edges from file
edges = []
nodes = set()

with open(input_file, 'r') as f:
    for line in f:
        if line.strip():
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            nodes.update([u, v])

# Step 2: Generate random coordinates for each node (x, y >= 0)
node_coords = {}
for node in nodes:
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    node_coords[node] = (x, y)

# Step 3: Assign a random cost to each edge
edge_with_cost = [(u, v, random.randint(1, 20)) for u, v in edges]

# Step 4: Write to the output file
with open(output_file, 'w') as f:
    f.write("Nodes:\n")
    for node, (x, y) in sorted(node_coords.items()):
        f.write(f"{node}: ({x},{y})\n")

    f.write("\nEdges:\n")
    for u, v, cost in edge_with_cost:
        f.write(f"({u},{v}): {cost}\n")

print(f"{output_file} created successfully.")