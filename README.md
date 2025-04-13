## 

## Table of Contents
- [Search Algorithms](#search-algorithms)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Search Algorithms](#running-the-searchpy-main-file)
  - [Running Tests](#running-tests)
  - [Visualizing Results](#visualizing-results)
- [Algorithm Details](#algorithm-details)
  - [ACO Implementation](#aco-implementation-details)
  - [Advanced Configuration](#advanced-configuration)
  - [Visualization](#visualization-controls)

## Search Algorithms

1. **CUS2 (Ant Colony Optimization)** - Implemented by Pink
   - Metaheuristic inspired by ant foraging behavior
   - Uses pheromone trails and heuristic information

2. **Floyd-Warshall Algorithm** - Implemented by Pink
   - All-pairs shortest path algorithm
   - Precomputes optimal paths between all node pairs
   - Used to enhance ACO performance by providing virtual edges

## Project Structure

```
PACO-Optimization/
├── search.py                # Main entry point for all algorithms
├── Data/                    # Graph data files
│   ├── Modified_TSP/        # Contains 28 test cases for algorithm testing
│   └── TSP/                 # Original TSP datasets
├── data_reader/            # Graph parsing utilities
│   └── parser.py           # File parser for graphs
├── ACO/          
│   ├── aco_search.py       # ACO main script
│   ├── aco_tuning.py       # ACO hyper-parameter tuning
│   └── aco_routing/        # ACO implementation components
│       ├── aco.py          # Main ACO algorithm
│       ├── ant.py          # Ant agent implementation
│       └── ...             # Supporting modules
└── Tests/                  # Testing infrastructure
    ├── run_test.py     # Script to run tests on all algorithms
    ├── visualize_results.py # Visualization of test results
    ├── Results/            # Test results in text format
    └── Visualizations/     # Generated visualizations
```

## Installation

### Requirements
- Python 3.6+ 
- Required packages:
  - matplotlib
  - numpy
  - pandas
  - seaborn (for visualizations)

### Installation Command:
```bash
pip install matplotlib numpy pandas seaborn
```

## Usage

### Running the search.py main file

Run any search algorithm from the command line:

```bash
python search.py <algorithm> <data_file>
# OR
python search.py <data_file> <algorithm>
```

Where `<algorithm>` is :
- `ACO` - Ant Colony Optimization

And `<data_file>` is the path to a graph file.

### Examples:

```bash
# Run ACO on a test file
python search.py ACO Data/Modified_TSP/test_0.txt

# Run ACO on a test file
python search.py Data/Modified_TSP/test_5.txt ACO
```

### Running Tests

The project includes a testing framework to evaluate all algorithms on multiple test cases. To run tests:

```bash
python Tests/run_test.py
```

This will:
1. Run each algorithm on the 28 test cases in `Data/Modified_TSP/`
2. Generate summary reports in `Tests/Results/`
3. Create a separate summary file for each algorithm: `summary_result_<algorithm>.txt`
4. Each result will include test number, origin, destinations, execution time, path cost, and path

### Visualizing Results

After running tests, visualize the results with:

```bash
# Generate visualizations from test results
python Tests/visualize_results.py
```

This creates:
1. Individual algorithm visualizations in `Tests/Visualizations/<algorithm>/`
   - Success/failure status
   - Execution times
   - Path costs

2. Comparative visualizations in `Tests/Visualizations/Comparison/`
   - Success rate comparison
   - Execution time comparison
   - Path cost comparison
   - Algorithm performance radar chart

## Algorithm Details

### ACO Implementation Details

The Ant Colony Optimization implementation includes several advanced features:

1. **Optimization Modes**:
   - Mode 0: Find any path to a single destination (if multiple destinations provided, it returns shortest destination)
   - Mode 1: Find paths to all destinations (From source to all destinations)
   - Mode 2: Solve TSP (visit all nodes with minimal cost / Random spawn)

2. **Parameter Tuning**:
   - `alpha`: Controls pheromone importance (default: 1)
   - `beta`: Controls heuristic information importance (default: 2)
   - `evaporation_rate`: Learning rate of gradient descent (default: 0.5)

3. **Performance Optimizations**:
   - Edge cost caching: Pre-computes and stores edge costs
   - Neighbor caching: Pre-computes and stores neighbor lists
   - Desirability caching: Stores repeated calculations for path selection

4. **Visualization**:
   - Real-time algorithm progress visualization
   - Pheromone level indication (red: high, green: low)
   - Path highlighting with node and edge details

### Advanced Configuration

For advanced usage, you can modify the parameters in the `aco_search.py` file:

```python
# Key parameters to adjust
ant_max_steps = node_count + 1  # Maximum steps an ant can take
iterations = 500               # Number of algorithm iterations
"""
I suggest that the iterations can be set from range 300-2000 for TSP depend on the complexity of problem and how well solution you want. For TSP ~~ 50 nodes, normally the Algorithm will convergence from iteration 300-500 and start to micro adjust from 500-2000.

For Shortest Path Problem, my recommend is from 50-100 or even 20 if the nodes are too high.
"""
num_ants = node_count           # Number of ants to deploy
alpha = 1                       # Pheromone influence factor
beta = 2                        # Heuristic influence factor
evaporation_rate = 0.5          # Learning rate, the smaller evaporation_rate the bigger pheromone update (1/evaporation_rate)
```

Combination with others algorithm to enhance the quality
``` python
use_search_local = True # Using 2opt
local_search_frequency = 10 # Frequency of using local search
use_floyd_warshall = True # Using floyd warshall to refine the graph before using ACO. This will automatically enable when using with mode 0 (avoiding no result found due to dead-end path)
```

### Visualization Controls

The ACO visualization shows:

- **Nodes**: Path nodes (red) and unused nodes (light blue, 20% opacity)
- **Edges**: Colored by pheromone level (red = high, green: low)
- **Path**: Highlighted with increased opacity and width
- **Information**: Edge costs and pheromone levels shown on path edges
- **Progress**: Current iteration and best path cost

To adjust visualization:
```python
aco = ACO(
    # ... other parameters ...
    visualize=True,              # Enable/disable visualization
    visualization_step=10        # Update frequency (iterations)
    logging = 10 # Change to None to disable logging on terminal
)
```
