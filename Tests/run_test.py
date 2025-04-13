import os
import sys
import subprocess
import time
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the parse_graph_file function from data_reader module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "data_reader"))
from parser import parse_graph_file

def run_test(test_file_path, algorithm):
    """
    Run specified algorithm on the given test file and return the results.
    
    Args:
        test_file_path (str): Path to the test file
        algorithm (str): Algorithm to run 
    Returns:
        dict: Results of the test run
    """
    start_time = time.time()
    
    if algorithm == "ACO":
        cmd = [sys.executable, str(project_root / "search.py"), test_file_path, "ACO"]
    else:
        return {
            "success": False,
            "error": f"Unknown algorithm: {algorithm}",
            "execution_time": 0
        }
    
    try:
        result = subprocess.run(cmd, 
                               capture_output=True, 
                               text=True, 
                               timeout=300) # 300 second timeout
        
        execution_time = time.time() - start_time
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr,
                "execution_time": execution_time
            }
        
        # Parse the output
        output_lines = result.stdout.strip().split('\n')
        if len(output_lines) < 3:
            return {
                "success": False,
                "error": "Incomplete output from algorithm",
                "output": result.stdout,
                "execution_time": execution_time
            }
        
        algorithm_info = output_lines[0]
        goal_info = output_lines[1]
        path = output_lines[2]
        
        # Extract cost from the path if it's not provided separately
        cost = "N/A"  # Default if cost can't be extracted
        
        # Check if there's a fourth line with cost
        if len(output_lines) >= 4 and output_lines[3].strip():
            cost = output_lines[3]
        else:
            # Try to extract cost from the path - look for a pattern like [...] (cost: X.XX)
            # or extract from goal_info if needed
            # This is a fallback in case the cost format changes
            try:
                # If path shows no path found, set cost to 0
                if "No path found" in path or "Destination already reached" in path:
                    cost = "0.0"
            except:
                pass
        
        # Try to extract test number from file name
        test_num = "Unknown"
        match = re.search(r'test_(\d+)\.txt', os.path.basename(test_file_path))
        if match:
            test_num = match.group(1)

        return {
            "success": True,
            "algorithm_info": algorithm_info,
            "goal_info": goal_info,
            "path": path,
            "cost": cost,
            "execution_time": execution_time,
            "test_num": test_num
        }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Test timed out after 300 seconds",
            "execution_time": 300
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time
        }

def get_test_info_from_parser(test_file_path):
    """
    Extract origin and destination information from test file using the parse_graph_file function
    
    Args:
        test_file_path (str): Path to the test file
        
    Returns:
        dict: Dictionary containing origin and destinations
    """
    try:
        # Use the parse_graph_file function to extract necessary information
        nodes, edges, origin, destinations = parse_graph_file(test_file_path)
        
        # Format destinations for display
        formatted_destinations = ", ".join(sorted(destinations)) if destinations else "Unknown"
        
        return {
            "origin": origin if origin else "Unknown",
            "destinations": formatted_destinations
        }
    except Exception as e:
        print(f"Error parsing graph from {test_file_path}: {e}")
        return {
            "origin": "Unknown",
            "destinations": "Unknown"
        }

def main():
    # Available algorithms to test
    algorithms = ["ACO"]
    
    # Create test results directory if it doesn't exist
    results_dir = project_root / "Tests" / "Results"
    results_dir.mkdir(exist_ok=True)
    
    # Get list of all test files from Modified_TSP directory
    test_cases_dir = project_root / "Data" / "Modified_TSP"
    test_files = sorted([f for f in test_cases_dir.glob("test_*.txt")])
    
    if not test_files:
        print("No test files found in", test_cases_dir)
        return
        
    print(f"Found {len(test_files)} test files in {test_cases_dir}")
    
    # Extract test information from all test files up front using parse_graph_file
    test_info_cache = {}
    for test_file in test_files:
        test_info = get_test_info_from_parser(str(test_file))
        test_info_cache[test_file.name] = test_info
    
    # Track overall results for all algorithms
    all_results = {alg: [] for alg in algorithms}
    
    # Test each algorithm
    for algorithm in algorithms:
        print(f"\n=== Testing {algorithm} algorithm ===")
        
        success_count = 0
        total_tests = len(test_files)
        algorithm_results = []
        
        for i, test_file in enumerate(test_files, 1):
            print(f"Running {algorithm} test {i}/{total_tests}: {test_file.name}")
            
            # Get cached test information
            test_info = test_info_cache[test_file.name]
            
            # Run the test with the specified algorithm
            result = run_test(str(test_file), algorithm)
            
            # Add test info and file name to the result
            result["test_file"] = test_file.name
            result["origin"] = test_info["origin"]
            result["destinations"] = test_info["destinations"]
            
            algorithm_results.append(result)
            all_results[algorithm].append(result)
            
            if result["success"]:
                success_count += 1
                print(f"  ✓ Success (Time: {result['execution_time']:.3f}s, Cost: {result['cost']})")
            else:
                print(f"  ✗ Failed: {result['error']}")
        
        # Generate summary report for this algorithm
        with open(results_dir / f"summary_result_{algorithm}.txt", "w", encoding='utf-8') as f:
            f.write(f"# {algorithm} Algorithm Test Results\n")
            f.write(f"Tests run: {total_tests}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {total_tests - success_count}\n\n")
            
            # Table header with fixed column widths
            f.write("| Test #   | Origin       | Destination(s)        | Time (s)  | Path Cost   | Path                                                |\n")
            f.write("|----------|--------------|----------------------|-----------|-------------|-----------------------------------------------------|\n")
            
            for result in algorithm_results:
                test_num = result.get("test_num", "Unknown")
                origin = result.get("origin", "Unknown")
                destinations = result.get("destinations", "Unknown")
                
                if result['success']:
                    # Format successful test result as table row with padding
                    f.write(f"| {test_num:<8} | {origin:<12} | {destinations:<20} | {result['execution_time']:.3f} | {result['cost']:<11} | {result['path']:<50} |\n")
                else:
                    # Format failed test result with padding
                    error_msg = result['error'][:40] + "..." if len(result['error']) > 40 else result['error']
                    f.write(f"| {test_num:<8} | {origin:<12} | {destinations:<20} | {result['execution_time']:.3f} | {'Failed':<11} | Error: {error_msg:<42} |\n")
            
        print(f"\n{algorithm} testing complete. {success_count}/{total_tests} tests passed.")
    
    # Generate comparative summary report for all algorithms
    with open(results_dir / "summary_comparison.txt", "w", encoding='utf-8') as f:
        f.write("# Comparative Algorithm Test Results\n\n")
        
        # Table header for comparison with fixed widths
        f.write("| Test #   | Algorithm   | Success   | Time (s)  | Path Cost   |\n")
        f.write("|----------|-------------|-----------|-----------|-------------|\n")
        
        # Group by test file
        for i, test_file in enumerate(test_files, 1):
            test_name = test_file.name
            test_num = re.search(r'test_(\d+)\.txt', test_name).group(1) if re.search(r'test_(\d+)\.txt', test_name) else i
            
            for algorithm in algorithms:
                results = [r for r in all_results[algorithm] if r["test_file"] == test_name]
                if results:
                    result = results[0]
                    # Use "Yes" and "No" instead of Unicode symbols to avoid encoding issues
                    success = "Yes" if result["success"] else "No"
                    cost = result["cost"] if result["success"] else "N/A"
                    f.write(f"| {test_num:<8} | {algorithm:<11} | {success:<9} | {result['execution_time']:.3f} | {cost:<11} |\n")
    
    print(f"\nAll testing complete. See {results_dir} for detailed results.")

if __name__ == "__main__":
    main()