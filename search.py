import sys
import subprocess
import os
import importlib.util
import traceback

def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python search.py <algorithm> [optional arguments]")
        print("   OR: python search.py <input_file> <algorithm>")
        print("Available algorithms: ACO")
        sys.exit(1)

    # Parse command line arguments - support both formats
    # Format 1: python search.py <algorithm> <input_file>
    # Format 2: python search.py <input_file> <algorithm>
    
    # Identify algorithm and file path
    first_arg = sys.argv[1]
    
    # List of recognized algorithms
    algorithms = ["ACO"]
    
    # Determine if the first argument is an algorithm or a file path
    if first_arg in algorithms:
        # Format 1: python search.py <algorithm> <input_file>
        algorithm = first_arg
        remaining_args = sys.argv[2:]  # The rest are args (potentially file path)
    elif len(sys.argv) >= 3 and sys.argv[2] in algorithms:
        # Format 2: python search.py <input_file> <algorithm>
        file_path = first_arg
        algorithm = sys.argv[2]
        remaining_args = [file_path] + sys.argv[3:]  # File path + any extra args
    else:
        print("Error: Could not identify algorithm.")
        print("Format should be: python search.py <algorithm> <input_file>")
        print("             OR: python search.py <input_file> <algorithm>")
        print(f"Recognized algorithms: {', '.join(algorithms)}")
        sys.exit(1)

    # Execute the appropriate algorithm
    if algorithm == "ACO":
        try:
            # Get paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_search_dir = os.path.join(current_dir, "ACO")
            module_path = os.path.join(custom_search_dir, "aco_search.py")
            
            # Add Custom_Search directory to sys.path so its modules can be found
            if custom_search_dir not in sys.path:
                sys.path.insert(0, custom_search_dir)
            
            # Check if the module exists
            if os.path.exists(module_path):
                
                # Import the module
                spec = importlib.util.spec_from_file_location("aco_search", module_path)
                aco_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(aco_module)
                
                # Save original argv
                original_argv = sys.argv.copy()
                
                # Update argv to pass the input file path if provided
                if remaining_args:
                    sys.argv = [module_path] + remaining_args
                else:
                    sys.argv = [module_path]
                
                # Run the main function
                aco_module.main()
                
                # Restore original argv
                sys.argv = original_argv
            else:
                print(f"Error: Module {module_path} not found!")
                sys.exit(1)
        except Exception as e:
            print(f"Error executing ACO search: {e}")
            traceback.print_exc()  # Print the full stack trace for debugging
            sys.exit(1)
        
    else:
        print(f"Unknown algorithm: {algorithm}")
        sys.exit(1)

if __name__ == "__main__":
    main()