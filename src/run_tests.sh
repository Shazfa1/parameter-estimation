#new bash script for running tests
#!/bin/bash

# Set the directory where your source code is located
SRC_DIR="./src"

# Set the name of your test file
TEST_FILE="TestSimplifiedThreePL.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run tests
run_tests() {
    echo "Running tests for SimplifiedThreePL..."
    
    # Run the unittest command and capture the output
    TEST_OUTPUT=$(python -m unittest $SRC_DIR/$TEST_FILE 2>&1)
    
    # Check the exit status of the test command
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed.${NC}"
        echo "Test output:"
        echo "$TEST_OUTPUT"
    fi
}

# Check if the test file exists
if [ ! -f "$SRC_DIR/$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file $TEST_FILE not found in $SRC_DIR${NC}"
    exit 1
fi

# Main execution
run_tests
