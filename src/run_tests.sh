#new bash script for running tests
#generated using ChatGPT
#!/bin/bash

# Set the directory where your source code is located
SRC_DIR="./src"

# Set the name of your test file
TEST_FILE="test_SimplifiedThreePL.py"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run tests
run_tests() {
    echo "Running tests for SimplifiedThreePL..."
    
    # Run the unittest command with verbose output and capture it
    TEST_OUTPUT=$(python3 -m unittest -v $SRC_DIR/$TEST_FILE 2>&1)
    
    # Check the exit status of the test command
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        echo "$TEST_OUTPUT"
    else
        echo -e "${RED}Some tests failed.${NC}"
        echo "Test output:"
        
        # Process the output to highlight failures
        while IFS= read -r line; do
            if [[ $line == FAIL:* ]]; then
                echo -e "${RED}$line${NC}"
            elif [[ $line == ERROR:* ]]; then
                echo -e "${RED}$line${NC}"
            elif [[ $line == test_* ]]; then
                if [[ $line == *ok ]]; then
                    echo -e "${GREEN}$line${NC}"
                else
                    echo -e "${YELLOW}$line${NC}"
                fi
            else
                echo "$line"
            fi
        done <<< "$TEST_OUTPUT"
    fi
}

# Check if the test file exists
if [ ! -f "$SRC_DIR/$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file $TEST_FILE not found in $SRC_DIR${NC}"
    exit 1
fi

# Main execution
run_tests
