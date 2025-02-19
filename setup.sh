#!/bin/bash

# Exit on error
set -e

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No color

# Check if script is run with --force
FORCE_INSTALL=false
if [[ "$1" == "--force" ]]; then
    FORCE_INSTALL=true
    echo -e "${RED}Forcing reinstallation... Removing existing submodule.${NC}"
    
    # Deinitialize and remove the submodule properly
    git submodule deinit -f include/gerbil-DataFrame
    rm -rf .git/modules/include/gerbil-DataFrame
    git rm -f include/gerbil-DataFrame
    rm -rf include/gerbil-DataFrame

    # Ensure changes are committed before re-adding
    git add .gitmodules || true
    git commit -m "Removed gerbil-DataFrame submodule" || true
fi

# Check if the software is already installed
if [ -f "include/gerbil-DataFrame/build/gerbil" ] && [ "$FORCE_INSTALL" = false ]; then
    echo -e "${BLUE}The software is already installed.${NC}"
    exit 0
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if the "module" command exists (for HPC systems)
if command_exists module; then
    USE_MODULES=true
else
    USE_MODULES=false
fi

# Check for required tools and attempt to load modules if missing
MISSING_TOOLS=()

# Check GCC
if ! command_exists gcc; then
    if $USE_MODULES; then
        echo -e "${RED}GCC not found. Trying to load module...${NC}"
        if ml gcc; then
            echo -e "${GREEN}Successfully loaded GCC via module.${NC}"
        else
            echo -e "${RED}Error: Could not load GCC via module.${NC}"
            MISSING_TOOLS+=("gcc")
        fi
    else
        echo -e "${RED}Error: GCC is not installed. Please install it before running the setup.${NC}"
        MISSING_TOOLS+=("gcc")
    fi
else
    echo -e "${GREEN}GCC is already installed.${NC}"
fi

# Check CMake
if ! command_exists cmake; then
    if $USE_MODULES; then
        echo -e "${RED}CMake not found. Trying to load module using 'ml cmake'...${NC}"
        if ml cmake; then
            echo -e "${GREEN}Successfully loaded CMake via 'ml cmake'.${NC}"
        else
            echo -e "${RED}Error: Could not load CMake via 'ml cmake'.${NC}"
            MISSING_TOOLS+=("cmake")
        fi
    else
        echo -e "${RED}Error: CMake is not installed. Please install it before running the setup.${NC}"
        MISSING_TOOLS+=("cmake")
    fi
else
    echo -e "${GREEN}CMake is already installed.${NC}"
fi

# Check Boost (Only Load It, No Searching)
if $USE_MODULES; then
    echo -e "${RED}Boost not found. Trying to load module using 'ml boost'...${NC}"
    if ml boost; then
        echo -e "${GREEN}Successfully loaded Boost via 'ml boost'.${NC}"
    else
        echo -e "${RED}Error: Could not load Boost via 'ml boost'.${NC}"
        MISSING_TOOLS+=("boost")
    fi
else
    echo -e "${RED}Error: Boost is not installed. Please install it before running the setup.${NC}"
    MISSING_TOOLS+=("boost")
fi

# If any tools are missing, exit
if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "${RED}Setup cannot continue. Please install/load the missing dependencies: ${MISSING_TOOLS[*]} and run the script again.${NC}"
    exit 1
fi

echo -e "${GREEN}All required dependencies are installed or loaded. Proceeding with setup.${NC}"

# Ensure the include directory exists
mkdir -p include

# Add the submodule only if it does not exist
if [ ! -d "include/gerbil-DataFrame" ]; then
    echo -e "${GREEN}Adding the submodule gerbil-DataFrame to include/ directory...${NC}"
    git submodule add https://github.com/M-Serajian/gerbil-DataFrame.git include/gerbil-DataFrame
fi

# Ensure submodule is initialized and updated
git submodule init
git submodule update --recursive

# Navigate to the submodule and build
cd include/gerbil-DataFrame

# Create build directory
mkdir -p build
cd build

echo "Running CMake..."
# Run CMake
if cmake ..; then
    echo -e "${GREEN}CMake configuration completed successfully.${NC}"
else
    echo -e "${RED}Error: CMake configuration failed.${NC}"
    exit 1
fi

# Build the project
echo "Running make..."
if make -j$(nproc); then
    echo -e "${GREEN}Compilation completed successfully.${NC}"
else
    echo -e "${RED}Error: Compilation failed. Please check the error messages above.${NC}"
    exit 1
fi

# Return to the main directory
cd ../../..

echo -e "${GREEN}Setup completed successfully.${NC}"
