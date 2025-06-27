#!/bin/bash

# Exit on error
set -e

# Define color codes for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

echo -e "${GREEN}Installing gerbil-DataFrame dependency...${NC}"
echo "******************************************************************"

# ------------------ Load Required Modules ------------------
echo -e "${BLUE}Loading required modules...${NC}"

ml gcc/12.2
ml cmake
ml boost/1.77

echo -e "${GREEN}Modules loaded successfully.${NC}"

# ------------------ Clone gerbil-DataFrame ------------------
GERBIL_DIR="include/gerbil-DataFrame"

if [ ! -d "$GERBIL_DIR" ]; then
    echo -e "${BLUE}Cloning gerbil-DataFrame repository...${NC}"
    git clone https://github.com/M-Serajian/gerbil-DataFrame.git "$GERBIL_DIR"
    echo -e "${GREEN}gerbil-DataFrame cloned successfully.${NC}"
else
    echo -e "${GREEN}[INFO] gerbil-DataFrame already exists. Skipping clone.${NC}"
fi

echo -e "${GREEN}gerbil-DataFrame setup complete!${NC}"
