#!/bin/bash

#SBATCH -J Tile_extraction
#SBATCH --mem=50G
#SBATCH --partition=highmem,PATHgpu
#SBATCH --time=100:00:00
###SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -D ../../siemen/PathBench/
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=s.brussee@lumc.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# Clear environment
module purge > /dev/null 2>&1

# Load modules
module load library/cuda/11.6.1/gcc.8.3.1
module load library/openslide/3.4.1/gcc-8.3.1
module load system/python/3.9.17
module load tools/miniconda/python3.9/4.12.0

source pathbench_env/bin/activate

echo "Test"
