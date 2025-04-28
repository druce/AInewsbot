#!/bin/bash

# make sure we are in the right directory, yaml and temp paths are relative to this
cd /Users/drucev/projects/windsurf/AInewsbot

# Activate the conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ainewsbot

# Run the Python script
python /Users/drucev/projects/windsurf/AInewsbot/AInewsbot_langgraph.py > AInewsbot.out 2>&1
