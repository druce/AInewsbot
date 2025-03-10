#!/bin/bash
# load env variables from .env file with export command
export $(grep -v '^#' .env | xargs)
