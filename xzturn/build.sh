#!/bin/bash

cd ..
git pull --recurse-submodules upstream master

sudo -H pip install --upgrade $1
