#!/bin/bash

cd ..
git pull --recurse-submodules upstream master

sudo -H pip3 install --upgrade $1
