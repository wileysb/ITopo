#!/usr/bin/env bash

parallel -j 8 ./gen_sunview.py {1}::: 0 3 6 9 12 15 18 21
