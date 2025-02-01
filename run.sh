#!/bin/bash

# Run script
python3 DetectorTemplate/main_detector.py

if [[ "$(whoami)" != "hamzarashid" ]]
 then python3 email_results.py
 else echo "testing done"
fi
