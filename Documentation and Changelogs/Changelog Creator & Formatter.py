# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 2022
Modal_tester V1.0.0
Author: Adill Al-Ashgar
University of Bristol

### NOTE to user.
# Must first run 
# git log --graph --date=short --full-history --all --pretty=format:"%x09%d%x20%ad%x20%s%x20[%an]" > "Git Tree.txt
# command in terminal to generate gitlog in appropriate formatting
# then run this autoformatter to take the raw git log and print it in my custom changeiog format.
"""

file_location = "Git Tree.txt" #"path/to/file.txt"
full_output_path = "Git Tree_Formatted.txt" #"path/to/output_file.txt"

#%% - Auto Parser & Fomatter
import re
from datetime import datetime

def create_changelog(file_location):
    # Read the file
    with open(file_location, "r") as file:
        data = file.read()

    # Extract relevant data using regex
    pattern = r"\*.*\[(.*)\]\n\*{1,2}\s*(\d{4}-\d{2}-\d{2})\s*(.*)"
    matches = re.findall(pattern, data)

    # Group matches by date
    grouped_matches = {}
    for match in matches:
        date = match[1]
        if date not in grouped_matches:
            grouped_matches[date] = {"Added": [], "Changed": [], "Deprecated": [], "Removed": [], "Fixed": []}

        message = match[2]
        if "add" or "added" or "built" or "created" in message.lower():
            grouped_matches[date]["Added"].append(message.strip())
        elif "change" in message.lower():
            grouped_matches[date]["Changed"].append(message.strip())
        elif "deprecate" or "archive" or "archived" in message.lower():
            grouped_matches[date]["Deprecated"].append(message.strip())
        elif "remove" or "deleted" in message.lower():
            grouped_matches[date]["Removed"].append(message.strip())
        elif "fix" in message.lower():
            grouped_matches[date]["Fixed"].append(message.strip())

    # Format grouped matches into a changelog
    changelog = "Changelog\nAll notable changes to this project will be documented in this file.\n[Unreleased]\n\n"
    for date, changes in sorted(grouped_matches.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'), reverse=True):
        changelog += f"[{date}]\n"
        if changes["Added"]:
            changelog += "Added\n"
            for item in changes["Added"]:
                changelog += f"• {item}\n"
            changelog += "\n"
        if changes["Changed"]:
            changelog += "Changed\n"
            for item in changes["Changed"]:
                changelog += f"• {item}\n"
            changelog += "\n"
        if changes["Deprecated"]:
            changelog += "Deprecated\n"
            for item in changes["Deprecated"]:
                changelog += f"• {item}\n"
            changelog += "\n"
        if changes["Removed"]:
            changelog += "Removed\n"
            for item in changes["Removed"]:
                changelog += f"• {item}\n"
            changelog += "\n"
        if changes["Fixed"]:
            changelog += "Fixed\n"
            for item in changes["Fixed"]:
                changelog += f"• {item}\n"
            changelog += "\n"

    return changelog


changelog = create_changelog(file_location)


# Write changlog to .txt file on disk with with loop file handling
with open(full_output_path, "w") as file:
    file.write(changelog)
