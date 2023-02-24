import subprocess
from datetime import datetime

changelog = """Changelog
All notable changes to this project will be documented in this file.

[Unreleased]
"""

# Get the Git log
output = subprocess.check_output(['git', 'log', '--pretty=format:%h - %s (%ad)', '--date=format:%Y-%m-%d', '--reverse'])

# Parse the log
tag = None
for line in output.decode('utf-8').split('\n'):
    if '(' in line and ')' in line:
        tag = line[line.index('(')+1:line.index(')')]
    elif not line.startswith('Merge') and line:
        if tag is None:
            print(f'No tag for commit {line}')
            exit(1)
        type_, msg = line.split(' ', 1)
        changelog += f"""
{tag}
{type_}
• {msg.strip()}"""

# Add a link to the GitHub repository
changelog += """

All changes up to this release can be found on [GitHub](https://github.com/user/repo/commits/master).
"""

# Write the changelog to a file
with open('CHANGELOG.md', 'w') as f:
    f.write(changelog)