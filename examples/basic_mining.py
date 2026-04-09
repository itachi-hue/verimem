#!/usr/bin/env python3
"""Example: mine a project folder into a VeriMem store."""

import sys

project_dir = sys.argv[1] if len(sys.argv) > 1 else "~/projects/my_app"
print("Step 1: Initialize rooms from folder structure")
print(f"  verimem init {project_dir}")
print("\nStep 2: Mine everything")
print(f"  verimem mine {project_dir}")
print("\nStep 3: Search")
print("  verimem search 'why did we choose this approach'")
