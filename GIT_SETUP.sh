# =====================================================
# GIT SETUP GUIDE FOR SANA & AQSA
# Follow these steps IN ORDER
# =====================================================

# -------------------------------------------------------
# FIRST TIME SETUP (Sana does this — creates the repo)
# -------------------------------------------------------

# Step 1: Go inside your project folder
cd PKR-Fake-Currency-Detection

# Step 2: Initialize git
git init

# Step 3: Add your name and email to git (one time only)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Step 4: Add all files
git add .

# Step 5: First commit
git commit -m "Initial project structure"

# Step 6: Go to GitHub.com → New Repository
# Name it: PKR-Fake-Currency-Detection
# Make it Public
# DO NOT check "Add README" (we already have one)

# Step 7: Connect your local project to GitHub
git remote add origin https://github.com/YOUR_USERNAME/PKR-Fake-Currency-Detection.git

# Step 8: Push to GitHub
git branch -M main
git push -u origin main


# -------------------------------------------------------
# AQSA JOINS THE REPO
# -------------------------------------------------------

# Step 1: Clone the repo to your computer
git clone https://github.com/SANA_USERNAME/PKR-Fake-Currency-Detection.git

# Step 2: Enter the folder
cd PKR-Fake-Currency-Detection

# Step 3: Create your own branch (IMPORTANT — don't work on main)
git checkout -b aqsa-branch


# -------------------------------------------------------
# DAILY WORKFLOW FOR BOTH (do this every day)
# -------------------------------------------------------

# --- Before starting work ---
# Pull latest changes from GitHub
git pull origin main

# --- After finishing your work ---

# Step 1: See what files you changed
git status

# Step 2: Add your changes
git add .

# Step 3: Commit with a message describing what you did
git commit -m "Added Grad-CAM visualization function"

# Step 4: Push to your branch
git push origin aqsa-branch     # Aqsa uses this
git push origin sana-branch     # Sana uses this


# -------------------------------------------------------
# MERGING WORK (when both are done with a feature)
# -------------------------------------------------------

# On GitHub: Create a Pull Request to merge branch into main
# Other person reviews and approves
# Merge on GitHub


# -------------------------------------------------------
# USEFUL GIT COMMANDS
# -------------------------------------------------------

git status              # see what changed
git log --oneline       # see commit history
git diff                # see exact line changes
git stash               # temporarily save changes
git stash pop           # bring back stashed changes


# -------------------------------------------------------
# RECOMMENDED BRANCH STRUCTURE
# -------------------------------------------------------
# main          → final working code only
# sana-branch   → Sana's work (dataset, augmentation, CNN)
# aqsa-branch   → Aqsa's work (Grad-CAM, Streamlit app)
