#!/bin/bash
# Daily YouTube Video Production Workflow Script
# Usage: ./daily-workflow.sh [day_number]

set -e  # Exit on error

DAY_NUMBER=${1:-1}
DATE=$(date +%Y-%m-%d)
VIDEO_FOLDER="day-$(printf "%02d" $DAY_NUMBER)-$(date +%m%d)"

echo "🚀 Starting Day $DAY_NUMBER Video Production Workflow"
echo "Date: $DATE"
echo "Folder: $VIDEO_FOLDER"

# Create directory structure
create_directory_structure() {
    echo "📁 Creating directory structure..."
    mkdir -p "$VIDEO_FOLDER"/{scripts,code,assets,documentation}
    mkdir -p "$VIDEO_FOLDER"/scripts/{english,chinese}
    mkdir -p "$VIDEO_FOLDER"/code/{demos,projects,solutions}
    mkdir -p "$VIDEO_FOLDER"/assets/{images,data}
    mkdir -p "$VIDEO_FOLDER"/documentation
}

# Generate content using Claude Code
generate_content() {
    echo "🤖 Generating content with Claude Code..."
    
    # Generate bilingual scripts
    claude-code "Based on TODO.md Day $DAY_NUMBER, generate complete English YouTube script for the topic, including intro, main content, hands-on demo section, and outro. Target duration: 15 minutes." > "$VIDEO_FOLDER/scripts/english/script.md"
    
    claude-code "Based on TODO.md Day $DAY_NUMBER, generate complete Chinese YouTube script for the topic, including intro, main content, hands-on demo section, and outro. Target duration: 15 minutes." > "$VIDEO_FOLDER/scripts/chinese/script.md"
    
    # Generate demo code
    claude-code "Create comprehensive Python code examples and hands-on projects for Day $DAY_NUMBER topic from TODO.md, including comments, error handling, and beginner-friendly explanations." > "$VIDEO_FOLDER/code/demos/main_demo.py"
    
    # Generate project files
    claude-code "Create a hands-on project for Day $DAY_NUMBER with step-by-step instructions, including requirements.txt, README.md, and sample data if needed." > "$VIDEO_FOLDER/code/projects/project_instructions.md"
    
    # Generate documentation
    claude-code "Create comprehensive README.md for Day $DAY_NUMBER video including learning objectives, prerequisites, setup instructions, code explanations, and additional resources." > "$VIDEO_FOLDER/README.md"
    
    echo "✅ Content generation complete!"
}

# Setup development environment
setup_environment() {
    echo "🔧 Setting up development environment..."
    cd "$VIDEO_FOLDER"
    
    # Create virtual environment if needed
    if [ ! -d "venv" ]; then
        python -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
    else
        source venv/bin/activate
    fi
    
    # Install requirements if they exist
    if [ -f "code/projects/requirements.txt" ]; then
        pip install -r code/projects/requirements.txt
    fi
    
    echo "✅ Environment setup complete!"
}

# Test all code examples
test_code() {
    echo "🧪 Testing all code examples..."
    cd "$VIDEO_FOLDER"
    source venv/bin/activate
    
    # Test demo code
    if [ -f "code/demos/main_demo.py" ]; then
        echo "Testing main demo..."
        python code/demos/main_demo.py
    fi
    
    # Test project code
    for project_file in code/projects/*.py; do
        if [ -f "$project_file" ]; then
            echo "Testing $project_file..."
            python "$project_file"
        fi
    done
    
    echo "✅ Code testing complete!"
}

# Create pre-production checklist
create_checklist() {
    echo "📋 Creating pre-production checklist..."
    
    cat > "$VIDEO_FOLDER/PRE_PRODUCTION_CHECKLIST.md" << EOF
# Day $DAY_NUMBER Pre-Production Checklist

**Date**: $DATE
**Topic**: [Fill from TODO.md]
**Estimated Duration**: 15-18 minutes

## Content Preparation
- [ ] Review English script (scripts/english/script.md)
- [ ] Review Chinese script (scripts/chinese/script.md)
- [ ] Test all demo code (code/demos/)
- [ ] Verify project instructions (code/projects/)
- [ ] Prepare sample data if needed (assets/data/)

## Technical Setup
- [ ] OBS Studio configured and tested
- [ ] Microphone levels checked
- [ ] Screen recording settings verified
- [ ] Lighting setup optimal
- [ ] Backup recording device ready

## Recording Preparation
- [ ] All browser tabs closed except necessary ones
- [ ] Phone on silent mode
- [ ] "Do Not Disturb" mode enabled
- [ ] Water bottle and notes ready
- [ ] Practice run completed (optional)

## Demo Environment
- [ ] Virtual environment activated
- [ ] All packages installed and working
- [ ] Sample data loaded
- [ ] Code examples tested
- [ ] Browser bookmarks for resources ready

## Post-Recording Tasks
- [ ] Raw video files backed up
- [ ] Audio quality verified
- [ ] Screen recordings clear and readable
- [ ] All demo code worked as expected
- [ ] Notes for editing made

EOF

    echo "✅ Pre-production checklist created!"
}

# Generate thumbnail concept
generate_thumbnail_concept() {
    echo "🎨 Generating thumbnail concept..."
    
    claude-code "Suggest 5 compelling YouTube thumbnail concepts for Day $DAY_NUMBER video topic, including text overlay, color scheme, and visual elements that would attract ML beginners." > "$VIDEO_FOLDER/assets/thumbnail_concepts.md"
    
    echo "✅ Thumbnail concepts generated!"
}

# Update main TODO.md progress
update_progress() {
    echo "📊 Updating progress tracking..."
    
    # Mark day as started in TODO.md
    sed -i "s/Day $DAY_NUMBER.*Status.*⏳ Pending/Day $DAY_NUMBER.*Status**: 🚧 In Progress/" TODO.md
    
    # Create GitHub issue for tracking
    gh issue create --title "Day $DAY_NUMBER Video Production" --body "Tracking progress for Day $DAY_NUMBER video production. See $VIDEO_FOLDER/ for all files." --label "video-production" --assignee @me
    
    echo "✅ Progress updated!"
}

# Main workflow
main() {
    create_directory_structure
    generate_content
    setup_environment
    test_code
    create_checklist
    generate_thumbnail_concept
    update_progress
    
    echo ""
    echo "🎉 Day $DAY_NUMBER workflow complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review generated content in: $VIDEO_FOLDER/"
    echo "2. Complete pre-production checklist: $VIDEO_FOLDER/PRE_PRODUCTION_CHECKLIST.md"
    echo "3. Record English version"
    echo "4. Record Chinese version"
    echo "5. Run post-production workflow: ./post-production.sh $DAY_NUMBER"
    echo ""
    echo "📁 Generated files:"
    find "$VIDEO_FOLDER" -type f | head -20
}

# Run main function
main
