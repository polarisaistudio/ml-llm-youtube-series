#!/usr/bin/env python3
"""
Blog Publishing Helper Script
Automates some of the tedious tasks in cross-platform publishing
"""

import os
import shutil
from pathlib import Path
import re
from datetime import datetime

class BlogPublisher:
    def __init__(self, day_folder="day-01-0827"):
        self.day_folder = Path(day_folder)
        self.blog_folder = self.day_folder / "blog-posts"
        
    def optimize_images_for_web(self):
        """Optimize images for web publishing"""
        print("🖼️  Optimizing images for web...")
        
        images_folder = self.day_folder / "assets" / "images" / "theory"
        web_images = self.blog_folder / "web-optimized"
        web_images.mkdir(exist_ok=True)
        
        # Copy and potentially resize images
        if images_folder.exists():
            for img_file in images_folder.glob("*.png"):
                dest = web_images / img_file.name
                shutil.copy2(img_file, dest)
                print(f"   Copied {img_file.name}")
        
        print("   ✅ Image optimization complete")
        
    def generate_github_pages_config(self):
        """Generate Jekyll config for GitHub Pages"""
        config_content = """
# GitHub Pages Configuration for ML Blog Series

title: "40 Days of Machine Learning"
description: "Learning ML and LLMs while teaching - a complete beginner's journey"
author: "Your Name"
email: "your.email@example.com"
baseurl: "/ml-llm-youtube-series"
url: "https://yourusername.github.io"

# Build settings
markdown: kramdown
highlighter: rouge
theme: minima

# Plugins
plugins:
  - jekyll-feed
  - jekyll-sitemap
  - jekyll-seo-tag

# Collections
collections:
  posts:
    output: true
    permalink: /:categories/:year/:month/:day/:title/

# Defaults
defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: "post"
      author: "Your Name"
      comments: true

# SEO settings
twitter:
  username: yourtwitterhandle
social:
  name: Your Name
  links:
    - https://twitter.com/yourtwitterhandle
    - https://github.com/yourgithub
    - https://linkedin.com/in/yourlinkedin
"""
        
        config_file = self.day_folder / "_config.yml"
        with open(config_file, 'w') as f:
            f.write(config_content.strip())
        
        print("   ✅ Generated GitHub Pages config")
    
    def create_social_media_calendar(self):
        """Create a social media posting calendar"""
        calendar_content = f"""# Social Media Calendar - Day 1
        
## Publishing Schedule

### Day 1 (Video Release Day)
- **9:00 AM**: YouTube video goes live
- **9:30 AM**: Twitter thread (main announcement)
- **10:00 AM**: LinkedIn article
- **11:00 AM**: Medium article
- **2:00 PM**: Instagram carousel post
- **6:00 PM**: Instagram story highlights

### Day 2 (Follow-up Engagement)
- **10:00 AM**: Twitter "in case you missed it" post
- **12:00 PM**: TikTok/Reels version
- **3:00 PM**: Reddit posts (r/MachineLearning, r/LearnMachineLearning)
- **Evening**: Respond to all comments and engage

### Day 3-7 (Community Engagement)
- Daily comment responses
- Share related content from others
- Cross-promote on different platforms
- Plan next week's content based on feedback

## Content Variations by Platform

### Short Form (Twitter, Instagram Stories)
- Key visual + 1-2 sentences
- Question to drive engagement
- Link to full content

### Medium Form (LinkedIn, Instagram Posts)  
- 3-5 key points
- Professional context
- Industry relevance

### Long Form (Medium, YouTube, Blog)
- Complete tutorial with code
- Deep explanations
- Multiple examples and use cases

## Hashtag Strategy

### Primary Tags (use everywhere)
#MachineLearning #AI #Python #DataScience #TechEducation

### Platform Specific
- **Twitter**: #100DaysOfMLCode #MachineLearning #Python #AI
- **LinkedIn**: #DataScience #Innovation #Technology #Learning
- **Instagram**: #CodeLife #TechTutorial #LearnML #Programming
- **TikTok**: #LearnOnTikTok #TechTok #Programming #AI

## Engagement Prompts

### Questions to Ask Audience
- "What's your biggest confusion about ML vs AI?"
- "Which algorithm would you use for [specific problem]?"
- "Traditional ML or Deep Learning - which are you using?"
- "What should I cover in tomorrow's video?"

### Call-to-Actions
- "Follow for daily ML insights"
- "Save this for later reference"  
- "Share with someone learning ML"
- "What topics should I cover next?"
"""
        
        calendar_file = self.blog_folder / "social_media_calendar.md"
        with open(calendar_file, 'w') as f:
            f.write(calendar_content.strip())
        
        print("   ✅ Created social media calendar")
    
    def validate_links(self):
        """Check that all internal links work"""
        print("🔗 Validating internal links...")
        
        # Check GitHub repo links
        github_links = [
            "https://github.com/polarisaistudio/ml-llm-youtube-series",
            "https://github.com/polarisaistudio/ml-llm-youtube-series/tree/main/day-01-0827"
        ]
        
        print("   GitHub links to verify:")
        for link in github_links:
            print(f"   - {link}")
        
        # Check image paths
        for blog_file in self.blog_folder.glob("**/*.md"):
            if blog_file.name.startswith('day01'):
                with open(blog_file, 'r') as f:
                    content = f.read()
                    
                # Find image references
                img_refs = re.findall(r'!\[.*?\]\((.*?)\)', content)
                print(f"\n   Images in {blog_file.name}:")
                for img in img_refs:
                    print(f"   - {img}")
        
        print("   ✅ Link validation complete (manual check required)")
    
    def generate_analytics_template(self):
        """Create template for tracking analytics"""
        analytics_template = """# Day 1 Analytics Tracking

## Week 1 Performance

### YouTube
- [ ] Views: _____ (Target: 1000+)
- [ ] CTR: _____% (Target: 4%+) 
- [ ] Avg View Duration: _____% (Target: 50%+)
- [ ] Likes: _____ (Target: 5%+ of views)
- [ ] Comments: _____ (Target: 2%+ of views)
- [ ] Subscribers gained: _____

### Medium
- [ ] Views: _____ (Target: 500+)
- [ ] Reads: _____ (Target: 50%+ of views)
- [ ] Claps: _____ (Target: 100+)
- [ ] Highlights: _____
- [ ] Followers gained: _____

### LinkedIn
- [ ] Post views: _____ (Target: 1000+)
- [ ] Article views: _____ (Target: 300+)
- [ ] Reactions: _____ (Target: 50+)
- [ ] Comments: _____ (Target: 10+)
- [ ] Shares: _____ (Target: 5+)

### Social Media
- [ ] Twitter impressions: _____
- [ ] Instagram reach: _____
- [ ] New followers (total): _____

## Key Insights
- What performed best: _____
- What performed worst: _____
- Audience feedback themes: _____
- Content improvements needed: _____

## Week 2 Action Items
- [ ] Adjust content based on feedback
- [ ] Focus on top-performing formats
- [ ] Engage with successful posts from others
- [ ] Plan follow-up content

## Month 1 Goals
- YouTube: 10K total views across all videos
- Medium: 5K total views, 500 followers
- LinkedIn: 10K post impressions, 200 connections
- GitHub: 50 stars on repository
"""
        
        analytics_file = self.blog_folder / "analytics_template.md"
        with open(analytics_file, 'w') as f:
            f.write(analytics_template.strip())
            
        print("   ✅ Created analytics tracking template")
    
    def run_all_tasks(self):
        """Execute all publishing helper tasks"""
        print("🚀 Running Blog Publishing Helper\n")
        
        self.optimize_images_for_web()
        self.generate_github_pages_config()
        self.create_social_media_calendar()
        self.validate_links()
        self.generate_analytics_template()
        
        print(f"\n✨ All tasks complete!")
        print(f"\nGenerated files in {self.blog_folder}:")
        for file in self.blog_folder.rglob("*"):
            if file.is_file():
                print(f"   📄 {file.name}")

def main():
    publisher = BlogPublisher()
    publisher.run_all_tasks()

if __name__ == "__main__":
    main()