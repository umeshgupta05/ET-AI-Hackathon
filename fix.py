import os
import re

css_vars = """:root {
  --ink-950: #0b0f17;
  --ink-850: #131926;
  --ink-750: #1c2436;
  --hairline: #2a3346;
  --text-primary: #e9ebf1;
  --text-muted: #8a93a8;
  --signal-alert: #d64545;
  --signal-caution: #d69a3c;
  --signal-clear: #3fa37e;

  --brand-orange: var(--signal-alert);
  --brand-orange-dark: #a83636;
  --surface-cream: var(--ink-850);
  --surface-wheat: var(--ink-750);
  --surface-card: var(--ink-850);
}"""

def process_css(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace :root
    if ':root {' in content:
        content = re.sub(r':root\s*\{[^}]+\}', css_vars, content, count=1)
    
    # 2. Replace border-radius: 999px with 6px
    content = content.replace('border-radius: 999px', 'border-radius: 6px')
    
    # 3. Replace box-shadow with border
    # But wait, we shouldn't replace *all* box-shadows if they already have borders, 
    # but the prompt says "Remove/neutralize box-shadow declarations in favor of border: 1px solid var(--hairline)."
    content = re.sub(r'box-shadow:\s*[^;]+;', 'box-shadow: none; border: 1px solid var(--hairline);', content)

    # 4. Replace hardcoded hexes
    content = content.replace('#f47c20', 'var(--brand-orange)')
    content = content.replace('#fff8ea', 'var(--surface-cream)')
    content = content.replace('#fff7e8', 'var(--ink-750)')
    content = content.replace('#3b2a1d', 'var(--ink-950)')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

process_css('frontend/src/index.css')
process_css('frontend/src/CommandCentre.css')
process_css('frontend/src/App.jsx')

# FIX 2: gitignore updates
def update_gitignore(path):
    if not os.path.exists(path):
        return
    with open(path, 'a', encoding='utf-8') as f:
        f.write("\n# Runtime data (any location, not just backend/data/)\n**/data/*.db\n**/data/*.sqlite*\n**/data/app.db\n**/data/operational_intelligence.db\n**/data/realtime_safety.db\n")

update_gitignore('.gitignore')
update_gitignore('backend/.gitignore')

# FIX 3: Add LICENSE
with open('LICENSE', 'w', encoding='utf-8') as f:
    f.write('''MIT License

Copyright (c) 2026 ET AI Hackathon Participant

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
''')

# Update README for FIX 1 and 3
readme_path = 'README.md'
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Add line to Backend Setup
    if '## Backend Setup' in content:
        content = content.replace('## Backend Setup', '## Backend Setup\n\n> Requires PostgreSQL 16+ running locally, or `docker compose up -d postgres`.\n> Connection settings are in `.env.example` (`DATABASE_URL`).\n')
    
    # Fix 3: Note in README about License
    content += "\n## License\nThis project is licensed under the MIT License. Note that the Kaggle-sourced currency training dataset carries its own CC BY-NC-SA 4.0 license, as disclosed in `training_metadata.json`.\n"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Fix 1: compose.yaml
compose_path = 'compose.yaml'
if os.path.exists(compose_path):
    with open(compose_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'postgres:' not in content:
        if 'services:' in content:
            postgres_service = """
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-postgres}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: ${POSTGRES_DB:-shield_db}
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-postgres}"]
      interval: 10s
      timeout: 5s
      retries: 6
    restart: unless-stopped
"""
            content = content.replace('services:\n', f'services:\n{postgres_service}')
            
            if 'volumes:' in content:
                content += "\n  postgres-data:\n"
            else:
                content += "\nvolumes:\n  postgres-data:\n"
                
            with open(compose_path, 'w', encoding='utf-8') as f:
                f.write(content)

print("Fixes applied.")
