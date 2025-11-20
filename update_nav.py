import yaml

# Load the mkdocs.yml file
with open('mkdocs.yml', 'r') as f:
    data = yaml.safe_load(f)

# Update the navigation
nav = []
for item in data['nav']:
    if isinstance(item, dict) and 'AI Engineering' in item:
        # Update AI Engineering section
        ai_eng = item['AI Engineering']
        for subitem in ai_eng:
            if isinstance(subitem, dict) and 'LLMOps Healthcare Triage' in subitem:
                # Update the LLMOps Healthcare Triage subsection
                subitem['LLMOps Healthcare Triage'] = [
                    'Part 1: Problem Space: ai-engineering/llmo…',
                    'Part 2: Data Profiling: ai-engineering/llmops-healthcare-triage/part2-data-profiling.md',
                    'Part 3: Bronze Ingestion: ai-engineering/llmops-healthcare-triage/part3-bronze-ingestion.md'
                ]
        nav.append({'AI Engineering': ai_eng})
    else:
        nav.append(item)

# Write back to mkdocs.yml
with open('mkdocs.yml', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
