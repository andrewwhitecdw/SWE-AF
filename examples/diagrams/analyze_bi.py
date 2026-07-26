#!/usr/bin/env python3
"""
BI-Quality Pipeline Analysis
=============================
Generates insight-driven visualizations for autonomous SWE pipeline execution.
Every chart answers a specific question at a glance.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_DIR = Path(
    "/Users/santoshkumarradha/Documents/agentfield/code/int-agentfield-examples/"
    "af-swe/example-diagrams/.artifacts/logs"
)
NB_PATH = Path(
    "/Users/santoshkumarradha/Documents/agentfield/code/int-agentfield-examples/"
    "af-swe/example-diagrams/pipeline_deep_analysis.ipynb"
)
CHART_DIR = Path(
    "/Users/santoshkumarradha/Documents/agentfield/code/int-agentfield-examples/"
    "af-swe/example-diagrams/charts"
)

# ---------------------------------------------------------------------------
# Parse all logs
# ---------------------------------------------------------------------------
def classify(stem):
    if stem in ('product_manager', 'architect', 'tech_lead', 'sprint_planner'):
        return 'planning', stem, None
    if stem.startswith('issue_writer_'):
        return 'issue_writer', stem[len('issue_writer_'):], None
    m = re.match(r'coder_(.+)_iter_(\d+)', stem)
    if m: return 'coder', m.group(1), int(m.group(2))
    m = re.match(r'reviewer_(.+)_iter_([a-f0-9]+)', stem)
    if m: return 'reviewer', m.group(1), m.group(2)
    m = re.match(r'qa_(.+)_iter_([a-f0-9]+)', stem)
    if m: return 'qa', m.group(1), m.group(2)
    m = re.match(r'synthesizer_(.+)_iter_([a-f0-9]+)', stem)
    if m: return 'synthesizer', m.group(1), m.group(2)
    m = re.match(r'merger_level_(\d+)', stem)
    if m: return 'merger', f'level_{m.group(1)}', int(m.group(1))
    m = re.match(r'integration_tester_level_(\d+)', stem)
    if m: return 'integration_tester', f'level_{m.group(1)}', int(m.group(1))
    m = re.match(r'workspace_(setup|cleanup)_level_(\d+)', stem)
    if m: return f'workspace', f'{m.group(1)}_level_{m.group(2)}', None
    return 'other', stem, None


def parse_all():
    records = []
    for lf in sorted(LOG_DIR.glob('*.jsonl')):
        events = []
        with open(lf) as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not events:
            continue

        cat, issue, iter_num = classify(lf.stem)
        first_ts = events[0].get('ts', 0)
        last_ts = events[-1].get('ts', 0)

        cost = 0.0
        num_turns = 0
        duration_ms = 0
        model = events[0].get('model', 'unknown') if events[0].get('event') == 'start' else 'unknown'
        is_error = False

        for e in events:
            if e.get('event') == 'result':
                cost = e.get('cost_usd', 0) or 0
                num_turns = e.get('num_turns', 0) or 0
                duration_ms = e.get('duration_ms', 0) or 0
            if e.get('event') == 'end':
                is_error = e.get('is_error', False)

        tool_calls = 0
        text_chars = 0
        text_blocks = 0
        for e in events:
            if e.get('event') == 'assistant':
                for c in e.get('content', []):
                    if isinstance(c, dict):
                        if c.get('type') == 'tool_use':
                            tool_calls += 1
                        if c.get('type') == 'text':
                            text_chars += len(c.get('text', ''))
                            text_blocks += 1

        # Estimate tokens from cost (Sonnet pricing: $3/M input, $15/M output)
        # Assume ~80% input, 20% output by cost
        # input_cost + output_cost = total_cost
        # input_tokens * 3/1M + output_tokens * 15/1M = cost
        # Rough: output_tokens ≈ text_chars/4, then input_tokens from remainder
        est_output_tokens = text_chars / 4 if text_chars else 0
        est_output_cost = est_output_tokens * 15 / 1_000_000
        est_input_cost = max(0, cost - est_output_cost)
        est_input_tokens = est_input_cost * 1_000_000 / 3 if est_input_cost > 0 else 0

        records.append({
            'file': lf.name,
            'category': cat,
            'issue': issue,
            'iter': iter_num,
            'start_ts': first_ts,
            'end_ts': last_ts,
            'duration_s': last_ts - first_ts,
            'duration_ms': duration_ms,
            'cost': cost,
            'num_turns': num_turns,
            'tool_calls': tool_calls,
            'text_chars': text_chars,
            'text_blocks': text_blocks,
            'est_input_tokens': est_input_tokens,
            'est_output_tokens': est_output_tokens,
            'model': model,
            'is_error': is_error,
        })

    return records


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------
def build_notebook(records):
    import nbformat  # optional dependency, only needed for notebook generation

    nb = nbformat.v4.new_notebook()
    data_json = json.dumps(records, indent=2)

    def md(src):
        nb.cells.append(nbformat.v4.new_markdown_cell(src))

    def code(src):
        nb.cells.append(nbformat.v4.new_code_cell(src))

    # ── Title ──
    md("""# Autonomous SWE Pipeline — Execution Intelligence

**Pipeline:** 1-line goal → PRD → Architecture → Issue Decomposition → Parallel Coding → Review → QA → Merge → Integration Test

**Project:** Diagrams-as-code CLI in Rust (DSL → SVG + ASCII preview)

Each visualization answers a specific question about pipeline behavior.""")

    # ── Data + Setup ──
    code(f"""import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import squarify
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Embedded data
records = json.loads('''{data_json}''')
df = pd.DataFrame(records)

# Derived columns
df['duration_min'] = df['duration_s'] / 60
df['est_total_tokens'] = df['est_input_tokens'] + df['est_output_tokens']
df['cost_per_min'] = df['cost'] / df['duration_min'].replace(0, np.nan)
df['tools_per_turn'] = df['tool_calls'] / df['num_turns'].replace(0, np.nan)

# Pipeline start reference
t0 = df['start_ts'].min()
df['start_offset_min'] = (df['start_ts'] - t0) / 60
df['end_offset_min'] = (df['end_ts'] - t0) / 60

# Nice category labels
CAT_LABELS = {{
    'planning': 'Planning',
    'issue_writer': 'Issue Writing',
    'coder': 'Coding',
    'reviewer': 'Code Review',
    'qa': 'QA Testing',
    'synthesizer': 'Synthesis',
    'merger': 'Branch Merging',
    'integration_tester': 'Integration Test',
    'workspace': 'Workspace Ops',
}}
df['cat_label'] = df['category'].map(CAT_LABELS).fillna(df['category'])

# Color map — warm = expensive, cool = cheap
CAT_PALETTE = {{
    'Coding': '#E74C3C',
    'QA Testing': '#E67E22',
    'Code Review': '#F1C40F',
    'Integration Test': '#9B59B6',
    'Issue Writing': '#3498DB',
    'Planning': '#1ABC9C',
    'Branch Merging': '#2ECC71',
    'Workspace Ops': '#95A5A6',
    'Synthesis': '#BDC3C7',
}}

# Global style
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

import os
os.makedirs('charts', exist_ok=True)

print(f"Loaded {{len(df)}} agent executions across {{df['category'].nunique()}} categories")
print(f"Total pipeline cost: ${{df['cost'].sum():.2f}}")
print(f"Total wall time: {{(df['end_ts'].max() - df['start_ts'].min()) / 60:.1f}} minutes")
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 1: Where does the money go? (Treemap)
    # ══════════════════════════════════════════════════════════════════════
    md("""## 1. Where Does the Money Go?

A treemap reveals **proportional cost allocation** at a glance. Larger rectangles = more spend. This immediately shows whether the system is spending on *thinking* (planning) or *doing* (coding/QA).""")

    code("""# Treemap: cost allocation by category
cat_cost = df.groupby('cat_label')['cost'].sum().sort_values(ascending=False)
cat_cost = cat_cost[cat_cost > 0]

labels = [f"{cat}\\n${cost:.2f}\\n({cost/cat_cost.sum()*100:.0f}%)"
          for cat, cost in cat_cost.items()]
colors = [CAT_PALETTE.get(cat, '#95A5A6') for cat in cat_cost.index]

fig, ax = plt.subplots(figsize=(14, 8))
squarify.plot(sizes=cat_cost.values, label=labels, color=colors,
              alpha=0.85, edgecolor='white', linewidth=3, text_kwargs={'fontsize': 11, 'fontweight': 'bold'}, ax=ax)
ax.set_title('Pipeline Cost Allocation — Where Does the Money Go?',
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

# Annotation
total = cat_cost.sum()
ax.text(0.99, -0.02, f'Total pipeline cost: ${total:.2f}',
        transform=ax.transAxes, ha='right', fontsize=11, color='#666')
plt.tight_layout()
plt.savefig('charts/01_cost_treemap.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 2: Where does the time go? (Treemap)
    # ══════════════════════════════════════════════════════════════════════
    md("""## 2. Where Does the Time Go?

Same treemap view, but for **cumulative agent-minutes**. Compare with cost treemap — are we paying proportionally for time, or are some phases disproportionately expensive?""")

    code("""# Treemap: time allocation by category
cat_time = df.groupby('cat_label')['duration_min'].sum().sort_values(ascending=False)
cat_time = cat_time[cat_time > 0]

labels = [f"{cat}\\n{dur:.1f} min\\n({dur/cat_time.sum()*100:.0f}%)"
          for cat, dur in cat_time.items()]
colors = [CAT_PALETTE.get(cat, '#95A5A6') for cat in cat_time.index]

fig, ax = plt.subplots(figsize=(14, 8))
squarify.plot(sizes=cat_time.values, label=labels, color=colors,
              alpha=0.85, edgecolor='white', linewidth=3, text_kwargs={'fontsize': 11, 'fontweight': 'bold'}, ax=ax)
ax.set_title('Pipeline Time Allocation — Where Does the Time Go?',
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')

total_min = cat_time.sum()
ax.text(0.99, -0.02, f'Total agent-minutes: {total_min:.0f} min (wall time: {(df["end_ts"].max() - df["start_ts"].min())/60:.0f} min)',
        transform=ax.transAxes, ha='right', fontsize=11, color='#666')
plt.tight_layout()
plt.savefig('charts/02_time_treemap.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 3: Cost vs Time Efficiency (Bubble chart)
    # ══════════════════════════════════════════════════════════════════════
    md("""## 3. Cost Efficiency — Who Costs the Most Per Minute?

Bubble chart: each bubble is an agent category. X = total time, Y = total cost, size = number of agents. The **slope** of each point from the origin reveals cost-per-minute. Points above the diagonal are expensive per minute; below are cheap.""")

    code("""cat_agg = df.groupby('cat_label').agg(
    total_cost=('cost', 'sum'),
    total_min=('duration_min', 'sum'),
    count=('file', 'count'),
    avg_cost=('cost', 'mean'),
).reset_index()
cat_agg = cat_agg[cat_agg['total_cost'] > 0]

fig, ax = plt.subplots(figsize=(12, 8))

for _, row in cat_agg.iterrows():
    color = CAT_PALETTE.get(row['cat_label'], '#95A5A6')
    ax.scatter(row['total_min'], row['total_cost'],
               s=row['count'] * 80, color=color, alpha=0.75,
               edgecolors='white', linewidth=2, zorder=3)
    ax.annotate(f"{row['cat_label']}\\n({row['count']} agents)",
                (row['total_min'], row['total_cost']),
                textcoords="offset points", xytext=(12, 5),
                fontsize=9, fontweight='bold')

# Reference lines for cost/min rates
max_t = cat_agg['total_min'].max() * 1.2
for rate, label in [(0.15, '$0.15/min'), (0.30, '$0.30/min')]:
    ax.plot([0, max_t], [0, rate * max_t], '--', color='#ccc', alpha=0.6, zorder=1)
    ax.text(max_t * 0.95, rate * max_t * 0.95, label, color='#999', fontsize=8, ha='right')

ax.set_xlabel('Total Agent-Minutes', fontsize=12)
ax.set_ylabel('Total Cost ($)', fontsize=12)
ax.set_title('Cost Efficiency by Category — Bubble Size = Agent Count',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, None)
ax.set_ylim(0, None)
sns.despine()
plt.tight_layout()
plt.savefig('charts/03_cost_efficiency_bubble.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 4: Burn rate — cumulative cost over time
    # ══════════════════════════════════════════════════════════════════════
    md("""## 4. Burn Rate — How Fast Are We Spending?

Cumulative cost plotted against wall-clock time. Steep slopes = expensive parallel work. Flat regions = sequential bottleneck or cheap operations. This reveals the **economic rhythm** of the pipeline.""")

    code("""# Build timeline of cost events
events = []
for _, row in df.iterrows():
    if row['cost'] > 0:
        events.append((row['start_offset_min'], 0, row['cat_label']))  # start
        events.append((row['end_offset_min'], row['cost'], row['cat_label']))  # cost realized at end

# Sort by time, accumulate
events.sort(key=lambda x: x[0])
times = []
cumcost = []
running = 0
for t, c, _ in events:
    running += c
    times.append(t)
    cumcost.append(running)

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(times, cumcost, alpha=0.15, color='#E74C3C')
ax.plot(times, cumcost, color='#E74C3C', linewidth=2.5, zorder=3)

# Mark phases
phase_markers = [
    ('Planning', 0, '#1ABC9C'),
    ('Issue Writing', df[df['category']=='issue_writer']['start_offset_min'].min() if len(df[df['category']=='issue_writer']) > 0 else 0, '#3498DB'),
    ('Coding Begins', df[df['category']=='coder']['start_offset_min'].min() if len(df[df['category']=='coder']) > 0 else 0, '#E74C3C'),
]
for label, t, color in phase_markers:
    if t > 0:
        ax.axvline(t, color=color, linestyle='--', alpha=0.5, zorder=2)
        ax.text(t + 0.3, running * 0.1, label, rotation=90, fontsize=9, color=color, va='bottom')

ax.set_xlabel('Wall-Clock Time (minutes)', fontsize=12)
ax.set_ylabel('Cumulative Cost ($)', fontsize=12)
ax.set_title('Pipeline Burn Rate — Cumulative Cost Over Time', fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
sns.despine()
plt.tight_layout()
plt.savefig('charts/04_burn_rate.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 5: Parallelism over time
    # ══════════════════════════════════════════════════════════════════════
    md("""## 5. Parallelism — How Many Agents Run Simultaneously?

A step plot showing concurrent agent count over wall-clock time. Peaks reveal where the pipeline exploits parallelism. Valleys reveal sequential bottlenecks. The **area under the curve** is total agent-minutes.""")

    code("""# Build start/end events
timeline = []
for _, row in df.iterrows():
    timeline.append((row['start_offset_min'], +1, row['cat_label']))
    timeline.append((row['end_offset_min'], -1, row['cat_label']))
timeline.sort(key=lambda x: x[0])

times = []
concurrency = []
current = 0
for t, delta, _ in timeline:
    current += delta
    times.append(t)
    concurrency.append(current)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(times, concurrency, step='post', alpha=0.3, color='#3498DB')
ax.step(times, concurrency, where='post', color='#2C3E50', linewidth=1.5)

peak = max(concurrency)
peak_t = times[concurrency.index(peak)]
ax.annotate(f'Peak: {peak} concurrent agents',
            xy=(peak_t, peak), xytext=(peak_t + 5, peak + 1),
            arrowprops=dict(arrowstyle='->', color='#E74C3C'),
            fontsize=11, fontweight='bold', color='#E74C3C')

ax.set_xlabel('Wall-Clock Time (minutes)', fontsize=12)
ax.set_ylabel('Concurrent Agents', fontsize=12)
ax.set_title('Pipeline Parallelism Over Time', fontsize=14, fontweight='bold')
ax.set_ylim(0, None)
sns.despine()
plt.tight_layout()
plt.savefig('charts/05_parallelism.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 6: Issue heatmap — duration by phase
    # ══════════════════════════════════════════════════════════════════════
    md("""## 6. Issue × Phase Heatmap — Where Do Issues Get Stuck?

Each cell shows time spent (minutes) for a specific issue in a specific pipeline phase. Hot cells reveal bottlenecks. This answers: *which issues and which phases are the most expensive?*""")

    code("""# Build issue × phase matrix
coding_cats = ['coder', 'reviewer', 'qa', 'synthesizer']
coding_df = df[df['category'].isin(coding_cats)].copy()

# Extract base issue name (remove iter suffixes for grouping)
issue_phase = coding_df.groupby(['issue', 'category'])['duration_min'].sum().unstack(fill_value=0)

# Reorder columns
col_order = [c for c in ['coder', 'reviewer', 'qa', 'synthesizer'] if c in issue_phase.columns]
issue_phase = issue_phase[col_order]

# Rename for display
issue_phase.columns = [c.title() for c in issue_phase.columns]

# Sort by total time
issue_phase['_total'] = issue_phase.sum(axis=1)
issue_phase = issue_phase.sort_values('_total', ascending=True)
issue_phase = issue_phase.drop('_total', axis=1)

fig, ax = plt.subplots(figsize=(10, max(8, len(issue_phase) * 0.5)))
sns.heatmap(issue_phase, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=1, linecolor='white', cbar_kws={'label': 'Minutes'},
            ax=ax)
ax.set_title('Time Spent per Issue × Phase (minutes)', fontsize=14, fontweight='bold')
ax.set_xlabel('Pipeline Phase', fontsize=12)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('charts/06_issue_phase_heatmap.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 7: Issue heatmap — cost by phase
    # ══════════════════════════════════════════════════════════════════════
    md("""## 7. Issue × Phase Cost Heatmap — Where Does Each Issue Spend Money?

Same structure as above, but showing **cost** instead of time. Differences between time and cost heatmaps reveal which phases are expensive per minute.""")

    code("""# Build issue × phase cost matrix
issue_cost = coding_df.groupby(['issue', 'category'])['cost'].sum().unstack(fill_value=0)
col_order = [c for c in ['coder', 'reviewer', 'qa', 'synthesizer'] if c in issue_cost.columns]
issue_cost = issue_cost[col_order]
issue_cost.columns = [c.title() for c in issue_cost.columns]

issue_cost['_total'] = issue_cost.sum(axis=1)
issue_cost = issue_cost.sort_values('_total', ascending=True)
issue_cost = issue_cost.drop('_total', axis=1)

fig, ax = plt.subplots(figsize=(10, max(8, len(issue_cost) * 0.5)))
sns.heatmap(issue_cost, annot=True, fmt='$.2f', cmap='YlOrRd',
            linewidths=1, linecolor='white', cbar_kws={'label': 'Cost ($)'},
            ax=ax)
ax.set_title('Cost per Issue × Phase ($)', fontsize=14, fontweight='bold')
ax.set_xlabel('Pipeline Phase', fontsize=12)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('charts/07_issue_cost_heatmap.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 8: Phase duration distributions (violin)
    # ══════════════════════════════════════════════════════════════════════
    md("""## 8. Phase Duration Distributions — How Variable Is Each Phase?

Violin plots show the **full distribution** of durations within each pipeline phase across all issues. Wide violins = high variability (unpredictable). Narrow = consistent. The white dot is the median.""")

    code("""phase_order = ['Coding', 'QA Testing', 'Code Review', 'Issue Writing', 'Planning',
               'Branch Merging', 'Integration Test', 'Synthesis', 'Workspace Ops']
phase_df = df[df['cat_label'].isin(phase_order)].copy()

fig, ax = plt.subplots(figsize=(14, 7))
order = [p for p in phase_order if p in phase_df['cat_label'].values]
palette = {k: CAT_PALETTE.get(k, '#95A5A6') for k in order}

sns.violinplot(data=phase_df, y='cat_label', x='duration_min', order=order,
               palette=palette, inner='box', cut=0, ax=ax, orient='h')

# Overlay individual points
sns.stripplot(data=phase_df, y='cat_label', x='duration_min', order=order,
              color='#2C3E50', alpha=0.4, size=5, jitter=0.15, ax=ax, orient='h')

ax.set_xlabel('Duration (minutes)', fontsize=12)
ax.set_ylabel('')
ax.set_title('Duration Distribution by Pipeline Phase', fontsize=14, fontweight='bold')
sns.despine()
plt.tight_layout()
plt.savefig('charts/08_phase_duration_violin.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 9: Agent effort — tool calls as a measure of work
    # ══════════════════════════════════════════════════════════════════════
    md("""## 9. Agent Effort — Who Does the Most Work?

Tool calls are a proxy for **actions taken** (file edits, command runs, searches). This shows which agent categories do the heaviest lifting. Size encodes cost — big expensive dots that are also high on tool calls are the pipeline's workhorses.""")

    code("""work_df = df[df['tool_calls'] > 0].copy()

fig, ax = plt.subplots(figsize=(14, 8))
for cat in work_df['cat_label'].unique():
    subset = work_df[work_df['cat_label'] == cat]
    color = CAT_PALETTE.get(cat, '#95A5A6')
    ax.scatter(subset['num_turns'], subset['tool_calls'],
               s=subset['cost'] * 200 + 20,
               c=color, alpha=0.6, edgecolors='white', linewidth=1,
               label=cat, zorder=3)

ax.set_xlabel('LLM Turns (thinking rounds)', fontsize=12)
ax.set_ylabel('Tool Calls (actions taken)', fontsize=12)
ax.set_title('Agent Effort — Turns vs Tool Calls (bubble size = cost)',
             fontsize=14, fontweight='bold')

# Diagonal line: 1 tool per turn
max_val = max(work_df['num_turns'].max(), work_df['tool_calls'].max())
ax.plot([0, max_val], [0, max_val], '--', color='#ccc', alpha=0.5, zorder=1, label='1 tool/turn')
ax.legend(fontsize=9, loc='upper left', ncol=2)
sns.despine()
plt.tight_layout()
plt.savefig('charts/09_agent_effort.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 10: Rework penalty
    # ══════════════════════════════════════════════════════════════════════
    md("""## 10. Rework Penalty — What Does QA Failure Cost?

For issues that required multiple coding iterations: how much extra time and money did the rework add? This quantifies the **cost of getting it wrong the first time**.""")

    code("""# Find issues with multiple coder iterations
coder_df = df[df['category'] == 'coder'].copy()
iter_counts = coder_df.groupby('issue')['iter'].max()
multi = iter_counts[iter_counts > 1]

if len(multi) > 0:
    # For each multi-iter issue, compare iter 1 cost vs total
    rework_data = []
    for issue_name in multi.index:
        issue_rows = df[df['issue'] == issue_name]
        first_iter_cost = issue_rows[issue_rows['category'].isin(['coder']) & (issue_rows['iter'] == 1)]['cost'].sum()
        first_iter_time = issue_rows[issue_rows['category'].isin(['coder']) & (issue_rows['iter'] == 1)]['duration_min'].sum()

        total_cost = issue_rows[issue_rows['category'].isin(['coder', 'reviewer', 'qa', 'synthesizer'])]['cost'].sum()
        total_time = issue_rows[issue_rows['category'].isin(['coder', 'reviewer', 'qa', 'synthesizer'])]['duration_min'].sum()

        rework_cost = total_cost - first_iter_cost
        rework_time = total_time - first_iter_time

        rework_data.append({
            'issue': issue_name,
            'first_iter_cost': first_iter_cost,
            'rework_cost': rework_cost,
            'first_iter_time': first_iter_time,
            'rework_time': rework_time,
            'iterations': int(multi[issue_name]),
        })

    rw = pd.DataFrame(rework_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cost comparison
    x = range(len(rw))
    ax1.barh(rw['issue'], rw['first_iter_cost'], label='First attempt', color='#3498DB', alpha=0.8)
    ax1.barh(rw['issue'], rw['rework_cost'], left=rw['first_iter_cost'],
             label='Rework cost', color='#E74C3C', alpha=0.8)
    ax1.set_xlabel('Cost ($)')
    ax1.set_title('Cost: First Attempt vs Rework')
    ax1.legend()

    # Time comparison
    ax2.barh(rw['issue'], rw['first_iter_time'], label='First attempt', color='#3498DB', alpha=0.8)
    ax2.barh(rw['issue'], rw['rework_time'], left=rw['first_iter_time'],
             label='Rework time', color='#E74C3C', alpha=0.8)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title('Time: First Attempt vs Rework')
    ax2.legend()

    plt.suptitle('The Cost of Rework — QA Failures Trigger Extra Iterations',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('charts/10_rework_penalty.png', bbox_inches='tight', facecolor='white')
    plt.show()
else:
    print("No multi-iteration issues found — all issues passed QA on first try!")
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 11: Pipeline stages waterfall
    # ══════════════════════════════════════════════════════════════════════
    md("""## 11. Pipeline Flow — End-to-End Stage Progression

A waterfall showing how the pipeline progresses through stages. Each bar shows a stage's wall-clock span (from first agent start to last agent end in that stage). Overlapping bars reveal parallelism between stages.""")

    code("""# Compute wall-clock span per category
stage_order = ['planning', 'issue_writer', 'coder', 'reviewer', 'qa',
               'synthesizer', 'merger', 'integration_tester']
stage_spans = []
for stage in stage_order:
    sdf = df[df['category'] == stage]
    if len(sdf) == 0:
        continue
    stage_spans.append({
        'stage': CAT_LABELS.get(stage, stage),
        'start': sdf['start_offset_min'].min(),
        'end': sdf['end_offset_min'].max(),
        'duration': sdf['end_offset_min'].max() - sdf['start_offset_min'].min(),
        'agent_count': len(sdf),
        'total_cost': sdf['cost'].sum(),
    })

spans = pd.DataFrame(stage_spans)

fig, ax = plt.subplots(figsize=(14, 6))
colors = [CAT_PALETTE.get(s, '#95A5A6') for s in spans['stage']]
bars = ax.barh(range(len(spans)), spans['duration'],
               left=spans['start'], color=colors, alpha=0.8,
               edgecolor='white', linewidth=2, height=0.6)

for i, row in spans.iterrows():
    ax.text(row['end'] + 0.5, i,
            f"{row['duration']:.0f} min · {row['agent_count']} agents · ${row['total_cost']:.1f}",
            va='center', fontsize=9, color='#555')

ax.set_yticks(range(len(spans)))
ax.set_yticklabels(spans['stage'], fontsize=11)
ax.set_xlabel('Wall-Clock Time (minutes from start)', fontsize=12)
ax.set_title('Pipeline Stage Spans — When Does Each Phase Run?',
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig('charts/11_pipeline_flow.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 12: Cost-per-issue ranking with breakdown
    # ══════════════════════════════════════════════════════════════════════
    md("""## 12. Issue Cost Ranking — Which Issues Are Most Expensive?

Each issue ranked by total cost, broken down by pipeline phase. This immediately reveals outliers — issues that consumed disproportionate resources.""")

    code("""coding_cats = ['coder', 'reviewer', 'qa', 'synthesizer']
issue_cost_df = df[df['category'].isin(coding_cats)].copy()

# Pivot
pivot = issue_cost_df.groupby(['issue', 'cat_label'])['cost'].sum().unstack(fill_value=0)
pivot['_total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('_total', ascending=True)
total_col = pivot.pop('_total')

fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.45)))

phase_colors = {'Coding': '#E74C3C', 'Code Review': '#F1C40F',
                'QA Testing': '#E67E22', 'Synthesis': '#BDC3C7'}
left = np.zeros(len(pivot))
for phase in ['Coding', 'Code Review', 'QA Testing', 'Synthesis']:
    if phase in pivot.columns:
        vals = pivot[phase].values
        ax.barh(range(len(pivot)), vals, left=left, label=phase,
                color=phase_colors.get(phase, '#95A5A6'), alpha=0.85,
                edgecolor='white', height=0.7)
        left += vals

# Total labels
for i, (idx, total) in enumerate(total_col.items()):
    ax.text(total + 0.05, i, f'${total:.2f}', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=10)
ax.set_xlabel('Cost ($)', fontsize=12)
ax.set_title('Issue Cost Ranking with Phase Breakdown', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('charts/12_issue_cost_ranking.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 13: Parallelism efficiency — agent-minutes vs wall-minutes
    # ══════════════════════════════════════════════════════════════════════
    md("""## 13. Parallelism Efficiency — Are We Using Concurrency Well?

Compares **total agent-minutes** (sum of all agent durations) vs **wall-clock minutes** per pipeline stage. The ratio reveals parallelism efficiency — a ratio of 15:1 for issue writers means 15 ran in parallel.""")

    code("""stage_data = []
for stage in ['planning', 'issue_writer', 'coder', 'reviewer', 'qa', 'synthesizer', 'merger', 'integration_tester']:
    sdf = df[df['category'] == stage]
    if len(sdf) == 0: continue
    agent_min = sdf['duration_min'].sum()
    wall_min = sdf['end_offset_min'].max() - sdf['start_offset_min'].min()
    wall_min = max(wall_min, 0.1)  # avoid div by zero
    stage_data.append({
        'stage': CAT_LABELS.get(stage, stage),
        'agent_min': agent_min,
        'wall_min': wall_min,
        'ratio': agent_min / wall_min,
    })

sdata = pd.DataFrame(stage_data).sort_values('ratio', ascending=True)

fig, ax = plt.subplots(figsize=(12, 6))
colors = [CAT_PALETTE.get(s, '#95A5A6') for s in sdata['stage']]
bars = ax.barh(sdata['stage'], sdata['ratio'], color=colors, alpha=0.85,
               edgecolor='white', height=0.6)

for bar, ratio, agent_m, wall_m in zip(bars, sdata['ratio'], sdata['agent_min'], sdata['wall_min']):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{ratio:.1f}x  ({agent_m:.0f} agent-min / {wall_m:.0f} wall-min)',
            va='center', fontsize=9)

ax.axvline(1, color='#E74C3C', linestyle='--', alpha=0.5, label='No parallelism (1x)')
ax.set_xlabel('Parallelism Ratio (agent-minutes / wall-minutes)', fontsize=12)
ax.set_title('Parallelism Efficiency — Higher = More Parallel',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig('charts/13_parallelism_efficiency.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # CHART 14: KPI Dashboard
    # ══════════════════════════════════════════════════════════════════════
    md("""## 14. Executive Dashboard — Key Metrics at a Glance""")

    code("""total_cost = df['cost'].sum()
wall_min = (df['end_ts'].max() - df['start_ts'].min()) / 60
agent_min = df['duration_min'].sum()
total_agents = len(df)
total_issues = df[df['category'] == 'coder']['issue'].nunique()
total_turns = df['num_turns'].sum()
total_tools = df['tool_calls'].sum()

# Issues with multiple iterations
coder_max_iter = df[df['category'] == 'coder'].groupby('issue')['iter'].max()
rework_count = (coder_max_iter > 1).sum()
first_pass_rate = (1 - rework_count / len(coder_max_iter)) * 100 if len(coder_max_iter) > 0 else 100

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle('Pipeline Execution Dashboard', fontsize=16, fontweight='bold', y=1.05)

kpis = [
    ('Total Cost', f'${total_cost:.2f}', '#E74C3C'),
    ('Wall Time', f'{wall_min:.0f} min', '#3498DB'),
    ('Agent Time', f'{agent_min:.0f} min', '#E67E22'),
    ('Parallelism', f'{agent_min/wall_min:.1f}x', '#9B59B6'),
    ('Total Agents', f'{total_agents}', '#1ABC9C'),
    ('Issues Built', f'{total_issues}', '#2ECC71'),
    ('First-Pass QA', f'{first_pass_rate:.0f}%', '#27AE60'),
    ('LLM Turns', f'{total_turns:,}', '#F1C40F'),
]

for ax, (label, value, color) in zip(axes.flat, kpis):
    ax.text(0.5, 0.55, value, transform=ax.transAxes, ha='center', va='center',
            fontsize=24, fontweight='bold', color=color)
    ax.text(0.5, 0.15, label, transform=ax.transAxes, ha='center', va='center',
            fontsize=11, color='#666')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#eee')
        spine.set_linewidth(2)

plt.tight_layout()
plt.savefig('charts/14_dashboard.png', bbox_inches='tight', facecolor='white')
plt.show()
""")

    # ══════════════════════════════════════════════════════════════════════
    # Summary text
    # ══════════════════════════════════════════════════════════════════════
    md("""## Summary Table""")
    code("""# Per-category summary
cat_summary = df.groupby('cat_label').agg(
    count=('file', 'count'),
    total_cost=('cost', 'sum'),
    total_min=('duration_min', 'sum'),
    avg_cost=('cost', 'mean'),
    avg_min=('duration_min', 'mean'),
    total_turns=('num_turns', 'sum'),
    total_tools=('tool_calls', 'sum'),
).sort_values('total_cost', ascending=False)

print("\\n" + "="*90)
print(f"  {'Category':<20} {'Agents':>6} {'Cost':>8} {'Time(min)':>10} {'Avg$/agent':>10} {'Turns':>7} {'Tools':>7}")
print("="*90)
for idx, row in cat_summary.iterrows():
    print(f"  {idx:<20} {row['count']:>6.0f} ${row['total_cost']:>7.2f} {row['total_min']:>10.1f} "
          f"${row['avg_cost']:>9.2f} {row['total_turns']:>7.0f} {row['total_tools']:>7.0f}")
print(f"  {'TOTAL':<20} {cat_summary['count'].sum():>6.0f} ${cat_summary['total_cost'].sum():>7.2f} "
      f"{cat_summary['total_min'].sum():>10.1f}")

# Key insights
print("\\n\\nKEY INSIGHTS:")
print(f"  • Coding is {cat_summary.loc['Coding','total_cost']/cat_summary['total_cost'].sum()*100:.0f}% of total cost")
print(f"  • QA is {cat_summary.loc['QA Testing','total_cost']/cat_summary['total_cost'].sum()*100:.0f}% of total cost — significant quality investment")
if 'Integration Test' in cat_summary.index:
    print(f"  • Integration testing: {cat_summary.loc['Integration Test','total_cost']/cat_summary['total_cost'].sum()*100:.0f}% of cost")
print(f"  • Planning is only {cat_summary.loc['Planning','total_cost']/cat_summary['total_cost'].sum()*100:.0f}% — cheap relative to execution")
print(f"  • Parallelism ratio: {cat_summary['total_min'].sum() / ((df['end_ts'].max() - df['start_ts'].min()) / 60):.1f}x (agent-min / wall-min)")
""")

    # Write notebook
    with open(NB_PATH, 'w') as f:
        nbformat.write(nb, f)
    print(f"\nNotebook written to: {NB_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    records = parse_all()
    print(f"Parsed {len(records)} agent executions")
    build_notebook(records)
    print("Done.")
