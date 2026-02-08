#!/usr/bin/env python3
"""
Grid Network Report Generator
Generates comprehensive HTML/PDF reports for grid network evaluation.
"""

import os
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_rl.network.multi_intersection import MultiIntersectionNetwork
from traffic_rl.utils.green_wave_viz import (
    plot_time_space_diagram,
    plot_green_wave_bands,
    plot_coordination_heatmap
)


def generate_grid_report(
    evaluation_results_path,
    output_dir='results/grid_report',
    format='html'
):
    """
    Generate comprehensive report from evaluation results.
    
    Args:
        evaluation_results_path: Path to evaluation results JSON
        output_dir: Directory for saving report
        format: Report format ('html' or 'pdf')
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load evaluation results
    with open(evaluation_results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Generating {format.upper()} report...")
    print(f"Loading results from {evaluation_results_path}")
    
    # Extract data
    network_config = results.get('network_config', {})
    avg_metrics = results.get('average_metrics', {})
    baseline_comparison = results.get('baseline_comparison', {})
    
    # Create visualizations
    viz_dir = output_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Coordination scores
    if 'coordination_scores' in results:
        coord_scores = results['coordination_scores']
        plot_green_wave_bands(
            coord_scores,
            output_path=viz_dir / 'coordination_scores.png'
        )
        
        # Heatmap
        rows = network_config.get('rows', 3)
        cols = network_config.get('cols', 3)
        plot_coordination_heatmap(
            coord_scores,
            grid_shape=(rows, cols),
            output_path=viz_dir / 'coordination_heatmap.png'
        )
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Grid Network Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-item {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .improvement {{
            color: #27ae60;
            font-weight: bold;
        }}
        .degradation {{
            color: #e74c3c;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 14px;
            margin-top: 30px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>🚦 Grid Network Evaluation Report</h1>
    
    <div class="metric-card">
        <h2>Network Configuration</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-label">Grid Size</div>
                <div class="metric-value">{network_config.get('rows', 'N/A')}×{network_config.get('cols', 'N/A')}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Total Intersections</div>
                <div class="metric-value">{network_config.get('total_intersections', 'N/A')}</div>
            </div>
        </div>
    </div>
    
    <div class="metric-card">
        <h2>Performance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-label">Total Throughput</div>
                <div class="metric-value">{avg_metrics.get('total_throughput', 0):.2f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">vehicles</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Avg Waiting Time</div>
                <div class="metric-value">{avg_metrics.get('network_avg_waiting_time', 0):.2f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">seconds</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Avg Delay</div>
                <div class="metric-value">{avg_metrics.get('network_avg_delay', 0):.2f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">seconds</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Avg Speed</div>
                <div class="metric-value">{avg_metrics.get('network_avg_speed', 0):.2f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">m/s</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Coordination Score</div>
                <div class="metric-value">{avg_metrics.get('avg_coordination_score', 0):.3f}</div>
                <div style="font-size: 12px; color: #7f8c8d;">0-1 scale</div>
            </div>
        </div>
    </div>
"""
    
    # Add baseline comparison if available
    if baseline_comparison:
        html_content += f"""
    <div class="metric-card">
        <h2>Baseline Comparison</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-label">Throughput Improvement</div>
                <div class="metric-value {'improvement' if baseline_comparison.get('throughput_improvement', 0) > 0 else 'degradation'}">
                    {baseline_comparison.get('throughput_improvement', 0):+.2f}%
                </div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Waiting Time Reduction</div>
                <div class="metric-value {'improvement' if baseline_comparison.get('waiting_time_reduction', 0) > 0 else 'degradation'}">
                    {baseline_comparison.get('waiting_time_reduction', 0):+.2f}%
                </div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Delay Reduction</div>
                <div class="metric-value {'improvement' if baseline_comparison.get('delay_reduction', 0) > 0 else 'degradation'}">
                    {baseline_comparison.get('delay_reduction', 0):+.2f}%
                </div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Overall Improvement</div>
                <div class="metric-value {'improvement' if baseline_comparison.get('overall_improvement', 0) > 0 else 'degradation'}">
                    {baseline_comparison.get('overall_improvement', 0):+.2f}%
                </div>
            </div>
        </div>
    </div>
"""
    
    # Add visualizations
    html_content += """
    <div class="metric-card">
        <h2>Coordination Analysis</h2>
        <img src="visualizations/coordination_scores.png" alt="Coordination Scores">
        <img src="visualizations/coordination_heatmap.png" alt="Coordination Heatmap">
    </div>
    
    <div class="timestamp">
        Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
    </div>
</body>
</html>
"""
    
    # Save HTML report
    report_file = output_path / 'evaluation_report.html'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ Report generated: {report_file}")
    print(f"✓ Visualizations saved to: {viz_dir}")
    
    return report_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate grid network report")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to evaluation results JSON")
    parser.add_argument("--output-dir", type=str, default='results/grid_report',
                       help="Output directory (default: results/grid_report)")
    parser.add_argument("--format", type=str, default='html',
                       choices=['html', 'pdf'],
                       help="Report format (default: html)")
    
    args = parser.parse_args()
    
    # Generate report
    generate_grid_report(
        evaluation_results_path=args.results,
        output_dir=args.output_dir,
        format=args.format
    )


if __name__ == "__main__":
    main()
