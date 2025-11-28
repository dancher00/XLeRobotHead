#!/usr/bin/env python3
"""
Data visualization script for palm state data collected from XLeRobotHead.

Usage:
    python data_vis.py <csv_file>
    python data_vis.py palm_data_20240101_120000.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_data(csv_file):
    """Load palm state data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

def plot_position_trajectory(df, hand='both'):
    """Plot 3D position trajectory"""
    fig = plt.figure(figsize=(12, 10))
    
    if hand == 'both' or hand == 'left':
        left_df = df[df['hand'] == 'left']
        if not left_df.empty:
            ax = fig.add_subplot(221, projection='3d')
            ax.plot(left_df['x'], left_df['y'], left_df['z'], 'b-', alpha=0.6, linewidth=1)
            ax.scatter(left_df['x'].iloc[0], left_df['y'].iloc[0], left_df['z'].iloc[0], 
                      color='green', s=100, marker='o', label='Start')
            ax.scatter(left_df['x'].iloc[-1], left_df['y'].iloc[-1], left_df['z'].iloc[-1], 
                      color='red', s=100, marker='x', label='End')
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Left)')
            ax.set_zlabel('Z (Up)')
            ax.set_title('Left Hand - 3D Position Trajectory')
            ax.legend()
            ax.grid(True)
    
    if hand == 'both' or hand == 'right':
        right_df = df[df['hand'] == 'right']
        if not right_df.empty:
            ax = fig.add_subplot(222, projection='3d')
            ax.plot(right_df['x'], right_df['y'], right_df['z'], 'r-', alpha=0.6, linewidth=1)
            ax.scatter(right_df['x'].iloc[0], right_df['y'].iloc[0], right_df['z'].iloc[0], 
                      color='green', s=100, marker='o', label='Start')
            ax.scatter(right_df['x'].iloc[-1], right_df['y'].iloc[-1], right_df['z'].iloc[-1], 
                      color='red', s=100, marker='x', label='End')
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Left)')
            ax.set_zlabel('Z (Up)')
            ax.set_title('Right Hand - 3D Position Trajectory')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_position_over_time(df, hand='both'):
    """Plot position (x, y, z) over time"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time_col = 'timestamp'
    if time_col not in df.columns:
        # Use index as time if timestamp not available
        df['time_index'] = range(len(df))
        time_col = 'time_index'
    
    for i, coord in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        
        if hand == 'both' or hand == 'left':
            left_df = df[df['hand'] == 'left']
            if not left_df.empty:
                ax.plot(left_df[time_col], left_df[coord], 'b-', label='Left', alpha=0.7, linewidth=1.5)
        
        if hand == 'both' or hand == 'right':
            right_df = df[df['hand'] == 'right']
            if not right_df.empty:
                ax.plot(right_df[time_col], right_df[coord], 'r-', label='Right', alpha=0.7, linewidth=1.5)
        
        coord_labels = {'x': 'X (Forward)', 'y': 'Y (Left)', 'z': 'Z (Up)'}
        ax.set_ylabel(coord_labels[coord])
        ax.set_title(f'{coord_labels[coord]} over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig

def plot_orientation_over_time(df, hand='both'):
    """Plot orientation (roll, pitch, yaw) over time"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time_col = 'timestamp'
    if time_col not in df.columns:
        df['time_index'] = range(len(df))
        time_col = 'time_index'
    
    for i, angle in enumerate(['roll', 'pitch', 'yaw']):
        ax = axes[i]
        
        if hand == 'both' or hand == 'left':
            left_df = df[df['hand'] == 'left']
            if not left_df.empty:
                ax.plot(left_df[time_col], left_df[angle], 'b-', label='Left', alpha=0.7, linewidth=1.5)
        
        if hand == 'both' or hand == 'right':
            right_df = df[df['hand'] == 'right']
            if not right_df.empty:
                ax.plot(right_df[time_col], right_df[angle], 'r-', label='Right', alpha=0.7, linewidth=1.5)
        
        ax.set_ylabel(f'{angle.capitalize()} (degrees)')
        ax.set_title(f'{angle.capitalize()} over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig

def plot_2d_projections(df, hand='both'):
    """Plot 2D projections of 3D trajectory"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if hand == 'both' or hand == 'left':
        left_df = df[df['hand'] == 'left']
        if not left_df.empty:
            # XY projection (Forward-Left)
            axes[0, 0].plot(left_df['x'], left_df['y'], 'b-', alpha=0.6, linewidth=1.5, label='Left')
            axes[0, 0].scatter(left_df['x'].iloc[0], left_df['y'].iloc[0], color='green', s=50, marker='o')
            axes[0, 0].scatter(left_df['x'].iloc[-1], left_df['y'].iloc[-1], color='red', s=50, marker='x')
            axes[0, 0].set_xlabel('X (Forward)')
            axes[0, 0].set_ylabel('Y (Left)')
            axes[0, 0].set_title('XY Projection (Top View)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axis('equal')
            
            # XZ projection (Forward-Up)
            axes[0, 1].plot(left_df['x'], left_df['z'], 'b-', alpha=0.6, linewidth=1.5, label='Left')
            axes[0, 1].scatter(left_df['x'].iloc[0], left_df['z'].iloc[0], color='green', s=50, marker='o')
            axes[0, 1].scatter(left_df['x'].iloc[-1], left_df['z'].iloc[-1], color='red', s=50, marker='x')
            axes[0, 1].set_xlabel('X (Forward)')
            axes[0, 1].set_ylabel('Z (Up)')
            axes[0, 1].set_title('XZ Projection (Side View)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axis('equal')
            
            # YZ projection (Left-Up)
            axes[1, 0].plot(left_df['y'], left_df['z'], 'b-', alpha=0.6, linewidth=1.5, label='Left')
            axes[1, 0].scatter(left_df['y'].iloc[0], left_df['z'].iloc[0], color='green', s=50, marker='o')
            axes[1, 0].scatter(left_df['y'].iloc[-1], left_df['z'].iloc[-1], color='red', s=50, marker='x')
            axes[1, 0].set_xlabel('Y (Left)')
            axes[1, 0].set_ylabel('Z (Up)')
            axes[1, 0].set_title('YZ Projection (Front View)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axis('equal')
    
    if hand == 'both' or hand == 'right':
        right_df = df[df['hand'] == 'right']
        if not right_df.empty:
            # XY projection
            axes[0, 0].plot(right_df['x'], right_df['y'], 'r-', alpha=0.6, linewidth=1.5, label='Right')
            axes[0, 0].scatter(right_df['x'].iloc[0], right_df['y'].iloc[0], color='green', s=50, marker='o')
            axes[0, 0].scatter(right_df['x'].iloc[-1], right_df['y'].iloc[-1], color='red', s=50, marker='x')
            
            # XZ projection
            axes[0, 1].plot(right_df['x'], right_df['z'], 'r-', alpha=0.6, linewidth=1.5, label='Right')
            axes[0, 1].scatter(right_df['x'].iloc[0], right_df['z'].iloc[0], color='green', s=50, marker='o')
            axes[0, 1].scatter(right_df['x'].iloc[-1], right_df['z'].iloc[-1], color='red', s=50, marker='x')
            
            # YZ projection
            axes[1, 0].plot(right_df['y'], right_df['z'], 'r-', alpha=0.6, linewidth=1.5, label='Right')
            axes[1, 0].scatter(right_df['y'].iloc[0], right_df['z'].iloc[0], color='green', s=50, marker='o')
            axes[1, 0].scatter(right_df['y'].iloc[-1], right_df['z'].iloc[-1], color='red', s=50, marker='x')
    
    # Statistics
    axes[1, 1].axis('off')
    stats_text = "Statistics:\n\n"
    for h in ['left', 'right']:
        hand_df = df[df['hand'] == h]
        if not hand_df.empty:
            stats_text += f"{h.upper()} Hand:\n"
            stats_text += f"  Samples: {len(hand_df)}\n"
            stats_text += f"  X range: [{hand_df['x'].min():.3f}, {hand_df['x'].max():.3f}]\n"
            stats_text += f"  Y range: [{hand_df['y'].min():.3f}, {hand_df['y'].max():.3f}]\n"
            stats_text += f"  Z range: [{hand_df['z'].min():.3f}, {hand_df['z'].max():.3f}]\n"
            stats_text += f"  Roll range: [{hand_df['roll'].min():.1f}°, {hand_df['roll'].max():.1f}°]\n"
            stats_text += f"  Pitch range: [{hand_df['pitch'].min():.1f}°, {hand_df['pitch'].max():.1f}°]\n"
            stats_text += f"  Yaw range: [{hand_df['yaw'].min():.1f}°, {hand_df['yaw'].max():.1f}°]\n\n"
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
                    family='monospace')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize palm state data from CSV file')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--hand', choices=['left', 'right', 'both'], default='both',
                       help='Which hand to plot (default: both)')
    parser.add_argument('--save', action='store_true', help='Save plots to files instead of showing')
    parser.add_argument('--output-dir', default='.', help='Output directory for saved plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = load_data(args.csv_file)
    print(f"Loaded {len(df)} samples")
    print(f"Hands: {df['hand'].unique()}")
    
    # Create plots
    print("Generating plots...")
    
    # 3D trajectory
    fig1 = plot_position_trajectory(df, hand=args.hand)
    if args.save:
        fig1.savefig(f"{args.output_dir}/trajectory_3d.png", dpi=150, bbox_inches='tight')
        print("Saved: trajectory_3d.png")
    else:
        plt.show(block=False)
    
    # Position over time
    fig2 = plot_position_over_time(df, hand=args.hand)
    if args.save:
        fig2.savefig(f"{args.output_dir}/position_over_time.png", dpi=150, bbox_inches='tight')
        print("Saved: position_over_time.png")
    else:
        plt.show(block=False)
    
    # Orientation over time
    fig3 = plot_orientation_over_time(df, hand=args.hand)
    if args.save:
        fig3.savefig(f"{args.output_dir}/orientation_over_time.png", dpi=150, bbox_inches='tight')
        print("Saved: orientation_over_time.png")
    else:
        plt.show(block=False)
    
    # 2D projections
    fig4 = plot_2d_projections(df, hand=args.hand)
    if args.save:
        fig4.savefig(f"{args.output_dir}/projections_2d.png", dpi=150, bbox_inches='tight')
        print("Saved: projections_2d.png")
    else:
        plt.show(block=False)
    
    if not args.save:
        print("\nPress Enter to close all plots...")
        input()
        plt.close('all')

if __name__ == '__main__':
    main()

