"""
Simple Job Completion Notifier.

A lightweight notification system for background jobs.
Can be run in a separate terminal to monitor job completion.

Usage:
    # In one terminal, run your training:
    python train.py
    
    # In another terminal, start monitoring:
    python utils/notify_on_complete.py --watch-dir results/
    
    # Or monitor a specific file:
    python utils/notify_on_complete.py --watch-file results/training_done.flag

Features:
- Monitors directory for new files (indicating job completion)
- Plays sound notification
- Shows Windows toast notification (if win10toast installed)
- Can execute a command on completion
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


def play_sound():
    """Play completion sound."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        print('\a', end='', flush=True)  # Bell character


def show_toast(title: str, message: str):
    """Show Windows toast notification."""
    try:
        import importlib
        win10toast = importlib.import_module('win10toast')
        ToastNotifier = win10toast.ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(title, message, duration=10, threaded=True)
    except (ImportError, AttributeError):
        pass  # Silently skip if not installed


def notify(title: str, message: str):
    """Send notification via multiple channels."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  [{timestamp}] 🔔 {title}")
    print(f"  {message}")
    print(f"{'='*60}\n")
    
    play_sound()
    show_toast(title, message)


def watch_directory(watch_dir: str, poll_interval: int = 10, 
                   pattern: str = "*", once: bool = False):
    """
    Watch a directory for new files matching pattern.
    
    Args:
        watch_dir: Directory to watch
        poll_interval: Seconds between checks
        pattern: File pattern to watch (e.g., "*.pt", "history.json")
        once: If True, exit after first detection
    """
    watch_path = Path(watch_dir)
    if not watch_path.exists():
        print(f"Creating watch directory: {watch_dir}")
        watch_path.mkdir(parents=True, exist_ok=True)
    
    print(f"👁️  Watching: {watch_dir}")
    print(f"   Pattern: {pattern}")
    print(f"   Poll interval: {poll_interval}s")
    print(f"   Press Ctrl+C to stop\n")
    
    # Get initial file list
    initial_files = set(f.name for f in watch_path.glob(pattern))
    print(f"   Initial files: {len(initial_files)}")
    
    try:
        while True:
            time.sleep(poll_interval)
            
            current_files = set(f.name for f in watch_path.glob(pattern))
            new_files = current_files - initial_files
            
            if new_files:
                for fname in sorted(new_files):
                    fpath = watch_path / fname
                    fsize = fpath.stat().st_size
                    notify(
                        "New File Detected",
                        f"File: {fname}\nSize: {fsize:,} bytes\nDir: {watch_dir}"
                    )
                    initial_files.add(fname)
                
                if once:
                    break
                    
    except KeyboardInterrupt:
        print(f"\n👋 Stopped watching {watch_dir}")


def watch_file(filepath: str, poll_interval: int = 5, 
               timeout: float = 0):
    """
    Watch for a specific file to be created.
    
    Args:
        filepath: File to watch
        poll_interval: Seconds between checks
        timeout: Maximum wait time in seconds (None = forever)
    """
    fpath = Path(filepath)
    
    print(f"👁️  Watching for file: {filepath}")
    print(f"   Poll interval: {poll_interval}s")
    if timeout:
        print(f"   Timeout: {timeout}s")
    print(f"   Press Ctrl+C to stop\n")
    
    start_time = time.time()
    
    try:
        while True:
            if fpath.exists():
                fsize = fpath.stat().st_size
                notify(
                    "File Created!",
                    f"File: {fpath.name}\nSize: {fsize:,} bytes\nPath: {filepath}"
                )
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                print(f"⏰ Timeout reached after {timeout}s")
                return False
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print(f"\n👋 Stopped watching {filepath}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Monitor jobs and notify on completion'
    )
    parser.add_argument('--watch-dir', type=str,
                       help='Directory to watch for new files')
    parser.add_argument('--watch-file', type=str,
                       help='Specific file to wait for')
    parser.add_argument('--pattern', type=str, default='*',
                       help='File pattern (with --watch-dir)')
    parser.add_argument('--poll-interval', type=int, default=10,
                       help='Seconds between checks')
    parser.add_argument('--timeout', type=float, default=None,
                       help='Timeout in seconds')
    parser.add_argument('--once', action='store_true',
                       help='Exit after first notification')
    
    args = parser.parse_args()
    
    if not args.watch_dir and not args.watch_file:
        # Default: watch results directory
        print("No watch target specified, using default: results/")
        args.watch_dir = "results"
    
    if args.watch_dir:
        watch_directory(
            args.watch_dir,
            poll_interval=args.poll_interval,
            pattern=args.pattern,
            once=args.once
        )
    elif args.watch_file:
        success = watch_file(
            args.watch_file,
            poll_interval=args.poll_interval,
            timeout=args.timeout
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
