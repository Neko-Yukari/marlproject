"""
Enhanced Terminal Job Monitor.

Monitors background jobs and provides completion notifications.
Can be used standalone or imported as a module.

Usage:
    python job_monitor.py <job_id> [--poll-interval 30]
    
Or programmatically:
    from job_monitor import watch_job, notify_completion
    status = watch_job("daring-comet-13", poll_interval=30)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Configuration
DEFAULT_POLL_INTERVAL = 30  # seconds
NOTIFICATION_SOUND = True
STATUS_FILE_DIR = Path("results/job_status")


def ensure_status_dir():
    """Ensure status directory exists."""
    STATUS_FILE_DIR.mkdir(parents=True, exist_ok=True)


def save_job_status(job_id: str, status: Dict):
    """Save job status to file for external monitoring."""
    ensure_status_dir()
    status_file = STATUS_FILE_DIR / f"{job_id}.json"
    status["updated_at"] = datetime.now().isoformat()
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def load_job_status(job_id: str) -> Optional[Dict]:
    """Load job status from file."""
    status_file = STATUS_FILE_DIR / f"{job_id}.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return None


def play_notification_sound():
    """Play system notification sound (Windows)."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception:
        pass  # Silent fail on non-Windows or no sound


def show_notification(title: str, message: str):
    """Show Windows toast notification."""
    try:
        try:
            from win10toast import ToastNotifier
        except ImportError:
            ToastNotifier = None
        if ToastNotifier is not None:
            toaster = ToastNotifier()
            toaster.show_toast(title, message, duration=10)
        else:
            raise ImportError("win10toast not installed")
    except ImportError:
        # Fallback: print prominently
        print(f"\n{'='*60}")
        print(f"  📢 {title}")
        print(f"  {message}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Notification error: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def notify_job_completion(job_id: str, status: str, duration: float, 
                          exit_code: Optional[int] = None):
    """Notify user that a job has completed."""
    duration_str = format_duration(duration)
    
    if status == "Completed":
        title = f"✅ Job Complete: {job_id}"
        message = f"Duration: {duration_str} | Exit: {exit_code or 'OK'}"
    elif status == "Failed":
        title = f"❌ Job Failed: {job_id}"
        message = f"Duration: {duration_str} | Exit: {exit_code or 'Error'}"
    elif status == "TimedOut":
        title = f"⏰ Job Timed Out: {job_id}"
        message = f"Duration: {duration_str}"
    elif status == "Canceled":
        title = f"🛑 Job Canceled: {job_id}"
        message = f"Duration: {duration_str}"
    else:
        title = f"ℹ️ Job Status: {job_id}"
        message = f"Status: {status} | Duration: {duration_str}"
    
    # Play sound
    if NOTIFICATION_SOUND:
        play_notification_sound()
    
    # Show notification
    show_notification(title, message)
    
    # Also save to completion log
    ensure_status_dir()
    completion_file = STATUS_FILE_DIR / "completion_log.txt"
    with open(completion_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {title} - {message}\n")


def watch_job(job_id: str, poll_interval: int = DEFAULT_POLL_INTERVAL,
              timeout: Optional[float] = None) -> Dict:
    """
    Monitor a background job until completion.
    
    This function polls the job status by checking for status files
    that should be written by the main process. In the MCP environment,
    this requires coordination with the main agent.
    
    Args:
        job_id: The enhanced_terminal job ID
        poll_interval: Seconds between checks
        timeout: Maximum time to wait (None = forever)
        
    Returns:
        Final status dictionary
    """
    print(f"🔍 Monitoring job: {job_id}")
    print(f"   Poll interval: {poll_interval}s")
    if timeout:
        print(f"   Timeout: {format_duration(timeout)}")
    print()
    
    start_time = time.time()
    last_status = None
    
    try:
        while True:
            # Check if we've exceeded timeout
            if timeout and (time.time() - start_time) > timeout:
                print(f"⏰ Watch timeout reached for {job_id}")
                return {"status": "WatchTimeout", "job_id": job_id}
            
            # Try to load current status
            status = load_job_status(job_id)
            
            if status is None:
                # No status file yet, job might not have started reporting
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for job to report status...")
            else:
                current_status = status.get("status", "Unknown")
                
                # Check if status changed
                if current_status != last_status:
                    elapsed = status.get("duration", time.time() - start_time)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {current_status} "
                          f"({format_duration(elapsed)})")
                    last_status = current_status
                
                # Check if job is complete
                if current_status in ["Completed", "Failed", "TimedOut", "Canceled"]:
                    duration = status.get("duration", time.time() - start_time)
                    exit_code = status.get("exit_code")
                    
                    # Notify
                    notify_job_completion(job_id, current_status, duration, exit_code)
                    
                    return status
            
            # Wait before next poll
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print(f"\n👋 Stopped monitoring {job_id}")
        return {"status": "MonitoringStopped", "job_id": job_id}


def mark_job_status(job_id: str, status: str, **kwargs):
    """
    Mark a job's status. This should be called by the main process
    when job status changes.
    
    Usage in main agent:
        from utils.job_monitor import mark_job_status
        mark_job_status("daring-comet-13", "Completed", 
                        duration=3600, exit_code=0)
    """
    status_data = {
        "job_id": job_id,
        "status": status,
        **kwargs
    }
    save_job_status(job_id, status_data)
    
    # If job is complete, also trigger notification
    if status in ["Completed", "Failed", "TimedOut", "Canceled"]:
        duration = kwargs.get("duration", 0)
        exit_code = kwargs.get("exit_code")
        notify_job_completion(job_id, status, duration, exit_code)


def list_monitored_jobs() -> list:
    """List all jobs being monitored."""
    ensure_status_dir()
    jobs = []
    for f in STATUS_FILE_DIR.glob("*.json"):
        if f.stem != "completion_log":
            with open(f, 'r') as fh:
                jobs.append(json.load(fh))
    return jobs


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python job_monitor.py <job_id> [--poll-interval 30]")
        print("\nExample:")
        print("  python job_monitor.py daring-comet-13")
        print("  python job_monitor.py daring-comet-13 --poll-interval 60")
        sys.exit(1)
    
    job_id = sys.argv[1]
    poll_interval = DEFAULT_POLL_INTERVAL
    
    # Parse optional arguments
    if "--poll-interval" in sys.argv:
        try:
            idx = sys.argv.index("--poll-interval")
            poll_interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid poll interval, using default")
    
    # Start watching
    final_status = watch_job(job_id, poll_interval=poll_interval)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL STATUS:")
    print(f"  Job ID: {final_status.get('job_id', job_id)}")
    print(f"  Status: {final_status.get('status', 'Unknown')}")
    if 'duration' in final_status:
        print(f"  Duration: {format_duration(final_status['duration'])}")
    if 'exit_code' in final_status:
        print(f"  Exit Code: {final_status['exit_code']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
