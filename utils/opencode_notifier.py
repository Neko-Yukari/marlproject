"""
Opencode Auto-Notification System.

This module provides a way for background jobs to notify the AI agent
when they complete, enabling automatic follow-up responses.

How it works:
1. Background job (e.g., training script) calls notify_completion() when done
2. Notification is saved to a JSON file
3. On next user interaction, the AI checks for pending notifications
4. AI automatically generates a response summarizing the completed job

Usage in training scripts:
    from utils.opencode_notifier import notify_completion
    
    # When training completes
    notify_completion(
        job_type="training",
        job_name="GNN Mixed Training",
        status="success",  # or "failed"
        results={
            "best_cost": 1.46,
            "episodes": 20000,
            "duration": 7200
        },
        summary="Mixed-config GNN training completed with best cost 1.46"
    )

The AI will automatically pick this up and respond to the user.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

NOTIFICATION_DIR = Path("results/notifications")
NOTIFICATION_FILE = NOTIFICATION_DIR / "pending.json"
NOTIFICATION_HISTORY = NOTIFICATION_DIR / "history.jsonl"


def ensure_dirs():
    """Ensure notification directories exist."""
    NOTIFICATION_DIR.mkdir(parents=True, exist_ok=True)


def notify_completion(job_type: str, job_name: str, status: str = "success",
                      results: Optional[Dict[str, Any]] = None, summary: str = "",
                      priority: str = "normal"):
    """
    Notify that a background job has completed.
    
    Args:
        job_type: Type of job (training, evaluation, testing, etc.)
        job_name: Human-readable job name
        status: success, failed, timeout, canceled
        results: Dictionary of key results/metrics
        summary: One-line summary of what happened
        priority: normal, high, low
    """
    ensure_dirs()
    
    notification = {
        "timestamp": datetime.now().isoformat(),
        "job_type": job_type,
        "job_name": job_name,
        "status": status,
        "results": results or {},
        "summary": summary,
        "priority": priority,
        "read": False,
        "notified_at": time.time()
    }
    
    # Save to pending file
    with open(NOTIFICATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(notification, f, ensure_ascii=False, indent=2)
    
    # Also append to history
    with open(NOTIFICATION_HISTORY, 'a', encoding='utf-8') as f:
        f.write(json.dumps(notification, ensure_ascii=False) + '\n')
    
    # Print visible completion message to stdout
    print(f"\n{'='*70}")
    print(f"[!] JOB COMPLETED - Notification sent to AI agent")
    print(f"   Type: {job_type}")
    print(f"   Name: {job_name}")
    print(f"   Status: {status}")
    print(f"   Summary: {summary}")
    print(f"{'='*70}\n")
    
    # Try to play sound
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        print('\a', end='', flush=True)  # Fallback bell


def get_pending_notification() -> Optional[Dict[str, Any]]:
    """
    Check if there's a pending notification.
    Returns the notification dict or None.
    """
    if not NOTIFICATION_FILE.exists():
        return None
    
    try:
        with open(NOTIFICATION_FILE, 'r', encoding='utf-8') as f:
            notification = json.load(f)
        
        # Check if it's already been read
        if notification.get("read", False):
            return None
        
        # Check if it's recent (within last 24 hours)
        notified_at = notification.get("notified_at", 0)
        if time.time() - notified_at > 86400:  # 24 hours
            # Mark as read to avoid stale notifications
            mark_as_read()
            return None
        
        return notification
        
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def mark_as_read():
    """Mark the current notification as read."""
    if NOTIFICATION_FILE.exists():
        try:
            with open(NOTIFICATION_FILE, 'r', encoding='utf-8') as f:
                notification = json.load(f)
            notification["read"] = True
            with open(NOTIFICATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(notification, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def format_notification_for_ai(notification: Dict[str, Any]) -> str:
    """
    Format a notification into a response message for the AI to send.
    """
    job_name = notification.get("job_name", "Unknown Job")
    status = notification.get("status", "unknown")
    summary = notification.get("summary", "")
    results = notification.get("results", {})
    
    # Status markers
    status_emoji = {
        "success": "[OK]",
        "failed": "[FAIL]",
        "timeout": "[TIMEOUT]",
        "canceled": "[CANCELED]"
    }.get(status, "[INFO]")
    
    message = f"{status_emoji} **{job_name}** 已完成\n\n"
    
    if summary:
        message += f"{summary}\n\n"
    
    if results:
        message += "**关键结果：**\n"
        for key, value in results.items():
            # Format numbers nicely
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            elif isinstance(value, int) and value > 1000:
                formatted = f"{value:,}"
            else:
                formatted = str(value)
            message += f"- {key}: {formatted}\n"
    
    # Add suggestion for next steps based on job type
    job_type = notification.get("job_type", "")
    if job_type == "training":
        message += "\n[Tip] You can: view detailed results, compare configs, or continue training"
    elif job_type == "evaluation":
        message += "\n[Tip] You can: view evaluation report, analyze failures, or tune parameters"
    
    return message


def check_and_notify() -> Optional[str]:
    """
    Check for pending notifications and return formatted message.
    Call this at the start of each interaction.
    
    Returns:
        Formatted message string if there's a pending notification, None otherwise.
    """
    notification = get_pending_notification()
    if notification is None:
        return None
    
    # Mark as read so we don't notify again
    mark_as_read()
    
    return format_notification_for_ai(notification)


# Convenience functions for common job types

def notify_training_complete(job_name: str, best_metric: float, episodes: int,
                             duration_seconds: float, **kwargs):
    """Convenience function for training completion."""
    hours = duration_seconds / 3600
    notify_completion(
        job_type="training",
        job_name=job_name,
        status="success",
        results={
            "best_metric": best_metric,
            "total_episodes": episodes,
            "duration_hours": round(hours, 2),
            **kwargs
        },
        summary=f"{job_name} training completed, best metric: {best_metric:.4f}, "
                f"duration: {hours:.1f} hours"
    )


def notify_evaluation_complete(job_name: str, metrics: Dict[str, float],
                               duration_seconds: float = 0):
    """Convenience function for evaluation completion."""
    notify_completion(
        job_type="evaluation",
        job_name=job_name,
        status="success",
        results=metrics,
        summary=f"{job_name} 评估完成"
    )


def notify_job_failed(job_name: str, error_message: str = ""):
    """Notify that a job failed."""
    notify_completion(
        job_type="training",
        job_name=job_name,
        status="failed",
        results={"error": error_message},
        summary=f"{job_name} 失败: {error_message}",
        priority="high"
    )


# Example usage
if __name__ == "__main__":
    # Test notification
    notify_training_complete(
        job_name="GNN Mixed Training Test",
        best_metric=1.46,
        episodes=20000,
        duration_seconds=7200
    )
    
    # Test reading
    notif = get_pending_notification()
    if notif:
        print("\nPending notification found:")
        print(format_notification_for_ai(notif))
