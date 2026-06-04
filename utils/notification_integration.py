"""
Auto-Notification Integration for Training Scripts.

Add this to the bottom of any training script to enable automatic
notification when the job completes.

Example:
    from utils.opencode_notifier import notify_training_complete
    
    # ... training code ...
    
    # At the end:
    notify_training_complete(
        job_name="GNN Mixed Training",
        best_metric=best_cost,
        episodes=20000,
        duration_seconds=elapsed_time,
        extra_info={"config": "2ES-3MD/5MD/7MD"}
    )
"""

# Quick integration example
INTEGRATION_TEMPLATE = '''
# Add at the end of your training script:
from utils.opencode_notifier import notify_training_complete
import time

# ... your existing training code ...

# Calculate duration
training_duration = time.time() - start_time

# Send notification
notify_training_complete(
    job_name="Your Training Job Name",
    best_metric=best_cost,
    episodes=total_episodes,
    duration_seconds=training_duration,
    # Add any extra metrics you want to report
    final_cost=final_cost,
    completion_rate=comp_rate
)
'''

if __name__ == "__main__":
    print("Auto-Notification Integration Template:")
    print(INTEGRATION_TEMPLATE)
