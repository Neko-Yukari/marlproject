# Auto-Notification System for Opencode

## Problem
When `enhanced_terminal` runs background jobs, they complete silently without notifying the AI agent, requiring manual polling to check status.

## Solution
A file-based notification system that:
1. Background jobs write completion status to a JSON file
2. AI agent checks for pending notifications on each interaction
3. Automatically generates a response when notification is found

## Files Created

### 1. `utils/opencode_notifier.py`
Main notification module. Provides:

**For training scripts (call when job completes):**
```python
from utils.opencode_notifier import notify_training_complete

notify_training_complete(
    job_name="GNN Mixed Training",
    best_metric=1.46,
    episodes=20000,
    duration_seconds=7200
)
```

**For AI agent (check at start of each interaction):**
```python
from utils.opencode_notifier import check_and_notify

message = check_and_notify()
if message:
    # Automatically respond to user with results
    return message
```

### 2. `utils/notify_on_complete.py`
Standalone file watcher. Run in separate terminal:
```bash
python utils/notify_on_complete.py --watch-dir results/
```

### 3. `utils/job_monitor.py`
Legacy monitoring with status files.

## How It Works

### Flow Diagram

```
User: Run training
    |
    v
[enhanced_terminal] ---> Background job
    |                           |
    |                           v
    |                  Training completes
    |                           |
    |                           v
    |              Write notification JSON
    |              to results/notifications/
    |                           |
    |                           v
    |              Play sound + show popup
    |                           |
    |                           v
    |                  [User gets alerted]
    |                           |
    v                           v
User: "How's it going?"  --->  AI checks
                                notifications
                                    |
                                    v
                            Found notification!
                                    |
                                    v
                            Auto-generate response:
                            "Training completed!
                             Best cost: 1.46..."
```

### Notification Lifecycle

1. **Created**: Job calls `notify_completion()` 
2. **Pending**: File exists at `results/notifications/pending.json`
3. **Read**: AI calls `check_and_notify()`, marks as read
4. **Archived**: Original notification saved to `history.jsonl`

## Integration Guide

### For Training Scripts

Add at the end of any training script:

```python
from utils.opencode_notifier import notify_training_complete
import time

start_time = time.time()

# ... training code ...

training_duration = time.time() - start_time

notify_training_complete(
    job_name="Your Training Name",
    best_metric=best_cost,
    episodes=num_episodes,
    duration_seconds=training_duration
)
```

### For AI Agent

The AI should check for notifications at the start of each interaction:

```python
def process_user_message(user_message):
    # First, check for completed jobs
    notification = check_and_notify()
    if notification:
        return notification  # Auto-respond with results
    
    # Otherwise, process normally
    return handle_message(user_message)
```

## Features

- **Persistent**: Notifications survive process restarts
- **Deduplication**: Read notifications won't be reported again
- **Expiration**: Notifications older than 24h are auto-cleaned
- **Priority**: High-priority notifications can be marked
- **History**: All notifications archived to JSONL file

## Testing

```bash
# Create test notification
python -c "from utils.opencode_notifier import notify_training_complete; notify_training_complete('Test', 0.5, 100, 60)"

# Check for notifications
python -c "from utils.opencode_notifier import check_and_notify; print(check_and_notify())"
```

## Notes

- Currently uses file-based signaling (works across processes)
- Sound notifications require Windows + winsound
- Toast notifications require `win10toast` package (optional)
- For non-Windows systems, falls back to terminal bell
