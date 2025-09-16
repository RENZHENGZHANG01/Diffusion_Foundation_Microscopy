"""
Phase Management for Automated Training Pipeline
==============================================
Handles phase transitions, checkpoints, and state persistence
"""

import json
import os
import fcntl
import time
from pathlib import Path
from typing import Dict, List


class MicroscopyPhaseManager:
    """Manages automated phase transitions and checkpoint handling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.save_path = Path(config['train']['save_path'])
        self.state_file = self.save_path / 'phase_state.json'
        self.state = self.load_state()
    
    def load_state(self):
        """Load or create phase state with file locking"""
        if self.state_file.exists():
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.state_file, 'r') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                        content = f.read().strip()
                        if not content:  # Empty file
                            print(f"[WARNING] Empty state file found: {self.state_file}")
                            return self._get_default_state()
                        return json.loads(content)
                except (json.JSONDecodeError, IOError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"[WARNING] Attempt {attempt + 1} failed to read state file: {e}")
                        time.sleep(0.1)  # Brief wait before retry
                        continue
                    else:
                        print(f"[WARNING] Failed to read state file after {max_retries} attempts: {e}")
                        print("[WARNING] Creating new state file")
                        return self._get_default_state()
        return self._get_default_state()
    
    def _get_default_state(self):
        """Get default state structure"""
        return {
            'current_phase': 0,
            'completed_phases': [],
            'phase_checkpoints': {},
            'best_metrics': {},
            'training_logs': []
        }
    
    def save_state(self):
        """Save current state with file locking"""
        self.save_path.mkdir(parents=True, exist_ok=True)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(self.state_file, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock for writing
                    json.dump(self.state, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                break
            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"[WARNING] Attempt {attempt + 1} failed to save state file: {e}")
                    time.sleep(0.1)  # Brief wait before retry
                    continue
                else:
                    print(f"[ERROR] Failed to save state file after {max_retries} attempts: {e}")
                    raise
    
    def get_current_phase(self):
        """Get current phase config"""
        phases = self.config['phases']
        if self.state['current_phase'] >= len(phases):
            return None
        return phases[self.state['current_phase']]
    
    def complete_phase(self, checkpoint_path: str, metrics: Dict = None):
        """Mark phase complete and setup next"""
        phase = self.get_current_phase()
        if not phase:
            return
            
        phase_name = phase['name']
        
        # Save checkpoint reference
        if phase_name not in self.state['phase_checkpoints']:
            self.state['phase_checkpoints'][phase_name] = []
        
        checkpoints = self.state['phase_checkpoints'][phase_name]
        checkpoints.append(checkpoint_path)
        
        # Keep only 4 checkpoints per phase
        if len(checkpoints) > 4:
            old_ckpt = checkpoints.pop(0)
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
        
        # Save metrics
        if metrics:
            self.state['best_metrics'][phase_name] = metrics
        
        # Mark complete
        self.state['completed_phases'].append(phase_name)
        
        # Move to next phase if available
        if self.state['current_phase'] + 1 < len(self.config['phases']):
            self.state['current_phase'] += 1
            
            # Auto-set base checkpoint for next phase
            next_phase = self.config['phases'][self.state['current_phase']]
            if next_phase.get('base_from_phase') == phase_name:
                next_phase['base_checkpoint'] = checkpoint_path
        
        self.save_state()
        print(f"[COMPLETED] Phase '{phase_name}' completed")
        print(f"[CHECKPOINT] {checkpoint_path}")
        
        # Log training event
        self.log_event(f"Completed phase: {phase_name}", {"checkpoint": checkpoint_path})
    
    def log_event(self, message: str, data: Dict = None):
        """Log training event with timestamp"""
        from datetime import datetime
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'phase': self.get_current_phase()['name'] if self.get_current_phase() else 'complete',
            'data': data or {}
        }
        
        self.state['training_logs'].append(event)
        
        # Keep only last 100 events
        if len(self.state['training_logs']) > 100:
            self.state['training_logs'] = self.state['training_logs'][-100:]
        
        self.save_state()
    
    def is_complete(self):
        """Check if all phases done"""
        return self.state['current_phase'] >= len(self.config['phases'])
    
    def get_best_checkpoint(self, phase_name: str):
        """Get best checkpoint from completed phase"""
        checkpoints = self.state['phase_checkpoints'].get(phase_name, [])
        return checkpoints[-1] if checkpoints else None
    
    def find_latest_checkpoint(self, phase_name: str):
        """Find latest checkpoint for a phase from filesystem"""
        checkpoint_dir = self.save_path / 'checkpoints'
        if not checkpoint_dir.exists():
            return None
        
        # Look for phase-specific checkpoints
        pattern = f"{phase_name}*.ckpt"
        checkpoints = list(checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Sort by modification time, return latest
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def get_training_summary(self):
        """Get training summary"""
        return {
            'total_phases': len(self.config['phases']),
            'completed_phases': len(self.state['completed_phases']),
            'current_phase': self.state['current_phase'],
            'is_complete': self.is_complete(),
            'checkpoints': self.state['phase_checkpoints'],
            'metrics': self.state['best_metrics']
        }