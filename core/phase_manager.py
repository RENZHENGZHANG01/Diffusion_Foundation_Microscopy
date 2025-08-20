"""
Phase Management for Automated Training Pipeline
==============================================
Handles phase transitions, checkpoints, and state persistence
"""

import json
import os
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
        """Load or create phase state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'current_phase': 0,
            'completed_phases': [],
            'phase_checkpoints': {},
            'best_metrics': {},
            'training_logs': []
        }
    
    def save_state(self):
        """Save current state"""
        self.save_path.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
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