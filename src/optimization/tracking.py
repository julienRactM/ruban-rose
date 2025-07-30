"""
Optimization Tracking and Storage System
Provides comprehensive tracking, storage, and analysis of hyperparameter optimization results
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import pickle
import optuna


@dataclass
class OptimizationResult:
    """Single optimization trial result"""
    trial_id: int
    model_type: str
    parameters: Dict[str, Any]
    f1_score: float
    sensitivity: float
    specificity: float
    accuracy: float
    mcc: float = 0.0  # Matthews Correlation Coefficient
    training_time: float = 0.0
    epochs_completed: int = 0
    timestamp: str = ""
    study_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationSummary:
    """Summary of optimization study"""
    study_name: str
    model_type: str
    total_trials: int
    completed_trials: int
    best_f1_score: float
    best_parameters: Dict[str, Any]
    optimization_time: float
    start_time: str
    end_time: str
    convergence_trial: int


class OptimizationTracker:
    """
    Tracks optimization progress and stores results
    """
    
    def __init__(self, storage_dir: str = "optimization_studies"):
        """
        Initialize optimization tracker
        
        Args:
            storage_dir: Directory to store optimization data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("OptimizationTracker")
        
        # Initialize database
        self.db_path = self.storage_dir / "optimization_results.db"
        self._init_database()
        
        # Current tracking state
        self.current_study: Optional[str] = None
        self.current_results: List[OptimizationResult] = []
        
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trials table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                study_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                parameters TEXT NOT NULL,
                f1_score REAL NOT NULL,
                sensitivity REAL NOT NULL,
                specificity REAL NOT NULL,
                accuracy REAL NOT NULL,
                mcc REAL DEFAULT 0.0,
                training_time REAL NOT NULL,
                epochs_completed INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(trial_id, study_name)
            )
        ''')
        
        # Add MCC column if it doesn't exist (for backward compatibility)
        cursor.execute("PRAGMA table_info(trials)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'mcc' not in columns:
            cursor.execute('ALTER TABLE trials ADD COLUMN mcc REAL DEFAULT 0.0')
        
        # Create studies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_name TEXT UNIQUE NOT NULL,
                model_type TEXT NOT NULL,
                total_trials INTEGER NOT NULL,
                completed_trials INTEGER NOT NULL,
                best_f1_score REAL NOT NULL,
                best_parameters TEXT NOT NULL,
                optimization_time REAL NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                convergence_trial INTEGER,
                status TEXT NOT NULL DEFAULT 'running'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_study(self, study_name: str, model_type: str):
        """Start tracking a new optimization study"""
        self.current_study = study_name
        self.current_results = []
        
        # Record study start
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO studies 
            (study_name, model_type, total_trials, completed_trials, 
             best_f1_score, best_parameters, optimization_time, start_time, status)
            VALUES (?, ?, 0, 0, 0.0, '{}', 0.0, ?, 'running')
        ''', (study_name, model_type, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Started tracking study: {study_name} for {model_type}")
    
    def record_trial(self, result: OptimizationResult):
        """Record a single trial result"""
        if self.current_study is None:
            raise ValueError("No active study. Call start_study() first.")
        
        self.current_results.append(result)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trials 
            (trial_id, study_name, model_type, parameters, f1_score, 
             sensitivity, specificity, accuracy, mcc, training_time, 
             epochs_completed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.trial_id, result.study_name, result.model_type, 
            json.dumps(result.parameters), result.f1_score,
            result.sensitivity, result.specificity, result.accuracy, result.mcc,
            result.training_time, result.epochs_completed, result.timestamp
        ))
        
        conn.commit()
        conn.close()
        
        # Update study progress
        self._update_study_progress()
    
    def _update_study_progress(self):
        """Update study progress in database"""
        if not self.current_results:
            return
        
        # Calculate current best
        best_result = max(self.current_results, key=lambda x: x.f1_score)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE studies 
            SET completed_trials = ?, best_f1_score = ?, best_parameters = ?
            WHERE study_name = ?
        ''', (
            len(self.current_results),
            best_result.f1_score,
            json.dumps(best_result.parameters),
            self.current_study
        ))
        
        conn.commit()
        conn.close()
    
    def finish_study(self, total_trials: int, optimization_time: float):
        """Mark study as completed"""
        if self.current_study is None:
            return
        
        # Find convergence point (where best score stopped improving significantly)
        convergence_trial = self._find_convergence_point()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE studies 
            SET total_trials = ?, optimization_time = ?, end_time = ?, 
                convergence_trial = ?, status = 'completed'
            WHERE study_name = ?
        ''', (
            total_trials, optimization_time, datetime.now().isoformat(),
            convergence_trial, self.current_study
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Finished study: {self.current_study}")
        self.current_study = None
    
    def _find_convergence_point(self) -> int:
        """Find the trial where the study converged (stopped improving)"""
        if len(self.current_results) < 10:
            return len(self.current_results)
        
        # Look for plateau in best scores
        best_scores = []
        current_best = 0.0
        
        for result in self.current_results:
            if result.f1_score > current_best:
                current_best = result.f1_score
            best_scores.append(current_best)
        
        # Find last significant improvement (> 0.01 F1 improvement)
        convergence_point = len(best_scores)
        improvement_threshold = 0.01
        
        for i in range(10, len(best_scores)):
            recent_improvement = best_scores[i] - best_scores[i-10]
            if recent_improvement < improvement_threshold:
                convergence_point = i
                break
        
        return convergence_point
    
    def get_study_progress(self, study_name: str) -> Dict[str, Any]:
        """Get current progress of a study"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM studies WHERE study_name = ?
        ''', (study_name,))
        
        study_data = cursor.fetchone()
        
        if study_data:
            columns = [desc[0] for desc in cursor.description]
            study_dict = dict(zip(columns, study_data))
            study_dict['best_parameters'] = json.loads(study_dict['best_parameters'])
            
            # Get recent trials
            cursor.execute('''
                SELECT * FROM trials WHERE study_name = ? 
                ORDER BY trial_id DESC LIMIT 10
            ''', (study_name,))
            
            recent_trials = cursor.fetchall()
            trial_columns = [desc[0] for desc in cursor.description]
            
            study_dict['recent_trials'] = [
                dict(zip(trial_columns, trial)) for trial in recent_trials
            ]
        else:
            study_dict = None
        
        conn.close()
        return study_dict
    
    def get_all_studies(self) -> List[Dict[str, Any]]:
        """Get all optimization studies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM studies ORDER BY start_time DESC')
        studies = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        studies_list = []
        for study in studies:
            study_dict = dict(zip(columns, study))
            study_dict['best_parameters'] = json.loads(study_dict['best_parameters'])
            studies_list.append(study_dict)
        
        conn.close()
        return studies_list
    
    def export_study_results(self, study_name: str, format: str = 'json') -> str:
        """Export study results to file"""
        conn = sqlite3.connect(self.db_path)
        
        # Get study info
        study_df = pd.read_sql_query(
            'SELECT * FROM studies WHERE study_name = ?', 
            conn, params=(study_name,)
        )
        
        # Get trial results
        trials_df = pd.read_sql_query(
            'SELECT * FROM trials WHERE study_name = ? ORDER BY trial_id', 
            conn, params=(study_name,)
        )
        
        conn.close()
        
        if study_df.empty:
            raise ValueError(f"Study {study_name} not found")
        
        # Prepare export data
        export_data = {
            'study_info': study_df.iloc[0].to_dict(),
            'trials': trials_df.to_dict('records')
        }
        
        # Export to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"{study_name}_results_{timestamp}.json"
            filepath = self.storage_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format == 'csv':
            filename = f"{study_name}_trials_{timestamp}.csv"
            filepath = self.storage_dir / filename
            trials_df.to_csv(filepath, index=False)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(filepath)


class OptimizationAnalyzer:
    """
    Analyzes optimization results and provides insights
    """
    
    def __init__(self, tracker: OptimizationTracker):
        self.tracker = tracker
    
    def analyze_parameter_importance(self, study_name: str) -> Dict[str, float]:
        """
        Analyze importance of different parameters
        
        Args:
            study_name: Name of the study to analyze
            
        Returns:
            Dictionary of parameter names and their importance scores
        """
        conn = sqlite3.connect(self.tracker.db_path)
        
        # Get trial data
        trials_df = pd.read_sql_query(
            'SELECT parameters, f1_score FROM trials WHERE study_name = ?', 
            conn, params=(study_name,)
        )
        
        conn.close()
        
        if trials_df.empty:
            return {}
        
        # Parse parameters
        param_data = []
        for _, row in trials_df.iterrows():
            params = json.loads(row['parameters'])
            params['f1_score'] = row['f1_score']
            param_data.append(params)
        
        param_df = pd.DataFrame(param_data)
        
        # Calculate correlation with F1 score
        importance_scores = {}
        for col in param_df.columns:
            if col != 'f1_score':
                try:
                    # Handle categorical parameters
                    if param_df[col].dtype == 'object':
                        # Use one-hot encoding for categorical
                        encoded = pd.get_dummies(param_df[col])
                        for encoded_col in encoded.columns:
                            corr = encoded[encoded_col].corr(param_df['f1_score'])
                            importance_scores[f"{col}_{encoded_col}"] = abs(corr) if not np.isnan(corr) else 0.0
                    else:
                        # Direct correlation for numerical
                        corr = param_df[col].corr(param_df['f1_score'])
                        importance_scores[col] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    importance_scores[col] = 0.0
        
        # Sort by importance
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def get_optimization_insights(self, study_name: str) -> Dict[str, Any]:
        """
        Get comprehensive insights about an optimization study
        
        Args:
            study_name: Name of the study to analyze
            
        Returns:
            Dictionary containing various insights
        """
        study_data = self.tracker.get_study_progress(study_name)
        
        if not study_data:
            return {}
        
        conn = sqlite3.connect(self.tracker.db_path)
        trials_df = pd.read_sql_query(
            'SELECT * FROM trials WHERE study_name = ? ORDER BY trial_id', 
            conn, params=(study_name,)
        )
        conn.close()
        
        insights = {
            'study_summary': {
                'total_trials': study_data['completed_trials'],
                'best_f1_score': study_data['best_f1_score'],
                'optimization_time': study_data['optimization_time'],
                'convergence_trial': study_data['convergence_trial']
            }
        }
        
        if not trials_df.empty:
            # Performance trends
            insights['performance_trends'] = {
                'f1_scores': trials_df['f1_score'].tolist(),
                'sensitivity_scores': trials_df['sensitivity'].tolist(),
                'specificity_scores': trials_df['specificity'].tolist(),
                'best_trial_id': trials_df.loc[trials_df['f1_score'].idxmax(), 'trial_id']
            }
            
            # Parameter importance
            insights['parameter_importance'] = self.analyze_parameter_importance(study_name)
            
            # Training efficiency
            insights['training_efficiency'] = {
                'avg_training_time': trials_df['training_time'].mean(),
                'avg_epochs_completed': trials_df['epochs_completed'].mean(),
                'fastest_trial': trials_df.loc[trials_df['training_time'].idxmin()].to_dict(),
                'most_efficient_trial': trials_df.loc[
                    (trials_df['f1_score'] / trials_df['training_time']).idxmax()
                ].to_dict()
            }
            
            # Medical metrics analysis
            medical_trials = trials_df[
                (trials_df['sensitivity'] >= 0.8) & (trials_df['specificity'] >= 0.8)
            ]
            
            insights['medical_performance'] = {
                'high_performance_trials': len(medical_trials),
                'avg_sensitivity': trials_df['sensitivity'].mean(),
                'avg_specificity': trials_df['specificity'].mean(),
                'medical_acceptable_rate': len(medical_trials) / len(trials_df) * 100
            }
        
        return insights
    
    def recommend_next_parameters(self, study_name: str, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend parameter combinations for future trials
        
        Args:
            study_name: Name of the study
            n_suggestions: Number of parameter combinations to suggest
            
        Returns:
            List of recommended parameter combinations
        """
        # This would typically use more sophisticated methods like
        # Gaussian Process models or Tree-based models
        # For now, we'll use simple heuristics based on best trials
        
        conn = sqlite3.connect(self.tracker.db_path)
        trials_df = pd.read_sql_query(
            'SELECT parameters, f1_score FROM trials WHERE study_name = ? ORDER BY f1_score DESC LIMIT 10', 
            conn, params=(study_name,)
        )
        conn.close()
        
        if trials_df.empty:
            return []
        
        # Extract parameters from best trials
        best_params = []
        for _, row in trials_df.iterrows():
            params = json.loads(row['parameters'])
            best_params.append(params)
        
        # Simple recommendation: variations of best parameters
        recommendations = []
        if best_params:
            best_param = best_params[0]  # Best trial
            
            # Create variations
            for i in range(n_suggestions):
                variation = best_param.copy()
                
                # Add small random variations
                for key, value in variation.items():
                    if isinstance(value, float):
                        variation[key] = value * (1 + np.random.normal(0, 0.1))
                    elif isinstance(value, int) and key != 'epochs':
                        variation[key] = max(1, int(value * (1 + np.random.normal(0, 0.1))))
                
                recommendations.append(variation)
        
        return recommendations


# Global tracker instance
_global_tracker: Optional[OptimizationTracker] = None

def get_global_tracker() -> OptimizationTracker:
    """Get or create global optimization tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = OptimizationTracker()
    return _global_tracker