"""
Mental Fatigue Prediction ML Model
Fixed version: scores now vary meaningfully based on actual input data.
"""

import numpy as np
import json
from datetime import datetime
import os


class FatiguePredictor:
    def __init__(self):
        self.model_weights = {
            'typing': {'speed_weight': 0.3, 'error_weight': 0.4, 'pause_weight': 0.3},
            'mouse':  {'reaction_weight': 0.4, 'accuracy_weight': 0.3, 'score_weight': 0.3},
            'facial': {'blink_weight': 0.3, 'closure_weight': 0.4, 'expression_weight': 0.3},
        }

        # FIX 3: Tighter std values so real inputs produce a wider spread of z-scores.
        # Previous std values were too large — almost all inputs landed near z=0 → same score.
        self.normalization_params = {
            'typing_speed':    {'mean': 45, 'std': 10},   # was 15
            'error_rate':      {'mean': 4,  'std': 2},    # was 3
            'pause_frequency': {'mean': 3,  'std': 1.5},  # was 2
            'reaction_time':   {'mean': 550, 'std': 150}, # was 200
            'mouse_accuracy':  {'mean': 65, 'std': 15},   # was 20
            'mouse_score':     {'mean': 7,  'std': 3},    # was 4
            'blink_rate':      {'mean': 18, 'std': 5},    # was 8
            'eye_closure':     {'mean': 180, 'std': 60},  # was 100
        }

    def normalize_feature(self, value, feature_name):
        """Z-score normalization. Clamps to [-3, 3] to avoid extreme outliers."""
        params = self.normalization_params.get(feature_name, {'mean': 0, 'std': 1})
        z = (value - params['mean']) / params['std']
        return max(-3.0, min(3.0, z))

    def predict_typing_fatigue(self, typing_data):
        if not typing_data:
            return 45

        speed      = typing_data.get('typingSpeed', 45)
        error_rate = typing_data.get('errorRate', 4)
        pause_freq = typing_data.get('pauseFrequency', 3)

        norm_speed = self.normalize_feature(speed, 'typing_speed')
        norm_error = self.normalize_feature(error_rate, 'error_rate')
        norm_pause = self.normalize_feature(pause_freq, 'pause_frequency')

        # FIX 3: Increased multipliers so the signal range is 0-100, not 20-40.
        speed_fatigue = max(0, min(100, -norm_speed * 25 + 50))   # slow=high fatigue
        error_fatigue = max(0, min(100,  norm_error * 30 + 30))   # many errors=high fatigue
        pause_fatigue = max(0, min(100,  norm_pause * 20 + 30))   # many pauses=high fatigue

        w = self.model_weights['typing']
        score = (
            speed_fatigue * w['speed_weight'] +
            error_fatigue * w['error_weight'] +
            pause_fatigue * w['pause_weight']
        )
        return min(max(score, 0), 100)

    def predict_mouse_fatigue(self, mouse_data):
        if not mouse_data:
            return 45

        reaction_time = mouse_data.get('reactionTime', 550)
        accuracy      = mouse_data.get('accuracy', 65)
        score         = mouse_data.get('score', 7)

        norm_reaction = self.normalize_feature(reaction_time, 'reaction_time')
        norm_accuracy = self.normalize_feature(accuracy, 'mouse_accuracy')
        norm_score    = self.normalize_feature(score, 'mouse_score')

        reaction_fatigue = max(0, min(100,  norm_reaction * 30 + 40))  # slow=high fatigue
        accuracy_fatigue = max(0, min(100, -norm_accuracy * 25 + 40))  # low acc=high fatigue
        score_fatigue    = max(0, min(100, -norm_score    * 20 + 35))  # low score=high fatigue

        w = self.model_weights['mouse']
        fatigue_score = (
            reaction_fatigue * w['reaction_weight'] +
            accuracy_fatigue * w['accuracy_weight'] +
            score_fatigue    * w['score_weight']
        )
        return min(max(fatigue_score, 0), 100)

    def predict_facial_fatigue(self, facial_data):
        if not facial_data:
            return 45

        blink_rate  = facial_data.get('blinkRate', 18)
        eye_closure = facial_data.get('eyeClosure', 180)
        expression  = facial_data.get('expression', 'Neutral')

        norm_blink   = self.normalize_feature(blink_rate, 'blink_rate')
        norm_closure = self.normalize_feature(eye_closure, 'eye_closure')

        # Both too-low and too-high blink rates indicate fatigue
        blink_fatigue   = max(0, min(100, abs(norm_blink) * 20 + 15))
        closure_fatigue = max(0, min(100, norm_closure * 25 + 30))  # longer=more fatigue

        expression_fatigue = {
            'Tired':      60,
            'Distracted': 40,
            'Neutral':    20,
            'Focused':    10,
        }.get(expression, 20)

        w = self.model_weights['facial']
        fatigue_score = (
            blink_fatigue      * w['blink_weight'] +
            closure_fatigue    * w['closure_weight'] +
            expression_fatigue * w['expression_weight']
        )
        return min(max(fatigue_score, 0), 100)

    def predict_combined_fatigue(self, typing_data, mouse_data, facial_data):
        typing_fatigue = self.predict_typing_fatigue(typing_data)
        mouse_fatigue  = self.predict_mouse_fatigue(mouse_data)
        facial_fatigue = self.predict_facial_fatigue(facial_data)

        weights = []
        scores  = []

        if typing_data and typing_data.get('typingSpeed', 0) > 0:
            weights.append(0.35)
            scores.append(typing_fatigue)

        if mouse_data and mouse_data.get('score', 0) > 0:
            weights.append(0.35)
            scores.append(mouse_fatigue)

        if facial_data and facial_data.get('analyzed', False):
            weights.append(0.30)
            scores.append(facial_fatigue)

        if weights:
            total_weight       = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            combined_score     = sum(s * w for s, w in zip(scores, normalized_weights))
        else:
            combined_score = 45

        # Slight regression toward mean only when variance is very high
        score_variance = np.var(scores) if len(scores) > 1 else 0
        if score_variance > 400:
            combined_score = combined_score * 0.92 + 45 * 0.08

        return {
            'combined_fatigue_score': round(min(max(combined_score, 0), 100), 1),
            'individual_scores': {
                'typing': round(typing_fatigue, 1),
                'mouse':  round(mouse_fatigue, 1),
                'facial': round(facial_fatigue, 1),
            },
            'confidence': round(max(0, 1 - score_variance / 1000), 2),
            'data_quality': {
                'typing_available': bool(typing_data and typing_data.get('typingSpeed', 0) > 0),
                'mouse_available':  bool(mouse_data  and mouse_data.get('score', 0) > 0),
                'facial_available': bool(facial_data and facial_data.get('analyzed', False)),
            },
        }

    def get_recommendations(self, fatigue_score, individual_scores=None):
        recommendations = []
        insights = []

        if fatigue_score < 30:
            recommendations.append({
                'type': 'Continue', 'action': 'Keep working',
                'description': 'Your fatigue level is low. You can continue your current activity.',
                'duration': 'N/A', 'priority': 'low',
            })
        elif fatigue_score < 50:
            recommendations.append({
                'type': 'Short Break', 'action': 'Take a 5-minute break',
                'description': 'Stand up, stretch, and rest your eyes for a few minutes.',
                'duration': '5 minutes', 'priority': 'medium',
            })
        elif fatigue_score < 70:
            recommendations.append({
                'type': 'Break', 'action': 'Take a 15-minute break',
                'description': 'Step away from your screen and do some light physical activity.',
                'duration': '15 minutes', 'priority': 'high',
            })
        else:
            recommendations.append({
                'type': 'Extended Break', 'action': 'Take a 30-minute break',
                'description': 'Consider a short nap, meditation, or light exercise.',
                'duration': '30 minutes', 'priority': 'urgent',
            })

        if individual_scores:
            if individual_scores.get('typing', 0) > 60:
                recommendations.append({
                    'type': 'Typing Rest', 'action': 'Rest your hands',
                    'description': 'Your typing patterns show signs of fatigue.',
                    'duration': '10 minutes', 'priority': 'medium',
                })
                insights.append('High typing fatigue detected — consider ergonomic improvements')

            if individual_scores.get('mouse', 0) > 60:
                recommendations.append({
                    'type': 'Hand Rest', 'action': 'Rest your mouse hand',
                    'description': 'Your mouse reaction time and accuracy show signs of fatigue.',
                    'duration': '5 minutes', 'priority': 'medium',
                })
                insights.append('Mouse performance degradation detected')

            if individual_scores.get('facial', 0) > 60:
                recommendations.append({
                    'type': 'Eye Rest', 'action': 'Rest your eyes',
                    'description': 'Your facial analysis shows signs of visual fatigue.',
                    'duration': '10 minutes', 'priority': 'high',
                })
                insights.append('Visual fatigue detected — follow the 20-20-20 rule')

        return {
            'recommendations': recommendations,
            'insights': insights,
            'fatigue_level': self._get_fatigue_level_text(fatigue_score),
            'timestamp': datetime.now().isoformat(),
        }

    def _get_fatigue_level_text(self, score):
        if score < 30:  return 'Low'
        if score < 50:  return 'Moderate'
        if score < 70:  return 'High'
        return 'Severe'

    def save_session_data(self, session_data, filename=None):
        if not filename:
            filename = f'session_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('data/sessions', exist_ok=True)
        filepath = os.path.join('data/sessions', filename)
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        return filepath


# Global model instance
fatigue_model = FatiguePredictor()


def predict_fatigue(typing_data=None, mouse_data=None, facial_data=None):
    return fatigue_model.predict_combined_fatigue(typing_data, mouse_data, facial_data)


def get_recommendations(fatigue_score, individual_scores=None):
    return fatigue_model.get_recommendations(fatigue_score, individual_scores)
