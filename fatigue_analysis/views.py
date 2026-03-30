"""
Views for fatigue analysis and ML model integration.
"""

import json
import sys
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render

# Add ml_models to path BEFORE attempting import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'ml_models'))
try:
    from fatigue_predictor import predict_fatigue, get_recommendations
    print("✅ ML model loaded successfully")
except ImportError as e:
    print("❌ ML model import failed:", e)
    def predict_fatigue(typing_data=None, mouse_data=None, facial_data=None):
        return {
            'combined_fatigue_score': 45,
            'individual_scores': {'typing': 45, 'mouse': 45, 'facial': 45},
            'confidence': 0.5,
            'data_quality': {'typing_available': False, 'mouse_available': False, 'facial_available': False}
        }
    def get_recommendations(fatigue_score, individual_scores=None):
        return {
            'recommendations': [{'type': 'Break', 'action': 'Take a break', 'description': 'Rest for a while', 'duration': '10 minutes', 'priority': 'medium'}],
            'insights': ['No ML model available'],
            'fatigue_level': 'Moderate',
            'timestamp': '2024-01-01T00:00:00'
        }


@csrf_exempt
@require_http_methods(["POST"])
def analyze_fatigue(request):
    try:
        data = json.loads(request.body)

        typing_data = data.get('typing_data')
        mouse_data  = data.get('mouse_data')
        facial_data = data.get('facial_data')

        print("📥 Received data:", json.dumps({
            'typing': typing_data,
            'mouse':  mouse_data,
            'facial': facial_data
        }, indent=2))

        prediction_result = predict_fatigue(typing_data, mouse_data, facial_data)
        print("📊 Prediction result:", prediction_result)

        recommendations = get_recommendations(
            prediction_result['combined_fatigue_score'],
            prediction_result['individual_scores']
        )

        response_data = {
            'status': 'success',
            'fatigue_analysis': prediction_result,
            'recommendations': recommendations,
            'message': 'Fatigue analysis completed successfully'
        }

        return JsonResponse(response_data)

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON data'}, status=400)
    except Exception as e:
        import traceback
        print("💥 Error:", traceback.format_exc())
        return JsonResponse({'status': 'error', 'message': f'Analysis failed: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_fatigue_insights(request):
    try:
        fatigue_score = float(request.GET.get('fatigue_score', 45))

        individual_scores = {}
        for key in ['typing_score', 'mouse_score', 'facial_score']:
            val = request.GET.get(key)
            if val:
                individual_scores[key.replace('_score', '')] = float(val)

        recommendations = get_recommendations(fatigue_score, individual_scores or None)
        return JsonResponse({'status': 'success', 'recommendations': recommendations})

    except ValueError:
        return JsonResponse({'status': 'error', 'message': 'Invalid fatigue score format'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Failed to get insights: {str(e)}'}, status=500)


def dashboard_view(request):
    context = {'ml_enabled': True, 'default_fatigue_score': 45}
    return render(request, 'dashboard/index.html', context)