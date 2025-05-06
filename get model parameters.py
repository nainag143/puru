from auth_app.db_config import get_db_connection
import os
import re
import json
import traceback
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def update_or_create_model_parameters_view(request):
    if request.method == 'GET':
        # Get model_id from query parameters
        model_id = request.GET.get('model_id')

        if not model_id:
            return JsonResponse({'status': 'error', 'message': 'model_id query param is required'}, status=400)

        try:
            # Establish DB connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Retrieve model parameters for the given model_id
            cursor.execute("""
                SELECT id, parameter_name, input_type, select_values, model_parameters
                FROM model_parameters
                WHERE model_id = %s
            """, (model_id,))

            rows = cursor.fetchall()
            if not rows:
                cursor.close()
                conn.close()
                return JsonResponse({'status': 'error', 'message': 'No parameters found for this model_id'}, status=404)
                
            params_data = []
            for row in rows:
                try:
                    # Safely parse select_values if valid JSON or a Python list/dict
                    select_values = ast.literal_eval(row[3]) if row[3] else []
                except:
                    select_values = []

                params_data.append({
                    "id": row[0],
                    "parameter_name": row[1],
                    "input_type": row[2],
                    "select_values": select_values,
                    "model_parameters": row[4]  # No parsing for this field yet
                })

            # Close the cursor and connection
            cursor.close()
            conn.close()

            # Return the retrieved parameters
            return JsonResponse({'status': 'success', 'model_parameters': params_data}, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Only GET method allowed'}, status=405)
