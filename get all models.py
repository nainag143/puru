import os
import re
import json
import traceback
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def get_all_models_view(request):
    if request.method == 'GET':
        try:
            # Establish DB connection
            conn = get_db_connection()
            cursor = conn.cursor()

            # Fetch all models
            cursor.execute("""
                SELECT id, model_name, url, library_details, thumbnail, model_type
                FROM models
            """)

            rows = cursor.fetchall()

            if not rows:
                cursor.close()
                conn.close()
                return JsonResponse({'status': 'error', 'message': 'No models found'}, status=404)

            models_data = []
            for row in rows:
                models_data.append({
                    "id": row[0],
                    "model_name": row[1],
                    "url": row[2],
                    "library_details": row[3],
                    "thumbnail": row[4],
                })

            # Close the cursor and connection
            cursor.close()
            conn.close()

            # Return the models data
            return JsonResponse({'status': 'success', 'models': models_data}, status=200)

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Only GET method allowed'}, status=405)
