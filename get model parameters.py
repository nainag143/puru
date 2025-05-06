@csrf_exempt
def get_model_parameters_by_id(request, model_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM bothub.model_parameters WHERE model_id = %s", [model_id])
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(rows, safe=False)
