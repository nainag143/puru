@csrf_exempt
def get_all_models(request):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM bothub.models")
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(rows, safe=False)
