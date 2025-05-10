@csrf_exempt
def get_chat_agents_by_user(request, user_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chat_agent_flow_config
                WHERE user_id = %s
            """, [user_id])
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(results, safe=False)


@csrf_exempt
def get_chat_history_by_user_agent(request, user_id, agent_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chat_history
                WHERE user_id = %s AND agent_id = %s
            """, [user_id, agent_id])
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(results, safe=False)



@csrf_exempt
def get_contacts_by_agent_user(request, agent_id, user_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chat_contacts
                WHERE agent_id = %s AND user_id = %s
            """, [agent_id, user_id])
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(results, safe=False)




@csrf_exempt
def get_chat_history_by_contact(request, contact_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chat_history
                WHERE contact_id = %s
            """, [contact_id])
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(results, safe=False)


@csrf_exempt
def get_contacts_by_agent_and_user(request, agent_id, user_id):
    if request.method == "GET":
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chat_contacts
                WHERE agent_id = %s AND user_id = %s
            """, [agent_id, user_id])
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return JsonResponse(results, safe=False)


@csrf_exempt
def delete_agent_by_id(request, agent_id):
    if request.method == "DELETE":
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chat_agent_flow_config WHERE id = %s", [agent_id])
        return JsonResponse({"message": "Agent deleted successfully"})




