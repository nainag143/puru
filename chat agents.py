################################     URLS    #######################################################


  path('agents/user/<int:user_id>/', views.get_agents_by_user, name='get_agents_by_user'),

    # Get chat history by user_id and agent_id
    path('chat-history/user/<int:user_id>/agent/<int:agent_id>/', views.get_chat_history_by_user_agent, name='get_chat_history_by_user_agent'),

    # Get contacts by agent_id and user_id
    path('contacts/agent/<int:agent_id>/user/<int:user_id>/', views.get_contacts_by_agent_user, name='get_contacts_by_agent_user'),

    # Get chat history by contact_id
    path('chat-history/contact/<int:contact_id>/', views.get_chat_history_by_contact, name='get_chat_history_by_contact'),

    # Get contacts by agent_id and user_id (alternate)
    path('contacts-by-agent-user/<int:agent_id>/<int:user_id>/', views.get_contacts_by_agent_and_user, name='get_contacts_by_agent_and_user'),

    # Delete agent by id
    path('agents/<int:agent_id>/delete/', views.delete_agent_by_id, name='delete_agent_by_id'),

##############################URLS##############################################################








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




