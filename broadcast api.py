
#    path('update-contact-group/', UpdateContactGroupRawView.as_view(), name='update-contact-group'),

###PAYLOAD####
{
  "id": 1,
  "assigned_agent": "Agent007",
  "scheduled_date": "2025-05-12T10:30:00"
}
###


def post(self, request):
        contact_id = request.data.get('id')
        assigned_agent = request.data.get('assigned_agent')
        scheduled_date = request.data.get('scheduled_date')  # Should be ISO 8601 format

        if not all([contact_id, assigned_agent, scheduled_date]):
            return Response({'error': 'id, assigned_agent, and scheduled_date are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with connection.cursor() as cursor:
                query = """
                UPDATE contact_groups
                SET assigned_agent = %s, scheduled_date = %s
                WHERE id = %s
                """
                cursor.execute(query, [assigned_agent, scheduled_date, contact_id])
            return Response({'message': f'Contact group with id={contact_id} updated successfully.'})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
