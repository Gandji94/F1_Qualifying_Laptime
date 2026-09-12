{% test direction_circut(model,column_name) %}
select *
from {{ model }}
where {{ column_name }} not in ('Anti-clockwise', 'Clockwise', 'Figure-eight')
{% endtest %}