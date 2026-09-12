{% test circut_complexity(model,column_name) %}
select *
from {{ model }}
where {{ column_name }} not in ('Balanced', 'Flowing', 'Twisty', 'Very twisty')
{% endtest %}