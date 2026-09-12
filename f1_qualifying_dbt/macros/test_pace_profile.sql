{% test pace_profile(model,column_name) %}
select *
from {{ model }}
where {{ column_name }} not in ('Balanced-high speed', 'High-speed', 'Low-speed', 'Balanced','Very high-speed','Mixed')
{% endtest %}