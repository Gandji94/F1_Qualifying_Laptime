{% test race_or_street(model,column_name) %}

select
    *
from {{ model }}
where {{ column_name }} not in ('Race','Street')

{% endtest %}