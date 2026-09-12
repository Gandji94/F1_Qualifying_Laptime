{% test non_negative(model, column_name) %}

/*
    model => corresponding model
    column_name => column being checked
*/

select *
from {{ model }}
where {{ column_name }} < 0

{% endtest %}