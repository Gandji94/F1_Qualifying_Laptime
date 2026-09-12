with raw as (
    select *
    from {{ ref('2026_qualifying_predictions') }}
),
final as (
    select *
    from raw
)
select *
from final