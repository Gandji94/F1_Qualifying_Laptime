with staging as (
    select *
    from {{ ref('stg_f1_qualifying_prediction') }}
),
final as (
    select *
    from staging
)
select * from final