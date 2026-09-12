with staging as (
    select *
    from {{ ref('stg_f1_qualifying_results') }}
),
final as (
    select *
    from staging
)
select * from final